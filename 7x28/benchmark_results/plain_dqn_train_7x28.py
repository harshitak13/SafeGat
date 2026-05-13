"""
plain_dqn_train_7x28.py
========================
Trains a plain fully-connected DQN (no graph) on the 7×28 SUMO network.
This is the 7×28 counterpart of benchmark_results/plain_dqn_train.py.

Produces models/plain_dqn_7x28_final.pt which benchmark_results_7x28.py
uses for the "Plain DQN (no graph)" column of the 7×28 comparison table.

Key differences vs the 4×4 version
-------------------------------------
- SUMO config  : network/7x28.sumocfg   (you must have built this)
- NUM_NODES    : 196   (7 rows × 28 cols)
- HIDDEN_DIM   : 256   (wider MLP for larger observation space)
- TRAIN_EPISODES: 50   (same — 7×28 episodes take longer wall-clock)
- MAX_STEPS    : 3200  (same)
- SAVE_PATH    : models/plain_dqn_7x28_final.pt

Usage::

    python plain_dqn_train_7x28.py

"""

from __future__ import annotations

import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── 7×28 network config ───────────────────────────────────────────────────────
# Controlled TLS for the 7×28 grid: J{row}_{col}  (row 0-6, col 0-27)
CONTROLLED_TLS_7x28 = [
    f"J{r}_{c}" for r in range(7) for c in range(28)
]
NUM_NODES   = len(CONTROLLED_TLS_7x28)   # 196
NUM_ACTIONS = 4                           # same 4 phases as 4×4

# ── Hyper-parameters ──────────────────────────────────────────────────────────
OBS_DIM         = 8
HIDDEN_DIM      = 256       # wider than 4×4 (128) for larger network
TRAIN_EPISODES  = 50
MAX_STEPS       = 3200
BATCH_SIZE      = 128       # larger batch for 196-node network
REPLAY_CAPACITY = 100_000   # doubled capacity
LR              = 5e-4      # slightly lower lr for stability
GAMMA           = 0.99
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY       = 0.995
TARGET_UPDATE   = 200
SAVE_PATH       = ROOT / "models" / "plain_dqn_7x28_final.pt"
SUMO_CFG        = str(ROOT / "network" / "7x28.sumocfg")


# ─────────────────────────────────────────────────────────────────────────────
#  Model  (identical architecture to 4×4, wider hidden layer)
# ─────────────────────────────────────────────────────────────────────────────

class PlainDQN(nn.Module):
    """
    Shared per-junction MLP.  Each row of the (NUM_NODES, OBS_DIM) tensor is
    processed independently — no cross-junction message passing.
    For 7×28 the hidden layer is wider (256) to handle the larger network.
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = HIDDEN_DIM,
                 n_actions: int = NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (..., obs_dim)   — works for single node or batched nodes
        returns : (..., n_actions)
        """
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Replay buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, actions, rewards, next_obs, done):
        self.buf.append((
            obs.astype(np.float32),
            actions.astype(np.int64),
            rewards.astype(np.float32),
            next_obs.astype(np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, nobs, done = zip(*batch)
        return (
            torch.FloatTensor(np.stack(obs)),      # (B, N, obs_dim)
            torch.LongTensor(np.stack(act)),        # (B, N)
            torch.FloatTensor(np.stack(rew)),       # (B, N)
            torch.FloatTensor(np.stack(nobs)),      # (B, N, obs_dim)
            torch.FloatTensor(np.array(done)),      # (B,)
        )

    def __len__(self):
        return len(self.buf)


# ─────────────────────────────────────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────────────────────────────────────

class PlainDQNTrainer7x28:

    def __init__(self):
        self.online  = PlainDQN()
        self.target  = PlainDQN()
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt        = torch.optim.Adam(self.online.parameters(), lr=LR)
        self.buf        = ReplayBuffer(REPLAY_CAPACITY)
        self.epsilon    = EPS_START
        self.grad_steps = 0

    def select_actions(self, obs: np.ndarray) -> np.ndarray:
        """obs: (NUM_NODES, OBS_DIM)  →  actions: (NUM_NODES,)"""
        if random.random() < self.epsilon:
            return np.random.randint(NUM_ACTIONS, size=NUM_NODES)
        obs_t = torch.FloatTensor(obs)
        with torch.no_grad():
            q = self.online(obs_t)     # (N, num_actions)
        return q.argmax(dim=1).numpy()

    def update(self) -> Optional[float]:
        if len(self.buf) < BATCH_SIZE:
            return None

        obs, act, rew, nobs, done = self.buf.sample(BATCH_SIZE)
        B, N, D = obs.shape

        obs_flat  = obs.view(B * N, D)
        nobs_flat = nobs.view(B * N, D)
        act_flat  = act.view(B * N)
        rew_flat  = rew.view(B * N)
        done_flat = done.unsqueeze(1).expand(B, N).reshape(B * N)

        q_vals  = self.online(obs_flat)
        q_taken = q_vals.gather(1, act_flat.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next    = self.target(nobs_flat).max(dim=1).values
            td_target = rew_flat + GAMMA * q_next * (1.0 - done_flat)

        loss = F.smooth_l1_loss(q_taken, td_target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        self.grad_steps += 1
        if self.grad_steps % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online":  self.online.state_dict(),
            "target":  self.target.state_dict(),
            "epsilon": self.epsilon,
        }, str(path))
        print(f"  [save] {path}")

    def load(self, path: Path):
        ckpt = torch.load(str(path), map_location="cpu")
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.epsilon = ckpt.get("epsilon", EPS_END)
        print(f"  [load] {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train():
    from utils.make_tsc_env import make_env
    from envs.grid_env_wrapper import make_grid_env

    LOG_PATH = ROOT / "log"
    LOG_PATH.mkdir(parents=True, exist_ok=True)

    env = make_grid_env(
        make_single_env_fn = make_env,
        tls_ids            = CONTROLLED_TLS_7x28,
        sumo_cfg           = SUMO_CFG,
        num_seconds        = MAX_STEPS,
        use_gui            = False,
        log_file           = str(LOG_PATH),
        obs_dim            = OBS_DIM,
    )

    trainer = PlainDQNTrainer7x28()
    ep_rewards: List[float] = []

    print(f"Training Plain DQN (7×28, {NUM_NODES} nodes) for {TRAIN_EPISODES} episodes …")

    for ep in range(1, TRAIN_EPISODES + 1):
        obs = env.reset()
        ep_reward = 0.0
        losses: List[float] = []

        for step in range(MAX_STEPS):
            actions  = trainer.select_actions(obs)
            nobs, rewards, done, infos = env.step(actions)

            trainer.buf.push(obs, actions, rewards, nobs, done)
            loss = trainer.update()
            if loss is not None:
                losses.append(loss)

            ep_reward += float(rewards.mean())
            obs = nobs
            if done:
                break

        trainer.decay_epsilon()
        ep_rewards.append(ep_reward)
        mean_loss = np.mean(losses) if losses else 0.0

        print(f"  Ep {ep:3d}/{TRAIN_EPISODES}  "
              f"reward={ep_reward:9.2f}  "
              f"loss={mean_loss:.4f}  "
              f"ε={trainer.epsilon:.3f}")

        if ep % 10 == 0:
            trainer.save(SAVE_PATH.parent / f"plain_dqn_7x28_ep{ep:03d}.pt")

    trainer.save(SAVE_PATH)
    env.close()
    print(f"\nTraining complete.  Checkpoint saved to {SAVE_PATH}")
    print(f"Best episode reward: {max(ep_rewards):.2f}  "
          f"(ep {ep_rewards.index(max(ep_rewards)) + 1})")


if __name__ == "__main__":
    train()
