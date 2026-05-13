"""
plain_dqn_train.py
==================
Trains a plain fully-connected DQN (no graph) on the 4x4 SUMO network.
This produces models/plain_dqn_final.pt which benchmark_comparison.py loads
for the "Plain DQN (no graph)" column.

Usage
-----
    python plain_dqn_train.py

The network architecture is identical to GAT-DQN except the GATConv layers
are replaced by a shared Linear layer, so there is NO neighbourhood
information exchange — each junction reasons independently.

Training is deliberately kept short (TRAIN_EPISODES=50) so the ablation
runs in a comparable wall-clock time to the GAT-DQN training.  Increase
TRAIN_EPISODES for a fairer comparison.
"""

from __future__ import annotations

import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from network.net_config import CONTROLLED_TLS, NUM_NODES, NUM_ACTIONS

# ── Hyper-parameters ──────────────────────────────────────────────────────────
OBS_DIM         = 8
HIDDEN_DIM      = 128
TRAIN_EPISODES  = 50
MAX_STEPS       = 3200          # steps per episode (matches train.py)
BATCH_SIZE      = 64
REPLAY_CAPACITY = 50_000
LR              = 1e-3
GAMMA           = 0.99
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY       = 0.995
TARGET_UPDATE   = 200           # hard target update every N gradient steps
SAVE_PATH       = ROOT / "models" / "plain_dqn_final.pt"


# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────

class PlainDQN(nn.Module):
    """
    Shared per-junction MLP.  Each row of the (NUM_NODES, OBS_DIM) tensor is
    processed independently — no cross-junction message passing.
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
#  Replay buffer (reuses same structure as GAT-DQN)
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

class PlainDQNTrainer:

    def __init__(self):
        self.online  = PlainDQN()
        self.target  = PlainDQN()
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt     = torch.optim.Adam(self.online.parameters(), lr=LR)
        self.buf     = ReplayBuffer(REPLAY_CAPACITY)
        self.epsilon = EPS_START
        self.grad_steps = 0

    # ── action selection ──────────────────────────────────────────────────────

    def select_actions(self, obs: np.ndarray) -> np.ndarray:
        """obs: (NUM_NODES, OBS_DIM)  → actions: (NUM_NODES,)"""
        if random.random() < self.epsilon:
            return np.random.randint(NUM_ACTIONS, size=NUM_NODES)
        obs_t = torch.FloatTensor(obs)        # (N, obs_dim)
        with torch.no_grad():
            q = self.online(obs_t)            # (N, num_actions)
        return q.argmax(dim=1).numpy()

    # ── learning step ─────────────────────────────────────────────────────────

    def update(self):
        if len(self.buf) < BATCH_SIZE:
            return

        obs, act, rew, nobs, done = self.buf.sample(BATCH_SIZE)
        # shapes: obs (B, N, D),  act (B, N),  rew (B, N),  nobs (B, N, D), done (B,)

        B, N, D = obs.shape

        # Flatten for shared MLP: (B*N, D)
        obs_flat  = obs.view(B * N, D)
        nobs_flat = nobs.view(B * N, D)
        act_flat  = act.view(B * N)
        rew_flat  = rew.view(B * N)
        done_flat = done.unsqueeze(1).expand(B, N).reshape(B * N)

        # Q(s,a) for taken actions
        q_vals   = self.online(obs_flat)                          # (B*N, A)
        q_taken  = q_vals.gather(1, act_flat.unsqueeze(1)).squeeze(1)  # (B*N,)

        # Target
        with torch.no_grad():
            q_next   = self.target(nobs_flat).max(dim=1).values  # (B*N,)
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

    # ── epsilon decay ─────────────────────────────────────────────────────────

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    # ── save / load ───────────────────────────────────────────────────────────

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
        tls_ids            = CONTROLLED_TLS,
        sumo_cfg           = str(ROOT / "network" / "4x4.sumocfg"),
        num_seconds        = MAX_STEPS,
        use_gui            = False,
        log_file           = str(LOG_PATH),
        obs_dim            = OBS_DIM,
    )

    trainer = PlainDQNTrainer()
    ep_rewards: List[float] = []

    print(f"Training Plain DQN for {TRAIN_EPISODES} episodes …")

    for ep in range(1, TRAIN_EPISODES + 1):
        obs = env.reset()               # obs: (NUM_NODES, OBS_DIM)  — no tuple
        ep_reward = 0.0
        losses: List[float] = []

        for step in range(MAX_STEPS):
            actions  = trainer.select_actions(obs)
            nobs, rewards, done, infos = env.step(actions)  # 4-tuple, no truncated

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
              f"reward={ep_reward:7.2f}  "
              f"loss={mean_loss:.4f}  "
              f"ε={trainer.epsilon:.3f}")

        if ep % 10 == 0:
            trainer.save(SAVE_PATH.parent / f"plain_dqn_ep{ep:03d}.pt")

    trainer.save(SAVE_PATH)
    env.close()
    print(f"\nTraining complete.  Checkpoint saved to {SAVE_PATH}")


if __name__ == "__main__":
    train()
