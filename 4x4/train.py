"""
train.py — Training entry point for SafeGAT-iLLM.

Trains the GAT-DQN (GATQNetwork) on the 4x4 SUMO grid using the
FastGATDQNTrainer (vectorised batched updates).

Key design decisions
--------------------
- The SUMO environment is reset() between episodes rather than
  closed and recreated, avoiding repeated process spawns.
- The batch update uses tiled edge-index offsets so a single forward
  pass processes the entire replay batch, giving ~30-50× speedup vs.
  the naive per-sample loop.
- Checkpoints are saved every CHECKPOINT_FREQ episodes and at the end.

Run from the project root::

    python train.py

Produces
--------
- models/gat_dqn_ep{N}.pt   — periodic checkpoints
- models/gat_dqn_final.pt    — final model (loaded by run_safegat.py)
- log/                       — SUMO and training logs
"""

from __future__ import annotations

import os

import numpy as np
import torch
from loguru import logger

# ── Project imports ────────────────────────────────────────────────────────────
# NOTE: network/net_config.py and network/graph_builder.py must exist.
#       Copy them from iLLM-TSC2/network/ and adapt junction IDs if needed.
from network.net_config    import CONTROLLED_TLS, NUM_NODES, NUM_ACTIONS
from network.graph_builder import EDGE_INDEX

from envs.grid_env_wrapper  import make_grid_env
from training.gat_dqn_trainer import FastGATDQNTrainer
from utils.make_tsc_env     import make_env

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT       = os.path.dirname(os.path.abspath(__file__))
LOG_PATH    = os.path.join(_ROOT, "log")
MODEL_PATH  = os.path.join(_ROOT, "models")
SUMO_CFG    = os.path.join(_ROOT, "network", "4x4.sumocfg")

# ── Observation / network dimensions ──────────────────────────────────────────
OBS_DIM    = 8
HIDDEN_DIM = 64
GAT_HEADS  = 4

# ── Training hyperparameters ───────────────────────────────────────────────────
TOTAL_EPISODES  = 100      # total training episodes
MAX_STEPS       = 1800     # simulation seconds per episode (30-min episodes)
CHECKPOINT_FREQ = 25       # save checkpoint every N episodes

# DQN / replay hyperparameters
LR                  = 1e-3
GAMMA               = 0.95
EPSILON_START       = 1.0
EPSILON_END         = 0.05
EPSILON_DECAY_STEPS = 25_000   # reaches near-greedy by ~episode 20
BATCH_SIZE          = 64
TARGET_UPDATE_FREQ  = 500
WARMUP_STEPS        = 500
BUFFER_CAPACITY     = 50_000
GRAD_CLIP           = 10.0


def main() -> None:
    os.makedirs(LOG_PATH,   exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Junctions: {CONTROLLED_TLS}  |  device={device}  |  episodes={TOTAL_EPISODES}")

    # ── Build trainer ──────────────────────────────────────────────────────────
    trainer = FastGATDQNTrainer(
        node_feature_dim    = OBS_DIM,
        num_nodes           = NUM_NODES,
        num_actions         = NUM_ACTIONS,
        hidden_dim          = HIDDEN_DIM,
        gat_heads           = GAT_HEADS,
        lr                  = LR,
        gamma               = GAMMA,
        epsilon_start       = EPSILON_START,
        epsilon_end         = EPSILON_END,
        epsilon_decay_steps = EPSILON_DECAY_STEPS,
        batch_size          = BATCH_SIZE,
        target_update_freq  = TARGET_UPDATE_FREQ,
        warmup_steps        = WARMUP_STEPS,
        buffer_capacity     = BUFFER_CAPACITY,
        grad_clip           = GRAD_CLIP,
        device              = device,
    )
    trainer.edge_index = EDGE_INDEX.to(device)

    # ── Create env once and reuse across episodes (avoids repeated SUMO spawns) ─
    env = make_grid_env(
        make_single_env_fn = make_env,
        tls_ids            = CONTROLLED_TLS,
        sumo_cfg           = SUMO_CFG,
        num_seconds        = MAX_STEPS,
        use_gui            = False,
        log_file           = LOG_PATH,
        obs_dim            = OBS_DIM,
    )

    reward_history: list[float] = []

    # ── Training loop ──────────────────────────────────────────────────────────
    for episode in range(1, TOTAL_EPISODES + 1):
        logger.info(f"Episode {episode}/{TOTAL_EPISODES}  |  ε={trainer.epsilon:.4f}")

        obs  = env.reset()
        done = False
        ep_reward = np.zeros(NUM_NODES, dtype=np.float32)
        step = 0

        while not done:
            # 1. Select actions (epsilon-greedy)
            actions, _q_vals, attn = trainer.select_actions(obs)

            # 2. Step environment
            next_obs, rewards, done, _infos = env.step(actions)

            # 3. Store transition
            trainer.store_transition(
                obs          = obs,
                actions      = actions,
                rewards      = rewards,
                next_obs     = next_obs,
                dones        = np.full(NUM_NODES, float(done), dtype=np.float32),
                attn_weights = attn,
            )

            # 4. Update network
            loss = trainer.update()

            ep_reward += rewards
            obs  = next_obs
            step += 1

            if step % 200 == 0:
                loss_str = "warmup" if loss is None else f"{loss:.4f}"
                logger.info(
                    f"  step={step:>4}  |  mean_rew={rewards.mean():.3f}  |  loss={loss_str}"
                )

        # ── Episode summary ────────────────────────────────────────────────────
        ep_total = float(ep_reward.sum())
        reward_history.append(ep_total)
        logger.info(
            f"Episode {episode:>3} done  |  total={ep_total:.2f}  "
            f"|  mean={ep_reward.mean():.2f}  |  steps={step}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if episode % CHECKPOINT_FREQ == 0:
            ckpt_path = os.path.join(MODEL_PATH, f"gat_dqn_ep{episode}.pt")
            trainer.save(ckpt_path)
            logger.info(f"Checkpoint saved → {ckpt_path}")

    env.close()

    # ── Final save + summary ───────────────────────────────────────────────────
    final_path = os.path.join(MODEL_PATH, "gat_dqn_final.pt")
    trainer.save(final_path)
    logger.info(f"Final model saved → {final_path}")

    best_ep  = int(np.argmax(reward_history)) + 1
    best_rew = max(reward_history)
    logger.info(f"Training complete.  Best episode: {best_ep}  (reward={best_rew:.2f})")


if __name__ == "__main__":
    main()
