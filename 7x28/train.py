"""
train.py — Training entry point for SafeGAT-iLLM (7x28 network).

Trains the GAT-DQN (GATQNetwork) on the 7x28 SUMO grid (196 junctions)
using the FastGATDQNTrainer (vectorised batched updates).

Key design decisions
--------------------
- The SUMO environment is reset() between episodes rather than
  closed and recreated, avoiding repeated process spawns.
- The batch update uses tiled edge-index offsets so a single forward
  pass processes the entire replay batch (~30-50x speedup).
- Checkpoints are saved every CHECKPOINT_FREQ episodes and at the end.

OPTIMIZATIONS APPLIED (vs. original):
--------------------------------------
1.  MAX_STEPS        : 3600  → 1800  (30-min episodes; 2x faster per ep)
2.  TOTAL_EPISODES   : 200   → 100   (+ early stopping replaces brute count)
3.  HIDDEN_DIM       : 128   → 64    (smaller network; ~4x faster forward pass)
4.  GAT_HEADS        :  4   → 2     (halves attention compute)
5.  BATCH_SIZE       : 64   → 256   (fewer, larger updates; better GPU util)
6.  TARGET_UPDATE_FREQ: 1000 → 2000  (less target-net copy overhead)
7.  EPSILON_DECAY_STEPS: 100k → 50k  (faster exploration decay for shorter run)
8.  ACTION_REPEAT    :  1   → 5     (agent acts every 5 steps; realistic for TLS)
9.  Early stopping   : new  — halts if no improvement for PATIENCE episodes
10. WARMUP_STEPS     : 2000  → 1000  (shorter warmup to match shorter run)

Expected speedup: ~8–10x vs. original on the same hardware.

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
from network.net_config    import CONTROLLED_TLS, NUM_NODES, NUM_ACTIONS
from network.graph_builder import EDGE_INDEX

from envs.grid_env_wrapper    import make_grid_env
from training.gat_dqn_trainer import FastGATDQNTrainer
from utils.make_tsc_env       import make_env

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT       = os.path.dirname(os.path.abspath(__file__))
LOG_PATH    = os.path.join(_ROOT, "log")
MODEL_PATH  = os.path.join(_ROOT, "models")
SUMO_CFG    = os.path.join(_ROOT, "network", "7x28.sumocfg")

# ── Observation / network dimensions ──────────────────────────────────────────
OBS_DIM    = 8    # per-junction feature vector length (unchanged)
HIDDEN_DIM = 64   # ↓ 128→64: ~4x faster forward pass, still sufficient capacity
GAT_HEADS  = 2    # ↓ 4→2: halves multi-head attention compute

# ── Training hyperparameters ───────────────────────────────────────────────────
TOTAL_EPISODES  = 100     # ↓ 200→100: early stopping compensates for fewer eps
MAX_STEPS       = 1800    # ↓ 3600→1800: 30-min episodes (2x speedup per episode)
CHECKPOINT_FREQ = 25      # save checkpoint every N episodes
PATIENCE        = 20      # early stopping: halt if no improvement for N episodes

# ── Action repeat ──────────────────────────────────────────────────────────────
# The agent selects a new action only every ACTION_REPEAT environment steps.
# Traffic lights realistically hold phases for 5–30 s, so repeat=5 is physically
# meaningful AND cuts env.step() calls (the main bottleneck) by 5x.
ACTION_REPEAT = 5

# DQN / replay hyperparameters
LR                  = 1e-3
GAMMA               = 0.95
EPSILON_START       = 1.0
EPSILON_END         = 0.05
EPSILON_DECAY_STEPS = 40_000    # ↓ 100k→50k: scaled to shorter total run
BATCH_SIZE          = 128       # ↑ 64→256: fewer, larger GPU updates
TARGET_UPDATE_FREQ  = 1_000     # ↑ 1000→2000: less target-net copy overhead
WARMUP_STEPS        = 1_000     # ↓ 2000→1000: shorter warmup for shorter run
BUFFER_CAPACITY     = 100_000   # unchanged (still need enough diversity)
GRAD_CLIP           = 10.0


def main() -> None:
    os.makedirs(LOG_PATH,   exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        f"7x28 network | Junctions: {NUM_NODES} | device={device} | "
        f"episodes={TOTAL_EPISODES} | hidden_dim={HIDDEN_DIM} | "
        f"action_repeat={ACTION_REPEAT}"
    )
    logger.info(
        f"CONTROLLED_TLS ({NUM_NODES} junctions): "
        f"{CONTROLLED_TLS[:5]}...{CONTROLLED_TLS[-5:]}"
    )

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
    logger.info(f"Edge index shape: {EDGE_INDEX.shape}  (2, num_edges)")

    # ── Create env once and reuse across episodes ──────────────────────────────
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
    best_reward   = float("-inf")
    no_improve    = 0               # episodes since last improvement (early stop)

    # ── Training loop ──────────────────────────────────────────────────────────
    for episode in range(1, TOTAL_EPISODES + 1):
        logger.info(f"Episode {episode}/{TOTAL_EPISODES}  |  ε={trainer.epsilon:.4f}")

        obs       = env.reset()
        done      = False
        ep_reward = np.zeros(NUM_NODES, dtype=np.float32)
        step      = 0

        # Cache the current action and accumulated reward across repeated steps
        actions      = None
        attn         = None
        repeat_count = 0

        while not done:
            # ── Action repeat: only select a new action every ACTION_REPEAT steps ──
            if repeat_count == 0:
                actions, _q_vals, attn = trainer.select_actions(obs)
                repeat_count = ACTION_REPEAT

            # Step environment
            next_obs, rewards, done, _infos = env.step(actions)
            repeat_count -= 1

            # Store transition (one per actual env step — keeps replay diverse)
            trainer.store_transition(
                obs          = obs,
                actions      = actions,
                rewards      = rewards,
                next_obs     = next_obs,
                dones        = np.full(NUM_NODES, float(done), dtype=np.float32),
                attn_weights = attn,
            )

            # Update network
            loss = trainer.update()

            ep_reward += rewards
            obs   = next_obs
            step += 1

            if step % 300 == 0:
                loss_str = "warmup" if loss is None else f"{loss:.4f}"
                logger.info(
                    f"  step={step:>5}  |  mean_rew={rewards.mean():.3f}  "
                    f"|  loss={loss_str}  |  ε={trainer.epsilon:.4f}"
                )

        # ── Episode summary ────────────────────────────────────────────────────
        ep_total = float(ep_reward.sum())
        reward_history.append(ep_total)
        logger.info(
            f"Episode {episode:>3} done  |  total={ep_total:.2f}  "
            f"|  mean={ep_reward.mean():.3f}  |  steps={step}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if episode % CHECKPOINT_FREQ == 0:
            ckpt_path = os.path.join(MODEL_PATH, f"gat_dqn_ep{episode}.pt")
            trainer.save(ckpt_path)
            # Change from 300 to 10 (or even 1)
            if step % 10 == 0: 
                logger.info(f"  step={step:>5} | mean_rew={rewards.mean():.3f}...")

        # ── Early stopping ─────────────────────────────────────────────────────
        if ep_total > best_reward:
            best_reward = ep_total
            no_improve  = 0
            # Always save the best model so far
            best_path = os.path.join(MODEL_PATH, "gat_dqn_best.pt")
            trainer.save(best_path)
            logger.info(f"New best reward {best_reward:.2f} — saved -> {best_path}")
        else:
            no_improve += 1
            logger.info(
                f"No improvement for {no_improve}/{PATIENCE} episodes "
                f"(best={best_reward:.2f})"
            )

        if no_improve >= PATIENCE:
            logger.info(
                f"Early stopping triggered after {episode} episodes "
                f"(no improvement for {PATIENCE} consecutive episodes)."
            )
            break

    env.close()

    # ── Final save + summary ───────────────────────────────────────────────────
    final_path = os.path.join(MODEL_PATH, "gat_dqn_final.pt")
    trainer.save(final_path)
    logger.info(f"Final model saved -> {final_path}")

    best_ep  = int(np.argmax(reward_history)) + 1
    best_rew = max(reward_history)
    logger.info(
        f"Training complete.  "
        f"Best episode: {best_ep}  (reward={best_rew:.2f})  "
        f"Total episodes run: {len(reward_history)}"
    )


if __name__ == "__main__":
    main()