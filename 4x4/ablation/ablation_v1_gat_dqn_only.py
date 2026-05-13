"""
ablation_v1_gat_dqn_only.py
============================
Ablation Variant 1 — GAT-DQN Only (no LLM, no safety shield).

Baseline: The trained GAT-DQN acts with a pure greedy policy.
No LLM is called. No safety shield post-processes actions.
This measures raw RL performance without any hybrid augmentation.

Run from the project root::

    python ablation_v1_gat_dqn_only.py

Outputs
-------
data/ablation/v1_gat_dqn_only/
    step_log.json          — per-step metrics
    summary.json           — aggregate statistics
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from loguru import logger

from network.net_config    import CONTROLLED_TLS, NUM_ACTIONS, NUM_NODES
from network.graph_builder import EDGE_INDEX
from envs.grid_env_wrapper import make_grid_env
from training.gat_dqn_trainer import FastGATDQNTrainer
from utils.make_tsc_env import make_env
from utils.readConfig   import read_config
from utils.margin       import compute_q_margins

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT       = os.path.dirname(os.path.abspath(__file__))
OUT_DIR     = os.path.join(_ROOT, "data", "ablation", "v1_gat_dqn_only")
MODEL_PATH  = os.path.join(_ROOT, "models", "gat_dqn_final.pt")
SUMO_CFG    = os.path.join(_ROOT, "network", "4x4.sumocfg")
LOG_PATH    = os.path.join(_ROOT, "log")

OBS_DIM     = 8
HIDDEN_DIM  = 64
GAT_HEADS   = 4
SIM_SECONDS = 1600

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)


def main() -> None:
    config = read_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load trained GAT-DQN ──────────────────────────────────────────────────
    trainer = FastGATDQNTrainer(
        node_feature_dim = OBS_DIM,
        num_nodes        = NUM_NODES,
        num_actions      = NUM_ACTIONS,
        hidden_dim       = HIDDEN_DIM,
        gat_heads        = GAT_HEADS,
        device           = device,
    )
    trainer.load(MODEL_PATH)
    trainer.epsilon    = 0.0          # pure greedy — no exploration
    trainer.edge_index = EDGE_INDEX.to(device)
    logger.info(f"[V1] GAT-DQN loaded (ε=0, inference mode). NO LLM, NO safety shield.")

    # ── SUMO environment ──────────────────────────────────────────────────────
    trip_info_path = os.path.join(OUT_DIR, "v1.tripinfo.xml")
    env = make_grid_env(
        make_single_env_fn = make_env,
        tls_ids            = CONTROLLED_TLS,
        sumo_cfg           = SUMO_CFG,
        num_seconds        = SIM_SECONDS,
        use_gui            = False,     # headless for ablation speed
        log_file           = LOG_PATH,
        obs_dim            = OBS_DIM,
        trip_info          = trip_info_path,
    )

    obs           = env.reset()
    done          = False
    sim_step      = 0
    total_rewards = np.zeros(NUM_NODES, dtype=np.float32)
    step_log: list = []

    # ── Track safety violations manually (for fair comparison) ────────────────
    # Count how many times a phase switch would have happened < min_green_hold
    MIN_GREEN_HOLD = 3
    phase_runtime  = np.zeros(NUM_NODES, dtype=int)
    last_phase     = np.full(NUM_NODES, -1, dtype=int)
    safety_violations = 0

    logger.info("[V1] Starting GAT-DQN only inference (no LLM, no safety shield)...")

    while not done:
        # GAT-DQN greedy action selection
        rl_actions, q_values, attn_np = trainer.select_actions(obs)
        margins = compute_q_margins(q_values)

        # Count safety violations (premature phase switches)
        for i in range(NUM_NODES):
            cur_phase = int(round(float(obs[i, 0]) * 3))
            if (last_phase[i] >= 0
                    and rl_actions[i] != last_phase[i]
                    and phase_runtime[i] < MIN_GREEN_HOLD):
                safety_violations += 1

        # Execute RL actions directly — NO safety shield, NO LLM
        obs, rewards, done, infos = env.step(rl_actions)
        total_rewards += rewards

        # Update phase tracking
        for i in range(NUM_NODES):
            cur = int(round(float(obs[i, 0]) * 3))
            if cur == last_phase[i]:
                phase_runtime[i] += 1
            else:
                last_phase[i]    = cur
                phase_runtime[i] = 1

        step_log.append({
            "step":              sim_step,
            "mean_reward":       float(rewards.mean()),
            "mean_occ":          float(obs[:, 2:6].mean()),
            "mean_margin":       float(margins.mean()),
            "safety_violations": safety_violations,
            "llm_calls":         0,     # always 0 for this variant
        })

        if sim_step % 100 == 0:
            logger.info(
                f"[V1] step={sim_step:>4}  |  mean_rew={rewards.mean():.4f}  "
                f"|  cumulative={total_rewards.sum():.2f}  "
                f"|  safety_violations={safety_violations}"
            )
        sim_step += 1

    env.close()

    # ── Summary ────────────────────────────────────────────────────────────────
    mean_rewards_per_step = [s["mean_reward"] for s in step_log]
    mean_occ_per_step     = [s["mean_occ"]    for s in step_log]

    summary = {
        "variant":             "V1_GAT_DQN_Only",
        "description":         "Pure GAT-DQN, no LLM, no safety shield",
        "total_sim_steps":     sim_step,
        "total_reward":        round(float(total_rewards.sum()), 4),
        "mean_step_reward":    round(float(np.mean(mean_rewards_per_step)), 6),
        "mean_occupancy":      round(float(np.mean(mean_occ_per_step)), 4),
        "llm_calls":           0,
        "llm_overrides":       0,
        "safety_adjustments":  0,
        "safety_violations":   safety_violations,
        "intervention_rate_%": 0.0,
    }

    logger.info(f"[V1] Summary:\n{json.dumps(summary, indent=2)}")

    with open(os.path.join(OUT_DIR, "step_log.json"),  "w") as f:
        json.dump(step_log, f, indent=2)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"[V1] Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
