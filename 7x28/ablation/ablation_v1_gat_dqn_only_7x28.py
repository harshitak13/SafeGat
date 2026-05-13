"""
ablation_v1_gat_dqn_only_7x28.py
==================================
Ablation Variant 1 — GAT-DQN Only (no LLM, no safety shield), 7×28 grid.

Mirrors ablation/ablation_v1_gat_dqn_only.py exactly, with the
following 7×28 adaptations:
  - SUMO config : network/7x28.sumocfg
  - NUM_NODES   : 196
  - MODEL_PATH  : models/gat_dqn_best.pt   (your 7×28 checkpoint)
  - Output dir  : data/ablation_7x28/v1_gat_dqn_only/

Run from the project root::

    python ablation_v1_gat_dqn_only_7x28.py

Outputs
-------
data/ablation_7x28/v1_gat_dqn_only/
    step_log.json
    summary.json
    v1.tripinfo.xml   (written by SUMO)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from network.graph_builder import EDGE_INDEX        # must export 7×28 edge index
from envs.grid_env_wrapper import make_grid_env
from training.gat_dqn_trainer import FastGATDQNTrainer
from utils.make_tsc_env import make_env
from utils.readConfig   import read_config
from utils.margin       import compute_q_margins

# ── 7×28 network config ───────────────────────────────────────────────────────
CONTROLLED_TLS_7x28 = [
    f"J{r}_{c}" for r in range(7) for c in range(28)
]
NUM_NODES   = len(CONTROLLED_TLS_7x28)   # 196
NUM_ACTIONS = 4

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent
OUT_DIR     = _ROOT / "data" / "ablation_7x28" / "v1_gat_dqn_only"
MODEL_PATH  = str(_ROOT / "models" / "gat_dqn_best.pt")
SUMO_CFG    = str(_ROOT / "network" / "7x28.sumocfg")
LOG_PATH    = str(_ROOT / "log")

OBS_DIM     = 8
HIDDEN_DIM  = 64
GAT_HEADS   = 4
SIM_SECONDS = 1600

OUT_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)


def main() -> None:
    config = read_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = FastGATDQNTrainer(
        node_feature_dim = OBS_DIM,
        num_nodes        = NUM_NODES,
        num_actions      = NUM_ACTIONS,
        hidden_dim       = HIDDEN_DIM,
        gat_heads        = GAT_HEADS,
        device           = device,
    )
    trainer.load(MODEL_PATH)
    trainer.epsilon    = 0.0          # pure greedy
    trainer.edge_index = EDGE_INDEX.to(device)
    logger.info(f"[V1-7x28] GAT-DQN loaded. NO LLM, NO safety shield. nodes={NUM_NODES}")

    trip_info_path = str(OUT_DIR / "v1.tripinfo.xml")
    env = make_grid_env(
        make_single_env_fn = make_env,
        tls_ids            = CONTROLLED_TLS_7x28,
        sumo_cfg           = SUMO_CFG,
        num_seconds        = SIM_SECONDS,
        use_gui            = False,
        log_file           = LOG_PATH,
        obs_dim            = OBS_DIM,
        trip_info          = trip_info_path,
    )

    obs              = env.reset()
    done             = False
    sim_step         = 0
    total_rewards    = np.zeros(NUM_NODES, dtype=np.float32)
    step_log: list   = []

    MIN_GREEN_HOLD     = 3
    phase_runtime      = np.zeros(NUM_NODES, dtype=int)
    last_phase         = np.full(NUM_NODES, -1, dtype=int)
    safety_violations  = 0

    logger.info("[V1-7x28] Starting GAT-DQN only inference (7×28, no LLM, no shield)...")

    while not done:
        rl_actions, q_values, attn_np = trainer.select_actions(obs)
        margins = compute_q_margins(q_values)

        for i in range(NUM_NODES):
            if (last_phase[i] >= 0
                    and rl_actions[i] != last_phase[i]
                    and phase_runtime[i] < MIN_GREEN_HOLD):
                safety_violations += 1

        obs, rewards, done, infos = env.step(rl_actions)
        total_rewards += rewards

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
            "llm_calls":         0,
        })

        if sim_step % 100 == 0:
            logger.info(
                f"[V1-7x28] step={sim_step:>4}  |  mean_rew={rewards.mean():.4f}  "
                f"|  cumulative={total_rewards.sum():.2f}  "
                f"|  safety_violations={safety_violations}"
            )
        sim_step += 1

    env.close()

    mean_rewards_per_step = [s["mean_reward"] for s in step_log]
    mean_occ_per_step     = [s["mean_occ"]    for s in step_log]

    summary = {
        "variant":             "V1_GAT_DQN_Only_7x28",
        "description":         "Pure GAT-DQN (7×28), no LLM, no safety shield",
        "network":             "7x28",
        "num_nodes":           NUM_NODES,
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

    logger.info(f"[V1-7x28] Summary:\n{json.dumps(summary, indent=2)}")

    with open(OUT_DIR / "step_log.json", "w") as f:
        json.dump(step_log, f, indent=2)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"[V1-7x28] Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
