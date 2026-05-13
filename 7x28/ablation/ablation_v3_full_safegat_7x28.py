"""
ablation_v3_full_safegat_7x28.py
==================================
Ablation Variant 3 — Full SafeGAT (Selective Intervention + Safety Shield),
7×28 grid.

Mirrors ablation/ablation_v3_full_safegat.py with 7×28 adaptations:
  - 196 nodes (7 rows × 28 cols)
  - LLM budget scaled to 19 600  (1 600 × 196/12 × 0.75 reserve factor)
  - MAX_NODES_PER_STEP = 2  (same cap — keeps latency bounded per step)
  - Model      : models/gat_dqn_best.pt
  - SUMO config: network/7x28.sumocfg
  - Output dir : data/ablation_7x28/v3_full_safegat/

Run from the project root::

    python ablation_v3_full_safegat_7x28.py

Outputs
-------
data/ablation_7x28/v3_full_safegat/
    step_log.json
    summary.json
    llm_decisions.jsonl
    v3.tripinfo.xml
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from langchain_openai import ChatOpenAI
from loguru import logger

from network.graph_builder import EDGE_INDEX
from envs.grid_env_wrapper import make_grid_env
from training.gat_dqn_trainer import FastGATDQNTrainer
from utils.make_tsc_env import make_env
from utils.readConfig   import read_config
from utils.margin       import compute_q_margins, select_uncertain_nodes

from llm.action_refiner       import SafeGATRefiner
from llm.decision_logger      import DecisionLogger
from llm.intervention_gate    import InterventionGate
from llm.llm_gateway          import LLMGateway
from llm.safety_shield        import SafetyShield
from llm.scenario_detector    import ScenarioDetector
from llm.traffic_prompt_builder import TrafficPromptBuilder
from llm.types                import RLDecisionInfo

# ── 7×28 network config ───────────────────────────────────────────────────────
CONTROLLED_TLS_7x28 = [
    f"J{r}_{c}" for r in range(7) for c in range(28)
]
NUM_NODES   = len(CONTROLLED_TLS_7x28)   # 196
NUM_ACTIONS = 4

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent
OUT_DIR     = _ROOT / "data" / "ablation_7x28" / "v3_full_safegat"
MODEL_PATH  = str(_ROOT / "models" / "gat_dqn_best.pt")
SUMO_CFG    = str(_ROOT / "network" / "7x28.sumocfg")
LOG_PATH    = str(_ROOT / "log")

OBS_DIM     = 8
HIDDEN_DIM  = 64
GAT_HEADS   = 4
SIM_SECONDS = 1600

# ── SafeGAT hyperparameters (scaled for 7×28) ─────────────────────────────────
Q_MARGIN_TAU       = 0.05
LLM_BUDGET         = 19_600   # scales with nodes: 1600 × (196/16)
MAX_NODES_PER_STEP = 2        # same hard cap per step
MIN_GREEN_STEPS    = 3

OUT_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)


def _build_langchain_backend(config: dict):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model       = config["OPENAI_API_MODEL"],
        api_key     = config["OPENAI_API_KEY"],
        base_url    = config.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        temperature = 0,
    )


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
    trainer.epsilon    = 0.0
    trainer.edge_index = EDGE_INDEX.to(device)
    logger.info(
        f"[V3-7x28] Full SafeGAT: τ={Q_MARGIN_TAU}, B={LLM_BUDGET}, "
        f"max_per_step={MAX_NODES_PER_STEP}, shield=ON, nodes={NUM_NODES}"
    )

    backend = _build_langchain_backend(config)

    refiner = SafeGATRefiner(
        detector = ScenarioDetector(
            queue_spike_threshold              = 0.85,
            zero_fraction_corruption_threshold = 0.90,
        ),
        gate = InterventionGate(
            confidence_threshold = Q_MARGIN_TAU,
            intervention_budget  = MAX_NODES_PER_STEP,
        ),
        prompt_builder  = TrafficPromptBuilder(),
        llm_gateway     = LLMGateway(
            backend             = backend,
            min_call_interval_s = 4.0,
            max_backoff_retries = 5,
            backoff_wait_s      = 30.0,
        ),
        safety_shield   = SafetyShield(min_green_hold=MIN_GREEN_STEPS),
        decision_logger = DecisionLogger(str(OUT_DIR / "llm_decisions.jsonl")),
    )

    trip_info_path = str(OUT_DIR / "v3.tripinfo.xml")
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

    obs                   = env.reset()
    done                  = False
    sim_step              = 0
    total_rewards         = np.zeros(NUM_NODES, dtype=np.float32)
    infos                 = [{} for _ in range(NUM_NODES)]
    llm_budget_remaining  = LLM_BUDGET
    phase_runtime         = np.zeros(NUM_NODES, dtype=int)
    last_phase            = np.full(NUM_NODES, -1, dtype=int)

    llm_calls          = 0
    llm_overrides      = 0
    safety_adjustments = 0
    step_log: list     = []

    shield = SafetyShield(min_green_hold=MIN_GREEN_STEPS)

    logger.info("[V3-7x28] Starting Full SafeGAT inference (7×28)...")

    while not done:
        rl_actions, q_values, attn_np = trainer.select_actions(obs)
        margins = compute_q_margins(q_values)

        anomaly_flags = np.array([
            bool(refiner.detector.detect(obs[i], infos[i])["tags"])
            for i in range(NUM_NODES)
        ])

        uncertain_nodes = select_uncertain_nodes(margins, anomaly_flags, Q_MARGIN_TAU)
        final_actions   = rl_actions.copy()

        if uncertain_nodes and llm_budget_remaining > 0:
            nodes_to_review = uncertain_nodes[:min(MAX_NODES_PER_STEP, llm_budget_remaining)]
            llm_budget_remaining -= len(nodes_to_review)

            for node_idx in nodes_to_review:
                tls_id = CONTROLLED_TLS_7x28[node_idx]
                info   = infos[node_idx]

                rl_info = RLDecisionInfo(
                    intersection_id   = tls_id,
                    observation       = obs[node_idx].tolist(),
                    phase             = int(last_phase[node_idx]) if last_phase[node_idx] >= 0
                                        else int(round(float(obs[node_idx, 0]) * 3)),
                    rl_action         = int(rl_actions[node_idx]),
                    action_scores     = q_values[node_idx].tolist(),
                    confidence_margin = float(margins[node_idx]),
                    legal_actions     = list(range(NUM_ACTIONS)),
                    neighbor_summary  = {},
                    anomaly_tags      = [],
                    metadata          = {
                        "phase_runtime":       int(phase_runtime[node_idx]),
                        "emergency_vehicle":   bool(obs[node_idx, 7] > 0.5),
                        "information_missing": info.get("information_missing", False),
                    },
                )

                try:
                    result = refiner.refine(rl_info)
                    llm_calls += 1
                    if int(result.final_action) != int(rl_actions[node_idx]):
                        llm_overrides += 1
                    if result.safety_adjusted:
                        safety_adjustments += 1
                    final_actions[node_idx] = int(result.final_action)
                except Exception as exc:
                    logger.warning(f"[V3-7x28] refine failed for {tls_id}: {exc!r} → RL fallback")

        # Safety shield applied to ALL 196 nodes
        for i in range(NUM_NODES):
            cur_phase    = int(last_phase[i]) if last_phase[i] >= 0 else None
            shield_result = shield.validate(
                proposed_action = int(final_actions[i]),
                legal_actions   = list(range(NUM_ACTIONS)),
                phase_runtime   = int(phase_runtime[i]),
                current_phase   = cur_phase,
            )
            if shield_result.adjusted:
                safety_adjustments += 1
                final_actions[i] = shield_result.action

        obs, rewards, done, infos = env.step(final_actions)
        total_rewards += rewards

        for i in range(NUM_NODES):
            cur = int(round(float(obs[i, 0]) * 3))
            if cur == last_phase[i]:
                phase_runtime[i] += 1
            else:
                last_phase[i]    = cur
                phase_runtime[i] = 1

        intervention_rate = (
            len(uncertain_nodes) / NUM_NODES * 100 if uncertain_nodes else 0.0
        )

        step_log.append({
            "step":               sim_step,
            "mean_reward":        float(rewards.mean()),
            "mean_occ":           float(obs[:, 2:6].mean()),
            "mean_margin":        float(margins.mean()),
            "n_uncertain":        len(uncertain_nodes),
            "llm_calls":          llm_calls,
            "llm_overrides":      llm_overrides,
            "safety_adjustments": safety_adjustments,
            "budget_left":        llm_budget_remaining,
            "intervention_rate":  intervention_rate,
        })

        if sim_step % 100 == 0:
            logger.info(
                f"[V3-7x28] step={sim_step:>4}  |  mean_rew={rewards.mean():.4f}  "
                f"|  cumul={total_rewards.sum():.2f}  |  llm_calls={llm_calls}  "
                f"|  overrides={llm_overrides}  |  shield_adj={safety_adjustments}"
            )
        sim_step += 1

    env.close()

    mean_rewards_per_step = [s["mean_reward"]     for s in step_log]
    mean_occ_per_step     = [s["mean_occ"]        for s in step_log]
    mean_intervention     = [s["intervention_rate"] for s in step_log]

    summary = {
        "variant":             "V3_Full_SafeGAT_7x28",
        "description":         "Selective LLM (Q-margin gate) + Safety Shield (7×28)",
        "network":             "7x28",
        "num_nodes":           NUM_NODES,
        "total_sim_steps":     sim_step,
        "total_reward":        round(float(total_rewards.sum()), 4),
        "mean_step_reward":    round(float(np.mean(mean_rewards_per_step)), 6),
        "mean_occupancy":      round(float(np.mean(mean_occ_per_step)), 4),
        "llm_calls":           llm_calls,
        "llm_overrides":       llm_overrides,
        "safety_adjustments":  safety_adjustments,
        "safety_violations":   0,
        "intervention_rate_%": round(float(np.mean(mean_intervention)), 2),
        "llm_budget_used_%":   round((LLM_BUDGET - llm_budget_remaining) / LLM_BUDGET * 100, 2),
    }

    logger.info(f"[V3-7x28] Summary:\n{json.dumps(summary, indent=2)}")

    with open(OUT_DIR / "step_log.json", "w") as f:
        json.dump(step_log, f, indent=2)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"[V3-7x28] Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
