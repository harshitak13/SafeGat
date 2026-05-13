"""
ablation_v2_uniform_llm_7x28.py
================================
Ablation Variant 2 — GAT-DQN + LLM Always (Uniform Intervention), 7×28 grid.

Mirrors ablation/ablation_v2_uniform_llm.py with 7×28 adaptations:
  - 196 nodes → every-step LLM sweep is extremely expensive.
  - LLM_CALL_EVERY_N is increased to 20 (vs 10 for 4×4) to keep
    total API calls in a reasonable range.
  - Rate-limit gap stays at 2.2 s.
  - Output dir : data/ablation_7x28/v2_uniform_llm/

Run from the project root::

    python ablation_v2_uniform_llm_7x28.py

Outputs
-------
data/ablation_7x28/v2_uniform_llm/
    step_log.json
    summary.json
    llm_decisions.jsonl
    v2.tripinfo.xml
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
from utils.margin       import compute_q_margins

from llm.action_refiner       import SafeGATRefiner
from llm.decision_logger      import DecisionLogger
from llm.llm_gateway          import LLMGateway
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
OUT_DIR     = _ROOT / "data" / "ablation_7x28" / "v2_uniform_llm"
MODEL_PATH  = str(_ROOT / "models" / "gat_dqn_best.pt")
SUMO_CFG    = str(_ROOT / "network" / "7x28.sumocfg")
LOG_PATH    = str(_ROOT / "log")

OBS_DIM     = 8
HIDDEN_DIM  = 64
GAT_HEADS   = 4
SIM_SECONDS = 1600
MIN_GREEN_HOLD = 3

# ── Rate-limit settings ────────────────────────────────────────────────────────
# 196 nodes × every sweep → cost is enormous even with spacing.
# LLM_CALL_EVERY_N=20 keeps total sweeps ≈ 16  → 196 calls each → ~3 136 calls.
# At 2.2 s/call that's ~6 900 s wall-clock if serial; use batch+async in prod.
LLM_MIN_INTERVAL_S = 2.2
LLM_CALL_EVERY_N   = 20   # wider gap than 4×4 (10) due to 196 nodes

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
    logger.info(f"[V2-7x28] GAT-DQN loaded. LLM=ALWAYS (uniform). NO safety shield. nodes={NUM_NODES}")

    backend = _build_langchain_backend(config)

    detector        = ScenarioDetector(queue_spike_threshold=0.85,
                                       zero_fraction_corruption_threshold=0.90)
    prompt_builder  = TrafficPromptBuilder()
    llm_gateway     = LLMGateway(backend=backend,
                                 min_call_interval_s=LLM_MIN_INTERVAL_S,
                                 max_backoff_retries=5,
                                 backoff_wait_s=30.0)
    decision_logger = DecisionLogger(str(OUT_DIR / "llm_decisions.jsonl"))

    trip_info_path = str(OUT_DIR / "v2.tripinfo.xml")
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

    obs               = env.reset()
    done              = False
    sim_step          = 0
    total_rewards     = np.zeros(NUM_NODES, dtype=np.float32)
    infos             = [{} for _ in range(NUM_NODES)]

    llm_calls         = 0
    llm_overrides     = 0
    safety_violations = 0
    phase_runtime     = np.zeros(NUM_NODES, dtype=int)
    last_phase        = np.full(NUM_NODES, -1, dtype=int)
    step_log: list    = []
    held_llm_action   = list(range(NUM_NODES))

    logger.info("[V2-7x28] Starting uniform-LLM inference (all 196 junctions, every sweep)...")

    while not done:
        rl_actions, q_values, attn_np = trainer.select_actions(obs)
        margins       = compute_q_margins(q_values)
        final_actions = rl_actions.copy()

        if sim_step == 0:
            held_llm_action = [int(a) for a in rl_actions]

        run_llm_this_step = (sim_step % LLM_CALL_EVERY_N == 0)

        for node_idx in range(NUM_NODES):
            tls_id        = CONTROLLED_TLS_7x28[node_idx]
            info          = infos[node_idx]
            rl_action_int = int(rl_actions[node_idx])

            if run_llm_this_step:
                rl_info = RLDecisionInfo(
                    intersection_id   = tls_id,
                    observation       = [float(x) for x in obs[node_idx]],
                    phase             = int(last_phase[node_idx]) if last_phase[node_idx] >= 0
                                        else int(round(float(obs[node_idx, 0]) * 3)),
                    rl_action         = rl_action_int,
                    action_scores     = [float(x) for x in q_values[node_idx]],
                    confidence_margin = float(margins[node_idx]),
                    legal_actions     = [int(i) for i in range(NUM_ACTIONS)],
                    neighbor_summary  = {},
                    anomaly_tags      = [],
                    metadata          = {
                        "phase_runtime":       int(phase_runtime[node_idx]),
                        "emergency_vehicle":   bool(obs[node_idx, 7] > 0.5),
                        "information_missing": bool(info.get("information_missing", False)),
                    },
                )

                try:
                    prompt   = prompt_builder.build(rl_info)
                    response = llm_gateway.query(prompt, label=tls_id)
                    llm_calls += 1

                    parsed_action = rl_action_int
                    if response and response.parsed:
                        proposed = response.parsed.get(
                            "final_phase", response.parsed.get("action", parsed_action))
                        try:
                            proposed = int(proposed)
                        except (TypeError, ValueError):
                            proposed = parsed_action
                        if 0 <= proposed < NUM_ACTIONS:
                            parsed_action = proposed

                    held_llm_action[node_idx] = parsed_action

                    if parsed_action != rl_action_int:
                        llm_overrides += 1
                    if (last_phase[node_idx] >= 0
                            and parsed_action != int(last_phase[node_idx])
                            and int(phase_runtime[node_idx]) < MIN_GREEN_HOLD):
                        safety_violations += 1

                    decision_logger.log({
                        "intersection_id":  tls_id,
                        "rl_action":        rl_action_int,
                        "final_action":     parsed_action,
                        "llm_decision":     response.decision,
                        "llm_reason":       response.reason,
                        "confidence_margin": float(margins[node_idx]),
                        "step":             sim_step,
                    })

                except Exception as exc:
                    logger.warning(f"[V2-7x28] LLM failed for {tls_id}: {exc!r} → RL fallback")
                    held_llm_action[node_idx] = rl_action_int

            final_actions[node_idx] = held_llm_action[node_idx]

        obs, rewards, done, infos = env.step(final_actions)
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
            "llm_calls":         llm_calls,
            "llm_overrides":     llm_overrides,
            "safety_violations": safety_violations,
        })

        if sim_step % 50 == 0:
            logger.info(
                f"[V2-7x28] step={sim_step:>4}  |  mean_rew={rewards.mean():.4f}  "
                f"|  llm_calls={llm_calls}  |  overrides={llm_overrides}  "
                f"|  safety_violations={safety_violations}"
            )
        sim_step += 1

    env.close()

    summary = {
        "variant":             "V2_Uniform_LLM_7x28",
        "description":         "GAT-DQN + LLM always (all 196 nodes), no gate, no shield",
        "network":             "7x28",
        "num_nodes":           NUM_NODES,
        "total_sim_steps":     sim_step,
        "total_reward":        round(float(total_rewards.sum()), 4),
        "mean_step_reward":    round(float(np.mean([s["mean_reward"] for s in step_log])), 6),
        "mean_occupancy":      round(float(np.mean([s["mean_occ"]    for s in step_log])), 4),
        "llm_calls":           llm_calls,
        "llm_overrides":       llm_overrides,
        "safety_adjustments":  0,
        "safety_violations":   safety_violations,
        "intervention_rate_%": 100.0,
        "llm_call_every_n_steps": LLM_CALL_EVERY_N,
    }

    logger.info(f"[V2-7x28] Summary:\n{json.dumps(summary, indent=2)}")

    with open(OUT_DIR / "step_log.json", "w") as f:
        json.dump(step_log, f, indent=2)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"[V2-7x28] Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
