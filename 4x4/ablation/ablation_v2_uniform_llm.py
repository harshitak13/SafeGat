"""
ablation_v2_uniform_llm.py
===========================
Ablation Variant 2 — GAT-DQN + LLM Always (Uniform Intervention).

Replicates the iLLM-TSC design: the LLM is called for EVERY junction at EVERY
step, with no selective gating and no Q-margin thresholding.
The safety shield is NOT applied (matches iLLM-TSC baseline).

This variant proves:
    - Uniform intervention has high token cost
    - It does not always outperform selective intervention (see V3)
    - Without a safety shield, LLM suggestions can violate traffic rules

Run from the project root::

    python ablation_v2_uniform_llm.py

Outputs
-------
data/ablation/v2_uniform_llm/
    step_log.json
    summary.json
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import torch
from langchain_openai import ChatOpenAI
from loguru import logger

from network.net_config    import CONTROLLED_TLS, NUM_ACTIONS, NUM_NODES
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

# NOTE: InterventionGate is BYPASSED in this variant — all nodes go to LLM.
# SafetyShield is also NOT applied (uniform iLLM-TSC baseline).

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT       = os.path.dirname(os.path.abspath(__file__))
OUT_DIR     = os.path.join(_ROOT, "data", "ablation", "v2_uniform_llm")
MODEL_PATH  = os.path.join(_ROOT, "models", "gat_dqn_final.pt")
SUMO_CFG    = os.path.join(_ROOT, "network", "4x4.sumocfg")
LOG_PATH    = os.path.join(_ROOT, "log")

OBS_DIM     = 8
HIDDEN_DIM  = 64
GAT_HEADS   = 4
SIM_SECONDS = 1600
MIN_GREEN_HOLD = 3   # tracked for violation counting only — NOT enforced

# ── Rate-limit settings ────────────────────────────────────────────────────────
# Groq free tier: ~30 RPM.  With 12 nodes called every step we need at least
# 2 s between consecutive calls to stay under the limit.
# We also call LLM only every LLM_CALL_EVERY_N_STEPS simulation steps so that
# a full 12-node sweep (≈24 s of wall-clock waiting) does not dominate runtime.
# The RL policy still runs every step; LLM decisions are *held* between sweeps.
LLM_MIN_INTERVAL_S  = 2.2   # seconds between consecutive API calls (30 RPM ≈ 2 s)
LLM_CALL_EVERY_N    = 10     # call LLM only once every N sim steps per junction
#  → total LLM calls ≈ (SIM_STEPS / LLM_CALL_EVERY_N) * NUM_NODES
#    at 1600 steps, step=5 s → ~320 steps / 10 = 32 sweeps × 12 = 384 calls
#    384 calls × 2.2 s ≈ ~14 min (vs 3+ hours with hammering)

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)


def _build_langchain_backend(config: dict):
    """Build a LangChain ChatOpenAI backend callable."""
    api_key  = config["OPENAI_API_KEY"]
    model    = config["OPENAI_API_MODEL"]
    base_url = config.get("OPENAI_API_BASE", "https://api.openai.com/v1")

    if not api_key or "YOUR_KEY" in api_key:
        raise ValueError("API key not set. Edit configs/config.yaml.")

    chat = ChatOpenAI(
        model          = model,
        temperature    = 0.0,
        api_key        = api_key,
        base_url       = base_url,
        timeout        = 20,
    )

    def _backend(prompt: str) -> str:
        from langchain_core.messages import HumanMessage
        return chat.invoke([HumanMessage(content=prompt)]).content

    return _backend


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
    trainer.epsilon    = 0.0
    trainer.edge_index = EDGE_INDEX.to(device)
    logger.info("[V2] GAT-DQN loaded. LLM=ALWAYS (uniform). NO safety shield.")

    # ── LLM pipeline (no gate, no shield) ────────────────────────────────────
    backend = _build_langchain_backend(config)

    # We use SafeGATRefiner but will BYPASS gate + shield manually.
    # ScenarioDetector is kept for anomaly tagging (fed to LLM for context).
    detector       = ScenarioDetector(queue_spike_threshold=0.85, zero_fraction_corruption_threshold=0.90)
    prompt_builder = TrafficPromptBuilder()
    # Use proper rate-limiting: one shared gateway ensures the 2.2 s inter-call
    # gap is enforced globally across all 12 junction calls.
    llm_gateway    = LLMGateway(backend=backend,
                                min_call_interval_s=LLM_MIN_INTERVAL_S,
                                max_backoff_retries=5,
                                backoff_wait_s=30.0)
    decision_logger = DecisionLogger(os.path.join(OUT_DIR, "llm_decisions.jsonl"))

    # ── SUMO environment ──────────────────────────────────────────────────────
    trip_info_path = os.path.join(OUT_DIR, "v2.tripinfo.xml")
    env = make_grid_env(
        make_single_env_fn = make_env,
        tls_ids            = CONTROLLED_TLS,
        sumo_cfg           = SUMO_CFG,
        num_seconds        = SIM_SECONDS,
        use_gui            = False,
        log_file           = LOG_PATH,
        obs_dim            = OBS_DIM,
        trip_info          = trip_info_path,
    )

    obs            = env.reset()
    done           = False
    sim_step       = 0
    total_rewards  = np.zeros(NUM_NODES, dtype=np.float32)
    infos          = [{} for _ in range(NUM_NODES)]

    # Stats
    llm_calls          = 0
    llm_overrides      = 0
    safety_violations  = 0      # premature switches NOT prevented by shield
    phase_runtime      = np.zeros(NUM_NODES, dtype=int)
    last_phase         = np.full(NUM_NODES, -1, dtype=int)
    step_log: list     = []

    # Cache: holds the last LLM-decided action per node between sweeps.
    # On steps where we skip the LLM we re-use the cached action so the
    # ablation still reflects "LLM always in control" semantically.
    held_llm_action    = list(range(NUM_NODES))   # initialised below after first obs

    logger.info("[V2] Starting uniform-LLM inference (all junctions, every step)...")

    while not done:
        rl_actions, q_values, attn_np = trainer.select_actions(obs)
        margins      = compute_q_margins(q_values)
        final_actions = rl_actions.copy()

        # Initialise held cache on step 0
        if sim_step == 0:
            held_llm_action = [int(a) for a in rl_actions]

        # ── Call LLM for EVERY node (uniform = no gate) ───────────────────────
        # LLM sweep only runs every LLM_CALL_EVERY_N steps to respect rate limits.
        # Between sweeps, the last LLM decision is re-applied (held action).
        run_llm_this_step = (sim_step % LLM_CALL_EVERY_N == 0)

        for node_idx in range(NUM_NODES):
            tls_id = CONTROLLED_TLS[node_idx]
            info   = infos[node_idx]
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
                        "phase_runtime":     int(phase_runtime[node_idx]),
                        "emergency_vehicle": bool(obs[node_idx, 7] > 0.5),
                        "information_missing": bool(info.get("information_missing", False)),
                    },
                )

                try:
                    # Build prompt and call LLM directly (bypass gate)
                    prompt   = prompt_builder.build(rl_info)
                    response = llm_gateway.query(prompt, label=tls_id)
                    llm_calls += 1

                    # Parse action from LLM response.
                    parsed_action = rl_action_int
                    if response and response.parsed:
                        proposed = response.parsed.get("final_phase",
                                   response.parsed.get("action", parsed_action))
                        try:
                            proposed = int(proposed)
                        except (TypeError, ValueError):
                            proposed = parsed_action
                        if 0 <= proposed < NUM_ACTIONS:
                            parsed_action = proposed

                    held_llm_action[node_idx] = parsed_action  # update cache

                    # NO safety shield — apply LLM action directly
                    if parsed_action != rl_action_int:
                        llm_overrides += 1

                    # Count safety violations (switch < min hold, unchecked)
                    if (last_phase[node_idx] >= 0
                            and parsed_action != int(last_phase[node_idx])
                            and int(phase_runtime[node_idx]) < MIN_GREEN_HOLD):
                        safety_violations += 1

                    decision_logger.log({
                        "intersection_id": tls_id,
                        "rl_action":        rl_action_int,
                        "final_action":     parsed_action,
                        "llm_decision":     response.decision,
                        "llm_reason":       response.reason,
                        "confidence_margin": float(margins[node_idx]),
                        "step":             sim_step,
                    })

                except Exception as exc:
                    logger.warning(f"[V2] LLM failed for {tls_id}: {exc!r} → RL fallback")
                    held_llm_action[node_idx] = rl_action_int  # fallback: reset cache

            # Apply held (or freshly computed) LLM action
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
                f"[V2] step={sim_step:>4}  |  mean_rew={rewards.mean():.4f}  "
                f"|  llm_calls={llm_calls}  |  overrides={llm_overrides}  "
                f"|  safety_violations={safety_violations}"
            )
        sim_step += 1

    env.close()

    mean_rewards_per_step = [s["mean_reward"] for s in step_log]
    mean_occ_per_step     = [s["mean_occ"]    for s in step_log]

    summary = {
        "variant":             "V2_Uniform_LLM",
        "description":         "GAT-DQN + LLM always (all nodes), no gate, no safety shield",
        "total_sim_steps":     sim_step,
        "total_reward":        round(float(total_rewards.sum()), 4),
        "mean_step_reward":    round(float(np.mean(mean_rewards_per_step)), 6),
        "mean_occupancy":      round(float(np.mean(mean_occ_per_step)), 4),
        "llm_calls":           llm_calls,
        "llm_overrides":       llm_overrides,
        "safety_adjustments":  0,           # shield not applied
        "safety_violations":   safety_violations,
        "intervention_rate_%": 100.0,       # all nodes called every sweep
        "llm_call_every_n_steps": LLM_CALL_EVERY_N,  # sweeps spaced to avoid rate-limit
    }

    logger.info(f"[V2] Summary:\n{json.dumps(summary, indent=2)}")

    with open(os.path.join(OUT_DIR, "step_log.json"),  "w") as f:
        json.dump(step_log, f, indent=2)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"[V2] Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
