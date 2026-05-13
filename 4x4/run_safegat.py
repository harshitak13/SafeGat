"""
run_safegat.py — Inference entry point for SafeGAT-iLLM.

Loads a trained GAT-DQN checkpoint and runs one inference episode with
the full SafeGAT-LLM pipeline:

    GAT-DQN proposes actions
        → Q-margin gating selects uncertain / anomalous nodes
        → SafeGATRefiner runs per flagged node:
              ScenarioDetector → InterventionGate → LLMGateway → SafetyShield
        → Final actions execute in SUMO
        → DecisionLogger records every refinement to JSONL

SafeGAT mechanisms
------------------
1. Uncertainty-aware gating   — LLM called only when Δ_i < τ (or anomaly)
2. Anomaly detection          — queue spike, NaN, emergency, packet-loss
3. Intervention budget        — hard cap B on total LLM calls per episode
4. Per-step node cap          — at most MAX_NODES_PER_STEP calls per step
5. Safety shield              — yellow-lock, min-green, illegal-action repair
6. Audit logging              — full JSONL trail for offline analysis

Run from the project root (after training)::

    python run_safegat.py

Requires
--------
- models/gat_dqn_final.pt
- configs/config.yaml with a valid API key
- network/ files (net_config.py, graph_builder.py, 4x4.sumocfg)
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from langchain_openai import ChatOpenAI
from loguru import logger

# ── Project imports ────────────────────────────────────────────────────────────
from network.net_config    import CONTROLLED_TLS, NUM_ACTIONS, NUM_NODES
from network.graph_builder import EDGE_INDEX

from envs.grid_env_wrapper    import make_grid_env
from training.gat_dqn_trainer import FastGATDQNTrainer

from llm.action_refiner       import SafeGATRefiner
from llm.decision_logger      import DecisionLogger
from llm.intervention_gate    import InterventionGate
from llm.llm_gateway          import LLMGateway
from llm.safety_shield        import SafetyShield
from llm.scenario_detector    import ScenarioDetector
from llm.traffic_prompt_builder import TrafficPromptBuilder
from llm.types                import RLDecisionInfo

from utils.make_tsc_env import make_env
from utils.readConfig   import read_config
from utils.margin       import compute_q_margins, select_uncertain_nodes

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT       = os.path.dirname(os.path.abspath(__file__))
LOG_PATH    = os.path.join(_ROOT, "log")
MODEL_PATH  = os.path.join(_ROOT, "models")
RESULT_PATH = os.path.join(_ROOT, "data", "output")
SUMO_CFG    = os.path.join(_ROOT, "network", "4x4.sumocfg")

# ── Observation / network dimensions ──────────────────────────────────────────
OBS_DIM    = 8
HIDDEN_DIM = 64
GAT_HEADS  = 4

# ── SafeGAT inference hyperparameters ─────────────────────────────────────────
#   τ: LLM called when Q-margin Δ_i = Q(a*) − Q(a_2nd) < Q_MARGIN_TAU
#   Low τ (0.05) drastically cuts token usage without losing key supervision.
Q_MARGIN_TAU       = 0.05

#   Hard cap on total LLM calls for the whole episode.
LLM_BUDGET         = 1600

#   Maximum nodes reviewed by LLM in a single simulation step.
MAX_NODES_PER_STEP = 2

#   Minimum green-phase hold (steps) before a phase switch is allowed.
MIN_GREEN_STEPS    = 3

#   Simulation length (seconds).
SIM_SECONDS        = 1600


# ── Intervention logger (statistics only, not the JSONL audit logger) ─────────

class InterventionStats:
    """Tracks LLM call and override statistics for the summary report."""

    def __init__(self) -> None:
        self.total_steps        = 0
        self.llm_calls          = 0
        self.llm_overrides      = 0
        self.safety_adjustments = 0
        self.confidence_scores: list = []
        self.margin_at_call:    list = []
        self.calls_by_reason:   dict = defaultdict(int)

    def record_call(self, margin: float, reason: str) -> None:
        self.llm_calls += 1
        self.margin_at_call.append(margin)
        self.calls_by_reason[reason] += 1

    def record_result(
        self,
        rl_action:       int,
        final_action:    int,
        safety_adjusted: bool,
        confidence:      float,
    ) -> None:
        if rl_action != final_action:
            self.llm_overrides += 1
        if safety_adjusted:
            self.safety_adjustments += 1
        self.confidence_scores.append(confidence)

    def summary(self) -> dict:
        n = max(self.llm_calls, 1)
        return {
            "total_sim_steps":      self.total_steps,
            "llm_calls":            self.llm_calls,
            "llm_overrides":        self.llm_overrides,
            "safety_adjustments":   self.safety_adjustments,
            "override_rate_%":      round(100 * self.llm_overrides / n, 2),
            "mean_confidence":      round(float(np.mean(self.confidence_scores))
                                          if self.confidence_scores else 0.0, 4),
            "mean_margin_at_call":  round(float(np.mean(self.margin_at_call))
                                          if self.margin_at_call else 0.0, 4),
            "calls_by_reason":      dict(self.calls_by_reason),
        }


# ── LLM backend factory ────────────────────────────────────────────────────────

def _build_langchain_backend(config: dict):
    """
    Build a LangChain ChatOpenAI backend callable for LLMGateway.

    Returns a callable (prompt: str) -> str that wraps the LangChain client.
    """
    api_key  = config["OPENAI_API_KEY"]
    model    = config["OPENAI_API_MODEL"]
    base_url = config.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    proxy    = config.get("OPENAI_PROXY", "") or None

    if not api_key or "YOUR_KEY" in api_key:
        raise ValueError(
            "\n\n*** API key not set. Edit configs/config.yaml and add your real key. ***\n"
        )

    kwargs: dict = {
        "model":           model,
        "temperature":     0.0,
        "openai_api_key":  api_key,
        "openai_api_base": base_url,
        "request_timeout": 20,
    }
    if proxy:
        try:
            import httpx
            kwargs["http_client"] = httpx.Client(proxy=proxy, timeout=20.0)
        except ImportError:
            logger.warning("httpx not installed — OPENAI_PROXY ignored.")

    chat = ChatOpenAI(**kwargs)
    logger.info(f"LLM backend: model={model}  base={base_url}")

    def _backend(prompt: str) -> str:
        from langchain_core.messages import HumanMessage
        response = chat.invoke([HumanMessage(content=prompt)])
        return response.content

    return _backend


# ── Neighbour summary builder ──────────────────────────────────────────────────

def _build_neighbor_summary(
    node_idx:   int,
    rl_actions: np.ndarray,
    all_infos:  list,
    attn_np:    np.ndarray,
) -> dict:
    """
    Build a compact neighbour context dict for junction ``node_idx``.

    Uses TLS_NEIGHBOR_MAP and TLS_INDEX from network.net_config.
    Falls back gracefully if those symbols are unavailable.
    """
    try:
        from network.net_config import CONTROLLED_TLS, TLS_NEIGHBOR_MAP, TLS_INDEX
        tls_id   = CONTROLLED_TLS[node_idx]
        neighbors = TLS_NEIGHBOR_MAP.get(tls_id, [])
        summary  = {}
        for nb_id in neighbors:
            nb_idx  = TLS_INDEX.get(nb_id, -1)
            if nb_idx < 0 or nb_idx >= len(all_infos):
                continue
            occ = all_infos[nb_idx].get("movement_occ", {})
            summary[nb_id] = {
                "mean_occ":  round(float(np.mean(list(occ.values()))) if occ else 0.0, 3),
                "rl_action": int(rl_actions[nb_idx]),
            }
        return summary
    except Exception:
        return {}


# ── Main inference loop ────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(LOG_PATH,    exist_ok=True)
    os.makedirs(RESULT_PATH, exist_ok=True)

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
    ckpt = os.path.join(MODEL_PATH, "gat_dqn_final.pt")
    trainer.load(ckpt)
    trainer.epsilon    = 0.0   # pure greedy at inference
    trainer.edge_index = EDGE_INDEX.to(device)
    logger.info(f"GAT-DQN loaded from {ckpt}  (ε=0, inference mode)")

    # ── Build SafeGAT-LLM pipeline ────────────────────────────────────────────
    backend = _build_langchain_backend(config)

    refiner = SafeGATRefiner(
        detector       = ScenarioDetector(
            queue_spike_threshold              = 0.85,
            zero_fraction_corruption_threshold = 0.90,
        ),
        gate           = InterventionGate(
            confidence_threshold = Q_MARGIN_TAU,
            intervention_budget  = MAX_NODES_PER_STEP,
        ),
        prompt_builder = TrafficPromptBuilder(),
        llm_gateway    = LLMGateway(
            backend             = backend,
            min_call_interval_s = 4.0,
            max_backoff_retries = 5,
            backoff_wait_s      = 30.0,
        ),
        safety_shield  = SafetyShield(min_green_hold=MIN_GREEN_STEPS),
        decision_logger = DecisionLogger(
            os.path.join(RESULT_PATH, "llm", "safegat_decisions.jsonl")
        ),
    )

    stats                = InterventionStats()
    llm_budget_remaining = LLM_BUDGET

    # ── Phase-runtime tracking (for safety shield) ─────────────────────────────
    phase_runtime = np.zeros(NUM_NODES, dtype=int)
    last_phase    = np.full(NUM_NODES, -1, dtype=int)

    # ── SUMO environment ──────────────────────────────────────────────────────
    trip_info_path = os.path.join(RESULT_PATH, "safegat.tripinfo.xml")
    env = make_grid_env(
        make_single_env_fn = make_env,
        tls_ids            = CONTROLLED_TLS,
        sumo_cfg           = SUMO_CFG,
        num_seconds        = SIM_SECONDS,
        use_gui            = True,
        log_file           = LOG_PATH,
        obs_dim            = OBS_DIM,
        trip_info          = trip_info_path,
    )

    obs          = env.reset()
    done         = False
    sim_step     = 0
    total_rewards = np.zeros(NUM_NODES, dtype=np.float32)
    infos         = [{} for _ in range(NUM_NODES)]
    step_log:     list = []

    logger.info(
        f"SafeGAT inference start  |  τ={Q_MARGIN_TAU}  |  B={LLM_BUDGET}  "
        f"|  max_per_step={MAX_NODES_PER_STEP}  |  t_min={MIN_GREEN_STEPS}"
    )

    while not done:
        # ── Step 1: GAT-DQN proposes actions + Q-values ───────────────────────
        rl_actions, q_values, attn_np = trainer.select_actions(obs)

        # ── Step 2: Compute uncertainty margins per node ───────────────────────
        margins = compute_q_margins(q_values)

        # ── Step 3: Build anomaly flags from raw obs ───────────────────────────
        # Reuse ScenarioDetector at batch level for efficiency
        anomaly_flags = np.array([
            bool(refiner.detector.detect(obs[i], infos[i])["tags"])
            for i in range(NUM_NODES)
        ])

        # ── Step 4: Select nodes for LLM review ───────────────────────────────
        uncertain_nodes = select_uncertain_nodes(margins, anomaly_flags, Q_MARGIN_TAU)

        final_actions = rl_actions.copy()

        if uncertain_nodes and llm_budget_remaining > 0:
            nodes_to_review = uncertain_nodes[:min(MAX_NODES_PER_STEP, llm_budget_remaining)]
            llm_budget_remaining -= len(nodes_to_review)

            logger.info(
                f"[SafeGAT] step={sim_step:>4}  |  review={nodes_to_review}  "
                f"|  Δ_min={margins[uncertain_nodes[0]]:.3f}  "
                f"|  budget_left={llm_budget_remaining}"
            )

            for node_idx in nodes_to_review:
                tls_id = CONTROLLED_TLS[node_idx]
                info   = infos[node_idx]

                # Build RLDecisionInfo for this junction
                action_scores = q_values[node_idx].tolist()
                rl_info = RLDecisionInfo(
                    intersection_id   = tls_id,
                    observation       = obs[node_idx].tolist(),
                    phase             = int(last_phase[node_idx]) if last_phase[node_idx] >= 0
                                        else int(round(float(obs[node_idx, 0]) * 3)),
                    rl_action         = int(rl_actions[node_idx]),
                    action_scores     = [float(s) for s in action_scores],
                    confidence_margin = float(margins[node_idx]),
                    legal_actions     = list(range(NUM_ACTIONS)),
                    neighbor_summary  = _build_neighbor_summary(
                                            node_idx, rl_actions, infos, attn_np),
                    anomaly_tags      = [],
                    metadata          = {
                        "phase_runtime":        int(phase_runtime[node_idx]),
                        "emergency_vehicle":    bool(obs[node_idx, 7] > 0.5),
                        "information_missing":  info.get("information_missing", False),
                        "observation_summary":  (
                            f"occ={obs[node_idx, 2:6].tolist()}  "
                            f"queue={obs[node_idx, 6]:.2f}  "
                            f"emerg={int(obs[node_idx, 7])}"
                        ),
                    },
                )

                reason = "anomaly" if anomaly_flags[node_idx] else "uncertain"
                stats.record_call(margin=float(margins[node_idx]), reason=reason)

                try:
                    result = refiner.refine(rl_info)
                    final_actions[node_idx] = result.final_action
                    stats.record_result(
                        rl_action       = int(rl_actions[node_idx]),
                        final_action    = result.final_action,
                        safety_adjusted = result.safety_adjusted,
                        confidence      = float(
                            result.llm_decision.parsed.get("confidence", 0.5)
                            if result.llm_decision else 0.5
                        ),
                    )
                except Exception as exc:
                    logger.warning(
                        f"[SafeGAT] refine failed for {tls_id}: {exc!r}  → RL fallback"
                    )

        elif llm_budget_remaining <= 0 and uncertain_nodes:
            logger.warning(
                f"[SafeGAT] step={sim_step}  |  budget exhausted  "
                f"|  {len(uncertain_nodes)} uncertain nodes forced to RL"
            )

        # ── Step 5: Execute final actions ─────────────────────────────────────
        obs, rewards, done, infos = env.step(final_actions)
        total_rewards += rewards
        stats.total_steps += 1

        # Update phase-runtime tracking
        for i in range(NUM_NODES):
            cur = int(round(float(obs[i, 0]) * 3))
            if cur == last_phase[i]:
                phase_runtime[i] += 1
            else:
                last_phase[i]    = cur
                phase_runtime[i] = 1

        # Per-step log entry
        step_log.append({
            "step":         sim_step,
            "mean_reward":  float(rewards.mean()),
            "mean_occ":     float(obs[:, 2:6].mean()),
            "mean_margin":  float(margins.mean()),
            "n_uncertain":  len(uncertain_nodes),
            "llm_calls":    stats.llm_calls,
            "budget_left":  llm_budget_remaining,
        })

        if sim_step % 100 == 0:
            logger.info(
                f"step={sim_step:>4}  |  mean_rew={rewards.mean():.3f}  "
                f"|  cumulative={total_rewards.sum():.1f}  "
                f"|  llm_calls={stats.llm_calls}  |  overrides={stats.llm_overrides}  "
                f"|  budget_left={llm_budget_remaining}"
            )

        sim_step += 1

    env.close()

    # ── Final report ───────────────────────────────────────────────────────────
    summary = stats.summary()
    logger.info("=" * 64)
    logger.info("SafeGAT Inference Complete")
    logger.info(f"Total reward : {total_rewards.sum():.2f}")
    try:
        from network.net_config import CONTROLLED_TLS as _tls
        per_jct = {_tls[i]: round(float(r), 2) for i, r in enumerate(total_rewards)}
        logger.info(f"Per-junction : {per_jct}")
    except Exception:
        pass
    logger.info(f"Intervention summary:\n{json.dumps(summary, indent=2)}")
    logger.info("=" * 64)

    # Save logs for offline analysis / plotting
    with open(os.path.join(RESULT_PATH, "step_log.json"), "w") as fh:
        json.dump(step_log, fh)
    with open(os.path.join(RESULT_PATH, "intervention_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info(f"Logs saved → {RESULT_PATH}")


if __name__ == "__main__":
    main()
