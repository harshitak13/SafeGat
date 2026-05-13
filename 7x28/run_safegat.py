"""
run_safegat.py — Inference entry point for SafeGAT-iLLM (7x28 network).

Loads a trained GAT-DQN checkpoint and runs one inference episode with
the full SafeGAT-LLM pipeline across all 196 junctions of the 7x28 grid:

    GAT-DQN proposes actions
        → Q-margin gating selects uncertain / anomalous nodes
        → SafeGATRefiner runs per flagged node:
              ScenarioDetector → InterventionGate → LLMGateway → SafetyShield
        → Final actions execute in SUMO
        → DecisionLogger records every refinement to JSONL

SafeGAT mechanisms
------------------
1. Uncertainty-aware gating   — LLM called only when delta_i < tau (or anomaly)
2. Anomaly detection          — queue spike, NaN, emergency, packet-loss
3. Intervention budget        — hard cap B on total LLM calls per episode
4. Per-step node cap          — at most MAX_NODES_PER_STEP calls per step
5. Safety shield              — yellow-lock, min-green, illegal-action repair
6. Audit logging              — full JSONL trail + per-junction CSV for all 196 nodes

Changes vs 4x4
--------------
- SUMO_CFG          : 7x28.sumocfg        (was 4x4.sumocfg)
- GAT_HEADS         : 2                   (was 4)
- LLM_BUDGET        : 800                 (was 1600; capped for 8-hr wall-clock target)
- MAX_NODES_PER_STEP: 2                   (parallel; kept under Groq RPM ceiling)
- SIM_SECONDS       : 1800                (was 3600; halved to hit 8-hr target)
- use_gui           : False               (was True)
- RESULT_PATH       : output/             (was data/output/)
- LLMGateway params : read from safegat_llm.yaml
- Model validation  : raises on known-invalid Groq model strings
- InterventionStats : extended with per-junction tracking
- Per-junction CSV  : output/per_junction_results.csv
- Simulated outputs : pre-loaded from training_curve.json on startup
- Wall-clock guard  : WALL_CLOCK_LIMIT_S=28800 hard-stops loop at 8 hrs

Run from the project root (after training)::

    python run_safegat.py

Requires
--------
- models/gat_dqn_final.pt
- configs/config.yaml        -- valid Groq API key + real model name
- configs/safegat_llm.yaml   -- LLM hyperparameters
- network/ files (net_config.py, graph_builder.py, 7x28.sumocfg)
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import torch
from langchain_openai import ChatOpenAI
from loguru import logger

# Project imports
from network.net_config    import CONTROLLED_TLS, NUM_ACTIONS, NUM_NODES
from network.graph_builder import EDGE_INDEX

from envs.grid_env_wrapper    import make_grid_env
from training.gat_dqn_trainer import FastGATDQNTrainer

from llm.action_refiner        import SafeGATRefiner
from llm.decision_logger       import DecisionLogger
from llm.intervention_gate     import InterventionGate
from llm.llm_gateway           import LLMGateway
from llm.safety_shield         import SafetyShield
from llm.scenario_detector     import ScenarioDetector
from llm.traffic_prompt_builder import TrafficPromptBuilder
from llm.types                 import RLDecisionInfo

from utils.make_tsc_env import make_env
from utils.readConfig   import read_config
from utils.margin       import compute_q_margins, select_uncertain_nodes

# Paths
_ROOT       = os.path.dirname(os.path.abspath(__file__))
LOG_PATH    = os.path.join(_ROOT, "log")
MODEL_PATH  = os.path.join(_ROOT, "models")
RESULT_PATH = os.path.join(_ROOT, "output")
SUMO_CFG    = os.path.join(_ROOT, "network", "7x28.sumocfg")

# Observation / network dimensions
OBS_DIM    = 8
HIDDEN_DIM = 64   # MUST match train.py
GAT_HEADS  = 2    # was 4 in 4x4

# SafeGAT inference hyperparameters — tuned for ~8 hr wall-clock on Groq free tier
Q_MARGIN_TAU       = 0.05
LLM_BUDGET         = 800    # ~312 expected calls (from training_curve) + 488 safety margin;
                             # hard ceiling so rate-limit storms can't blow the budget
MAX_NODES_PER_STEP = 2      # parallel dispatch; 2 * 2.5s = 5s/step worst case
MIN_GREEN_STEPS    = 3
SIM_SECONDS        = 1800   # 30-min sim (was 3600); halved so SUMO+LLM fits in 8 hrs

# Wall-clock hard stop: abort inference loop after this many seconds (8 hours).
# Saves all outputs collected so far rather than dying mid-run.
WALL_CLOCK_LIMIT_S = 28_800  # 8 * 3600

# Known-invalid model strings on Groq
_GROQ_INVALID_MODELS = {"openai/gpt-oss-20b"}


# Intervention statistics tracker

class InterventionStats:
    def __init__(self) -> None:
        self.total_steps        = 0
        self.llm_calls          = 0
        self.llm_overrides      = 0
        self.safety_adjustments = 0
        self.confidence_scores: list = []
        self.margin_at_call:    list = []
        self.calls_by_reason:   dict = defaultdict(int)
        self.per_junction_calls:     dict = defaultdict(int)
        self.per_junction_overrides: dict = defaultdict(int)
        self.per_junction_rewards:   dict = defaultdict(list)

    def record_call(self, tls_id: str, margin: float, reason: str) -> None:
        self.llm_calls += 1
        self.margin_at_call.append(margin)
        self.calls_by_reason[reason] += 1
        self.per_junction_calls[tls_id] += 1

    def record_result(self, tls_id, rl_action, final_action, safety_adjusted, confidence):
        if rl_action != final_action:
            self.llm_overrides += 1
            self.per_junction_overrides[tls_id] += 1
        if safety_adjusted:
            self.safety_adjustments += 1
        self.confidence_scores.append(confidence)

    def record_reward(self, tls_id: str, reward: float) -> None:
        self.per_junction_rewards[tls_id].append(reward)

    def summary(self) -> dict:
        n = max(self.llm_calls, 1)
        return {
            "total_sim_steps":     self.total_steps,
            "llm_calls":           self.llm_calls,
            "llm_overrides":       self.llm_overrides,
            "safety_adjustments":  self.safety_adjustments,
            "override_rate_%":     round(100 * self.llm_overrides / n, 2),
            "mean_confidence":     round(float(np.mean(self.confidence_scores)) if self.confidence_scores else 0.0, 4),
            "mean_margin_at_call": round(float(np.mean(self.margin_at_call)) if self.margin_at_call else 0.0, 4),
            "calls_by_reason":     dict(self.calls_by_reason),
        }

    def per_junction_summary(self) -> list:
        rows = []
        for tls_id in CONTROLLED_TLS:
            rews = self.per_junction_rewards.get(tls_id, [0.0])
            rows.append({
                "junction_id":   tls_id,
                "llm_calls":     self.per_junction_calls.get(tls_id, 0),
                "llm_overrides": self.per_junction_overrides.get(tls_id, 0),
                "mean_reward":   round(float(np.mean(rews)), 4),
                "total_reward":  round(float(np.sum(rews)), 4),
            })
        return rows


# LLM backend factory

def _build_langchain_backend(config: dict):
    api_key  = config["OPENAI_API_KEY"]
    model    = config["OPENAI_API_MODEL"]
    base_url = config.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    proxy    = config.get("OPENAI_PROXY", "") or None

    if not api_key or "YOUR_KEY" in api_key:
        raise ValueError("\n\n*** API key not set. Edit configs/config.yaml. ***\n")

    if model in _GROQ_INVALID_MODELS:
        raise ValueError(
            f"\n\n*** Model '{model}' does not exist on Groq. "
            f"Use e.g. 'llama-3.1-8b-instant'. ***\n"
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
            logger.warning("httpx not installed -- OPENAI_PROXY ignored.")

    chat = ChatOpenAI(**kwargs)
    logger.info(f"LLM backend: model={model}  base={base_url}")

    def _backend(prompt: str) -> str:
        from langchain_core.messages import HumanMessage
        return chat.invoke([HumanMessage(content=prompt)]).content

    return _backend


# Neighbour summary builder

def _build_neighbor_summary(node_idx, rl_actions, all_infos, attn_np) -> dict:
    try:
        from network.net_config import CONTROLLED_TLS as _tls, TLS_NEIGHBOR_MAP, TLS_INDEX
        tls_id    = _tls[node_idx]
        neighbors = TLS_NEIGHBOR_MAP.get(tls_id, [])
        summary   = {}
        for nb_id in neighbors:
            nb_idx = TLS_INDEX.get(nb_id, -1)
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


# Per-junction CSV writer

def _save_per_junction_csv(rows: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# Training curve loader

def _load_training_curve() -> list:
    """Load training_curve.json produced by the simulated training run.
    Logs expected vs actual performance and gives plot scripts full history."""
    path = os.path.join(RESULT_PATH, "training_curve.json")
    if os.path.exists(path):
        with open(path) as f:
            curve = json.load(f)
        best = max(curve, key=lambda r: r["total_reward"])
        logger.info(
            f"Loaded training_curve.json  ({len(curve)} episodes)  "
            f"| best_ep={best['episode']}  reward={best['total_reward']:.2f}  "
            f"| final_eps={curve[-1]['epsilon']:.4f}"
        )
        return curve
    logger.warning("training_curve.json not found in output/ — skipping training context.")
    return []


# Main inference loop

def main() -> None:
    os.makedirs(LOG_PATH,    exist_ok=True)
    os.makedirs(RESULT_PATH, exist_ok=True)
    os.makedirs(os.path.join(RESULT_PATH, "llm"), exist_ok=True)

    config = read_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read LLM hyperparameters from safegat_llm.yaml
    llm_cfg             = config.get("llm", {})
    min_call_interval_s = float(llm_cfg.get("min_call_interval_s", 2.0))
    backoff_wait_s      = float(llm_cfg.get("backoff_wait_s",      60.0))
    max_backoff_retries = int(  llm_cfg.get("max_backoff_retries",  3))

    # Load trained GAT-DQN
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
    trainer.epsilon    = 0.0
    trainer.edge_index = EDGE_INDEX.to(device)
    logger.info(f"GAT-DQN loaded from {ckpt}  (epsilon=0, inference mode)")
    logger.info(f"Edge index shape: {EDGE_INDEX.shape}")

    # Build SafeGAT-LLM pipeline
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
            min_call_interval_s = min_call_interval_s,
            max_backoff_retries = max_backoff_retries,
            backoff_wait_s      = backoff_wait_s,
        ),
        safety_shield  = SafetyShield(min_green_hold=MIN_GREEN_STEPS),
        decision_logger = DecisionLogger(
            os.path.join(RESULT_PATH, "llm", "safegat_decisions.jsonl")
        ),
    )

    # Load training context (training_curve.json from simulated run)
    training_curve = _load_training_curve()

    stats                = InterventionStats()
    llm_budget_remaining = LLM_BUDGET
    # Cooldown tracker: junction index -> first step it is eligible for LLM again.
    # After exhausting all backoff retries, a node is skipped for 30 steps so the
    # retry storm observed in the logs (J145 failing every single step) cannot recur.
    llm_cooldown: dict[int, int] = {}

    # Wall-clock guard: record start time; inference loop checks this every step
    # and performs a clean early-exit (saving all outputs) if limit is exceeded.
    _wall_start = time.time()

    phase_runtime = np.zeros(NUM_NODES, dtype=int)
    last_phase    = np.full(NUM_NODES, -1, dtype=int)

    # SUMO environment
    trip_info_path = os.path.join(RESULT_PATH, "safegat.tripinfo.xml")
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

    obs           = env.reset()
    done          = False
    sim_step      = 0
    total_rewards = np.zeros(NUM_NODES, dtype=np.float32)
    infos         = [{} for _ in range(NUM_NODES)]
    step_log:     list = []

    logger.info(
        f"SafeGAT 7x28 inference start  |  tau={Q_MARGIN_TAU}  |  B={LLM_BUDGET}  "
        f"|  max_per_step={MAX_NODES_PER_STEP}  |  t_min={MIN_GREEN_STEPS}  "
        f"|  min_interval={min_call_interval_s}s  |  backoff={backoff_wait_s}s  "
        f"|  sim_seconds={SIM_SECONDS}  |  wall_limit={WALL_CLOCK_LIMIT_S//3600}h"
    )
    if training_curve:
        best_tc = max(training_curve, key=lambda r: r["total_reward"])
        logger.info(
            f"Training context loaded  |  episodes={len(training_curve)}  "
            f"|  best_ep={best_tc['episode']}  reward={best_tc['total_reward']:.2f}  "
            f"|  expected_llm_call_rate=35-50%  |  expected_margin=0.02-0.06"
        )

    while not done:
        # Wall-clock guard: clean early exit if we're approaching 8 hours
        _elapsed = time.time() - _wall_start
        if _elapsed >= WALL_CLOCK_LIMIT_S:
            logger.warning(
                f"[SafeGAT] Wall-clock limit reached ({_elapsed/3600:.2f}h >= "
                f"{WALL_CLOCK_LIMIT_S/3600:.0f}h) at step={sim_step}  -> saving and exiting."
            )
            done = True
            break

        # Step 1: GAT-DQN proposes actions + Q-values
        rl_actions, q_values, attn_np = trainer.select_actions(obs)

        # Step 2: Q-margin uncertainty for all 196 nodes
        margins = compute_q_margins(q_values)

        # Step 3: Anomaly detection
        anomaly_flags = np.array([
            bool(refiner.detector.detect(obs[i], infos[i])["tags"])
            for i in range(NUM_NODES)
        ])

        # Step 4: Select nodes for LLM review
        uncertain_nodes = select_uncertain_nodes(margins, anomaly_flags, Q_MARGIN_TAU)

        final_actions = rl_actions.copy()

        if uncertain_nodes and llm_budget_remaining > 0:
            # Filter out nodes still in their cooldown window (recent retry-storm victims)
            eligible_nodes = [
                n for n in uncertain_nodes
                if llm_cooldown.get(n, 0) <= sim_step
            ]
            nodes_to_review = eligible_nodes[:min(MAX_NODES_PER_STEP, llm_budget_remaining)]
            llm_budget_remaining -= len(nodes_to_review)

            logger.info(
                f"[SafeGAT] step={sim_step:>4}  |  review={nodes_to_review}  "
                f"|  delta_min={margins[uncertain_nodes[0]]:.3f}  "
                f"|  budget_left={llm_budget_remaining}"
            )

            # Build per-node RLDecisionInfo objects (cheap, do before spawning threads)
            def _make_rl_info(node_idx: int) -> tuple[int, str, str, "RLDecisionInfo"]:
                tls_id = CONTROLLED_TLS[node_idx]
                info   = infos[node_idx]
                reason = "anomaly" if anomaly_flags[node_idx] else "uncertain"
                rl_info = RLDecisionInfo(
                    intersection_id   = tls_id,
                    observation       = obs[node_idx].tolist(),
                    phase             = int(last_phase[node_idx])
                                        if last_phase[node_idx] >= 0
                                        else int(round(float(obs[node_idx, 0]) * 3)),
                    rl_action         = int(rl_actions[node_idx]),
                    action_scores     = [float(s) for s in q_values[node_idx].tolist()],
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
                return node_idx, tls_id, reason, rl_info

            node_payloads = [_make_rl_info(n) for n in nodes_to_review]

            # Record calls before dispatch (stats is not thread-safe for writes,
            # but record_call only appends to lists/dicts so GIL protects us)
            for node_idx, tls_id, reason, _ in node_payloads:
                stats.record_call(tls_id=tls_id, margin=float(margins[node_idx]),
                                  reason=reason)

            # --- Parallel LLM dispatch ---
            # ThreadPoolExecutor is safe here: the GIL is released during I/O-bound
            # API calls, so threads make real progress concurrently.
            # max_workers matches MAX_NODES_PER_STEP so we never spawn idle threads.
            def _call_refiner(payload):
                node_idx, tls_id, reason, rl_info = payload
                logger.info(
                    f"  -> calling LLM for {tls_id} ({reason}, delta={margins[node_idx]:.3f})..."
                )
                return node_idx, tls_id, reason, refiner.refine(rl_info)

            with ThreadPoolExecutor(max_workers=MAX_NODES_PER_STEP) as pool:
                future_map = {pool.submit(_call_refiner, p): p for p in node_payloads}
                for future in as_completed(future_map):
                    node_idx, tls_id, reason, _ = future_map[future]
                    try:
                        _, _, _, result = future.result()
                        logger.info(
                            f"  -> {tls_id} done: rl={rl_actions[node_idx]} final={result.final_action}"
                        )
                        final_actions[node_idx] = result.final_action
                        stats.record_result(
                            tls_id          = tls_id,
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
                            f"[SafeGAT] refine failed for {tls_id}: {exc!r}  -> RL fallback"
                        )
                        # Put node in cooldown for 30 steps so it won't be retried
                        # every step and trigger another backoff storm.
                        llm_cooldown[node_idx] = sim_step + 30
                        logger.info(
                            f"  -> {tls_id} cooldown set: skipping LLM until step {llm_cooldown[node_idx]}"
                        )

        elif llm_budget_remaining <= 0 and uncertain_nodes:
            logger.warning(
                f"[SafeGAT] step={sim_step}  |  budget exhausted  "
                f"|  {len(uncertain_nodes)} uncertain nodes forced to RL"
            )

        # Step 5: Execute final actions in SUMO
        obs, rewards, done, infos = env.step(final_actions)
        total_rewards += rewards
        stats.total_steps += 1

        for i, tls_id in enumerate(CONTROLLED_TLS):
            stats.record_reward(tls_id, float(rewards[i]))

        for i in range(NUM_NODES):
            cur = int(round(float(obs[i, 0]) * 3))
            if cur == last_phase[i]:
                phase_runtime[i] += 1
            else:
                last_phase[i]    = cur
                phase_runtime[i] = 1

        step_log.append({
            "step":        sim_step,
            "mean_reward": float(rewards.mean()),
            "mean_occ":    float(obs[:, 2:6].mean()),
            "mean_margin": float(margins.mean()),
            "n_uncertain": len(uncertain_nodes),
            "llm_calls":   stats.llm_calls,
            "budget_left": llm_budget_remaining,
            "elapsed_s":   round(time.time() - _wall_start, 1),
        })

        if sim_step % 100 == 0:
            _elapsed_h = (time.time() - _wall_start) / 3600
            logger.info(
                f"step={sim_step:>4}  |  mean_rew={rewards.mean():.3f}  "
                f"|  cumulative={total_rewards.sum():.1f}  "
                f"|  llm_calls={stats.llm_calls}  |  overrides={stats.llm_overrides}  "
                f"|  budget_left={llm_budget_remaining}  |  elapsed={_elapsed_h:.2f}h"
            )

        sim_step += 1

    env.close()

    _total_wall = time.time() - _wall_start

    # Final report
    summary      = stats.summary()
    per_jct_rows = stats.per_junction_summary()

    logger.info("=" * 70)
    logger.info("SafeGAT 7x28 Inference Complete")
    logger.info(f"Total reward  : {total_rewards.sum():.2f}")
    logger.info(f"Mean reward   : {total_rewards.mean():.4f}")
    logger.info(f"Sim steps     : {sim_step}")
    logger.info(f"Wall-clock    : {_total_wall/3600:.2f}h  ({_total_wall:.0f}s)")
    if training_curve:
        best_tc = max(training_curve, key=lambda r: r["total_reward"])
        logger.info(
            f"Training ref  : best_ep={best_tc['episode']}  "
            f"reward={best_tc['total_reward']:.2f}  final_eps={training_curve[-1]['epsilon']:.4f}"
        )
    logger.info(f"Intervention summary:\n{json.dumps(summary, indent=2)}")

    top10 = sorted(per_jct_rows, key=lambda r: r["llm_calls"], reverse=True)[:10]
    logger.info(f"Top-10 junctions by LLM calls: {top10}")
    logger.info("=" * 70)

    # Enrich summary with wall-clock and training context before saving
    summary["wall_clock_s"]      = round(_total_wall, 1)
    summary["wall_clock_h"]      = round(_total_wall / 3600, 3)
    summary["sim_seconds"]       = SIM_SECONDS
    summary["llm_budget_used"]   = LLM_BUDGET - llm_budget_remaining
    if training_curve:
        best_tc = max(training_curve, key=lambda r: r["total_reward"])
        summary["training_ref"] = {
            "episodes":        len(training_curve),
            "best_episode":    best_tc["episode"],
            "best_reward":     best_tc["total_reward"],
            "final_epsilon":   training_curve[-1]["epsilon"],
            "final_env_steps": training_curve[-1]["env_steps"],
        }

    with open(os.path.join(RESULT_PATH, "step_log.json"), "w") as fh:
        json.dump(step_log, fh)
    with open(os.path.join(RESULT_PATH, "intervention_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # Write combined output: training curve + inference step log in one file
    # so plot_safegat_metrics.py has everything it needs without two file reads.
    combined = {
        "training_curve":  training_curve,
        "inference_steps": step_log,
        "summary":         summary,
    }
    with open(os.path.join(RESULT_PATH, "combined_results.json"), "w") as fh:
        json.dump(combined, fh)

    _save_per_junction_csv(
        per_jct_rows,
        os.path.join(RESULT_PATH, "per_junction_results.csv"),
    )

    logger.info(
        f"All logs saved -> {RESULT_PATH}  "
        f"| step_log.json  intervention_summary.json  "
        f"combined_results.json  per_junction_results.csv"
    )


if __name__ == "__main__":
    main()
