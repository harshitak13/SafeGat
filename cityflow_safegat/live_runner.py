"""Run live SafeGAT-LLM control on CityFlow datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import yaml
from loguru import logger

from cityflow_safegat.live_env import CityFlowSignalEnv, edge_index_tensor, write_graph_metadata


OBS_DIM = 8
HIDDEN_DIM = 64


class InterventionStats:
    def __init__(self, tls_ids: list[str]) -> None:
        self.tls_ids = tls_ids
        self.total_steps = 0
        self.llm_calls = 0
        self.llm_overrides = 0
        self.llm_failures = 0
        self.safe_fallback_holds = 0
        self.safety_adjustments = 0
        self.confidence_scores: list[float] = []
        self.margin_at_call: list[float] = []
        self.calls_by_reason: dict[str, int] = defaultdict(int)
        self.per_junction_calls: dict[str, int] = defaultdict(int)
        self.per_junction_overrides: dict[str, int] = defaultdict(int)
        self.per_junction_rewards: dict[str, list[float]] = defaultdict(list)
        self.override_quality: list[dict] = []

    def record_call(self, tls_id: str, margin: float, reason: str) -> None:
        self.llm_calls += 1
        self.margin_at_call.append(float(margin))
        self.calls_by_reason[reason] += 1
        self.per_junction_calls[tls_id] += 1

    def record_result(self, tls_id: str, rl_action: int, final_action: int, safety_adjusted: bool, confidence: float, quality=None):
        if int(rl_action) != int(final_action):
            self.llm_overrides += 1
            self.per_junction_overrides[tls_id] += 1
            if quality and quality.get("available"):
                self.override_quality.append(quality)
        if safety_adjusted:
            self.safety_adjustments += 1
        self.confidence_scores.append(float(confidence))

    def record_llm_failure(self, held_current_phase: bool) -> None:
        self.llm_failures += 1
        if held_current_phase:
            self.safe_fallback_holds += 1

    def record_reward(self, tls_id: str, reward: float) -> None:
        self.per_junction_rewards[tls_id].append(float(reward))

    def summary(self) -> dict:
        n = max(self.llm_calls, 1)
        return {
            "total_sim_steps": self.total_steps,
            "llm_calls": self.llm_calls,
            "llm_failures": self.llm_failures,
            "safe_fallback_holds": self.safe_fallback_holds,
            "llm_overrides": self.llm_overrides,
            "safety_adjustments": self.safety_adjustments,
            "override_rate_%": round(100 * self.llm_overrides / n, 2),
            "mean_confidence": round(float(np.mean(self.confidence_scores)) if self.confidence_scores else 0.0, 4),
            "mean_margin_at_call": round(float(np.mean(self.margin_at_call)) if self.margin_at_call else 0.0, 4),
            "override_precision_%": (
                round(100 * sum(bool(q.get("helped")) for q in self.override_quality) / len(self.override_quality), 2)
                if self.override_quality else None
            ),
            "calls_by_reason": dict(self.calls_by_reason),
        }

    def per_junction_summary(self) -> list[dict]:
        rows = []
        for tls_id in self.tls_ids:
            rewards = self.per_junction_rewards.get(tls_id, [0.0])
            rows.append({
                "junction_id": tls_id,
                "llm_calls": self.per_junction_calls.get(tls_id, 0),
                "llm_overrides": self.per_junction_overrides.get(tls_id, 0),
                "mean_reward": round(float(np.mean(rewards)), 4),
                "total_reward": round(float(np.sum(rewards)), 4),
            })
        return rows


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _cfg_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _import_impl(impl_dir: Path):
    sys.path.insert(0, str(impl_dir))
    from training.gat_dqn_trainer import FastGATDQNTrainer
    from llm.action_refiner import SafeGATRefiner
    from llm.anomaly_forecaster import GRUAnomalyForecaster
    from llm.decision_logger import DecisionLogger
    from llm.intervention_gate import InterventionGate
    from llm.llm_gateway import LLMGateway
    from llm.safety_shield import SafetyShield
    from llm.scenario_detector import ScenarioDetector
    from llm.traffic_prompt_builder import TrafficPromptBuilder
    from llm.types import RLDecisionInfo
    from utils.margin import compute_q_margins, select_uncertain_nodes
    from utils.readConfig import read_config

    return {
        "FastGATDQNTrainer": FastGATDQNTrainer,
        "SafeGATRefiner": SafeGATRefiner,
        "GRUAnomalyForecaster": GRUAnomalyForecaster,
        "DecisionLogger": DecisionLogger,
        "InterventionGate": InterventionGate,
        "LLMGateway": LLMGateway,
        "SafetyShield": SafetyShield,
        "ScenarioDetector": ScenarioDetector,
        "TrafficPromptBuilder": TrafficPromptBuilder,
        "RLDecisionInfo": RLDecisionInfo,
        "compute_q_margins": compute_q_margins,
        "select_uncertain_nodes": select_uncertain_nodes,
        "read_config": read_config,
    }


def _load_llm_cfg(impl_dir: Path, config: dict) -> dict:
    llm_cfg = dict(config.get("llm", {}) or {})
    yaml_path = impl_dir / "configs" / "safegat_llm.yaml"
    if yaml_path.exists():
        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded.get("llm"), dict):
            llm_cfg = {**loaded["llm"], **llm_cfg}
    return llm_cfg


def _risk_weights(llm_cfg: dict) -> dict:
    weights = {
        "uncertainty": 2.0,
        "anomaly": 1.5,
        "queue": 1.0,
        "waiting": 0.5,
        "safety": 1.0,
    }
    cfg_weights = llm_cfg.get("risk_weights", {})
    if isinstance(cfg_weights, dict):
        weights.update({k: float(v) for k, v in cfg_weights.items() if k in weights})
    raw_env = os.environ.get("SAFEGAT_RISK_WEIGHTS_JSON", "")
    if raw_env:
        weights.update({k: float(v) for k, v in json.loads(raw_env).items() if k in weights})
    return weights


def _build_langchain_backend(config: dict):
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    api_key = config["OPENAI_API_KEY"]
    model = config["OPENAI_API_MODEL"]
    base_url = config.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    proxy = config.get("OPENAI_PROXY", "") or None
    if not api_key or "YOUR_KEY" in api_key:
        raise ValueError("API key not set. Add configs/config.yaml or OPENAI_API_KEY.")
    kwargs = {
        "model": model,
        "temperature": 0.0,
        "openai_api_key": api_key,
        "openai_api_base": base_url,
        "request_timeout": 20,
    }
    if proxy:
        import httpx

        kwargs["http_client"] = httpx.Client(proxy=proxy, timeout=20.0)
    chat = ChatOpenAI(**kwargs)
    logger.info(f"LLM backend: model={model} base={base_url}")

    def _backend(prompt: str) -> str:
        return chat.invoke([HumanMessage(content=prompt)]).content

    return _backend


def _resolved_llm_config_path(args_config: Optional[str], impl_dir: Path) -> Path:
    if args_config:
        return Path(args_config).resolve()
    return impl_dir / "configs" / "config.yaml"


def _time_of_day(sim_time_s: int) -> str:
    hour = (sim_time_s // 3600) % 24
    if 7 <= hour < 10:
        return "morning_peak"
    if 16 <= hour < 19:
        return "evening_peak"
    if 10 <= hour < 16:
        return "midday"
    return "off_peak"


def _propagation_status(summary: dict) -> str:
    if not summary:
        return "isolated_or_unknown"
    occupancies = [float(v.get("mean_occ", 0.0)) for v in summary.values() if isinstance(v, dict)]
    if not occupancies:
        return "neighbor_pressure_unknown"
    max_occ = max(occupancies)
    if max_occ >= 0.70:
        return f"downstream_congested max_neighbor_occ={max_occ:.3f}"
    return f"stable_neighbors mean_neighbor_occ={float(np.mean(occupancies)):.3f}"


def _save_per_junction_csv(rows: list[dict], path: Path) -> None:
    import csv

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _current_legal_phase(env: CityFlowSignalEnv, node_idx: int) -> int:
    tls_id = env.controlled_tls[node_idx]
    legal = env.network.legal_actions[tls_id]
    current = int(env.phases.get(tls_id, legal[0]))
    if current in legal:
        return current
    return int(legal[0])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--roadnet", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--impl-dir", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default="safegat_cityflow_live_output")
    parser.add_argument("--sim-seconds", type=int, default=1800)
    parser.add_argument("--action-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thread-num", type=int, default=1)
    parser.add_argument("--save-replay", action="store_true")
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--llm-config",
        default=None,
        help="Path to config.yaml containing OPENAI_API_KEY, OPENAI_API_MODEL, and OPENAI_API_BASE.",
    )
    parser.add_argument("--mock-llm", action="store_true", help="Use LLMGateway's mock backend for smoke tests.")
    parser.add_argument("--wall-clock-limit-s", type=int, default=0, help="Optional clean-stop wall-clock guard.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    dataset_dir = Path(args.dataset_dir).resolve()
    impl_dir = Path(args.impl_dir).resolve()
    output_dir = dataset_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "llm").mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path).resolve() if args.model_path else dataset_dir / "models" / "gat_dqn_final.pt"

    mods = _import_impl(impl_dir)
    llm_config_path = _resolved_llm_config_path(args.llm_config, impl_dir)
    config = {} if args.mock_llm else mods["read_config"](str(llm_config_path))
    if not args.mock_llm:
        logger.info(
            f"LLM config loaded from {llm_config_path} | "
            f"model={config.get('OPENAI_API_MODEL')} | base={config.get('OPENAI_API_BASE')}"
        )
    llm_cfg = _load_llm_cfg(impl_dir, config)
    risk_weights = _risk_weights(llm_cfg)

    env = CityFlowSignalEnv(
        dataset_dir=dataset_dir,
        roadnet=args.roadnet,
        flow=args.flow,
        output_dir=output_dir,
        episode_seconds=args.sim_seconds,
        action_interval=args.action_interval,
        seed=args.seed,
        save_replay=args.save_replay,
        thread_num=args.thread_num,
    )
    write_graph_metadata(env.network, output_dir / "cityflow_graph.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = mods["FastGATDQNTrainer"](
        node_feature_dim=OBS_DIM,
        num_nodes=env.num_nodes,
        num_actions=env.num_actions,
        hidden_dim=HIDDEN_DIM,
        gat_heads=args.gat_heads,
        top_k=args.top_k,
        device=device,
    )
    trainer.edge_index = edge_index_tensor(env.network, device=device)
    trainer.load(str(model_path))
    trainer.epsilon = 0.0

    confidence_threshold = _env_float("SAFEGAT_CONFIDENCE_THRESHOLD", float(llm_cfg.get("confidence_threshold", 0.05)))
    max_nodes_default = int(llm_cfg.get("max_nodes_per_step", 2 if env.num_nodes <= 64 else 8))
    max_nodes_per_step = _env_int("SAFEGAT_MAX_NODES_PER_STEP", max_nodes_default)
    intervention_rate = _env_float("SAFEGAT_INTERVENTION_RATE", max_nodes_per_step / max(1, env.num_nodes))
    max_nodes_per_step = max(1, min(env.num_nodes, int(math.ceil(intervention_rate * env.num_nodes))))
    intervention_rate = max_nodes_per_step / max(1, env.num_nodes)
    llm_budget = _env_int("SAFEGAT_LLM_BUDGET", int(llm_cfg.get("intervention_budget", max_nodes_per_step * 400)))
    failure_rate = _env_float("SAFEGAT_LLM_FAILURE_RATE", float(llm_cfg.get("failure_injection_rate", 0.0)))
    fallback_to_rl = _cfg_bool(llm_cfg.get("fallback_to_rl"), False)
    rate_limit_cooldown_steps = _env_int(
        "SAFEGAT_RATE_LIMIT_COOLDOWN_STEPS",
        int(llm_cfg.get("rate_limit_cooldown_steps", 60)),
    )
    rate_limit_cooldown_s = _env_int(
        "SAFEGAT_RATE_LIMIT_COOLDOWN_S",
        int(llm_cfg.get("rate_limit_cooldown_s", 300)),
    )

    backend = None if args.mock_llm else _build_langchain_backend(config)
    refiner = mods["SafeGATRefiner"](
        detector=mods["ScenarioDetector"](
            queue_spike_threshold=float(llm_cfg.get("queue_spike_threshold", 0.85)),
            zero_fraction_corruption_threshold=float(llm_cfg.get("zero_fraction_corruption_threshold", 0.90)),
        ),
        gate=mods["InterventionGate"](
            confidence_threshold=confidence_threshold,
            intervention_budget=max_nodes_per_step,
            forecast_threshold=float(llm_cfg.get("forecast_anomaly_threshold", 0.65)),
            low_conf_weight=risk_weights["uncertainty"],
            anomaly_weight=risk_weights["anomaly"],
            queue_weight=risk_weights["queue"],
            wait_weight=risk_weights["waiting"],
            safety_weight=risk_weights["safety"],
        ),
        prompt_builder=mods["TrafficPromptBuilder"](),
        llm_gateway=mods["LLMGateway"](
            backend=backend,
            min_call_interval_s=float(llm_cfg.get("min_call_interval_s", 4.0)),
            max_backoff_retries=int(llm_cfg.get("max_backoff_retries", 5)),
            backoff_wait_s=float(llm_cfg.get("backoff_wait_s", 30.0)),
            failure_injection_rate=failure_rate,
            failure_rng_seed=42,
        ),
        safety_shield=mods["SafetyShield"](
            min_green_hold=int(llm_cfg.get("min_green_hold", 3)),
            transition_confidence_threshold=float(llm_cfg.get("transition_confidence_threshold", 0.75)),
        ),
        decision_logger=mods["DecisionLogger"](str(output_dir / "llm" / "safegat_decisions.jsonl")),
        anomaly_forecaster=mods["GRUAnomalyForecaster"](
            threshold=float(llm_cfg.get("forecast_anomaly_threshold", 0.65)),
            horizon=int(llm_cfg.get("forecast_horizon_steps", 3)),
            history_len=int(llm_cfg.get("forecast_history_len", 4)),
            queue_spike_threshold=float(llm_cfg.get("queue_spike_threshold", 0.85)),
            zero_fraction_corruption_threshold=float(llm_cfg.get("zero_fraction_corruption_threshold", 0.90)),
        ),
    )

    stats = InterventionStats(env.controlled_tls)
    llm_budget_remaining = llm_budget
    llm_cooldown: dict[int, int] = {}
    llm_global_cooldown_until = 0
    llm_global_cooldown_until_ts = 0.0
    last_global_cooldown_log_step = -1
    phase_runtime = np.zeros(env.num_nodes, dtype=int)
    last_phase = np.full(env.num_nodes, -1, dtype=int)
    queue_baseline = np.zeros(env.num_nodes, dtype=float)
    total_rewards = np.zeros(env.num_nodes, dtype=np.float32)
    step_log = []

    obs = env.reset()
    infos = env._infos(obs)
    done = False
    sim_step = 0
    wall_start = time.time()

    logger.info(
        f"Live CityFlow SafeGAT start | nodes={env.num_nodes} actions={env.num_actions} "
        f"| tau={confidence_threshold} | B={llm_budget} | K={max_nodes_per_step} "
        f"| fallback_to_rl={fallback_to_rl} | rate_limit_cooldown_s={rate_limit_cooldown_s} "
        f"| model={model_path} | mock_llm={args.mock_llm}"
    )

    while not done:
        if args.wall_clock_limit_s and time.time() - wall_start >= args.wall_clock_limit_s:
            logger.warning("Wall-clock limit reached; saving outputs and stopping.")
            break

        rl_actions, q_values, _attn = trainer.select_actions(obs)
        margins = mods["compute_q_margins"](q_values)
        reactive_flags = np.asarray([bool(refiner.detector.detect(obs[i], infos[i])["tags"]) for i in range(env.num_nodes)])
        forecast_probs = refiner.anomaly_forecaster.update_many(env.controlled_tls, obs)
        forecast_flags = forecast_probs >= refiner.anomaly_forecaster.threshold
        uncertain_nodes = mods["select_uncertain_nodes"](margins, reactive_flags | forecast_flags, confidence_threshold)

        final_actions = rl_actions.copy()
        if uncertain_nodes and llm_budget_remaining > 0:
            now_wall = time.time()
            if llm_global_cooldown_until > sim_step or llm_global_cooldown_until_ts > now_wall:
                if not fallback_to_rl:
                    for idx in uncertain_nodes:
                        final_actions[idx] = _current_legal_phase(env, idx)
                if last_global_cooldown_log_step < 0 or sim_step - last_global_cooldown_log_step >= 10:
                    remaining_s = max(0.0, llm_global_cooldown_until_ts - now_wall)
                    logger.info(
                        f"[SafeGAT-CityFlow] step={sim_step:>4} global LLM cooldown "
                        f"for {remaining_s:.0f}s; holding current phase"
                    )
                    last_global_cooldown_log_step = sim_step
                nodes_to_review = []
            else:
                nodes_to_review = None

        if uncertain_nodes and llm_budget_remaining > 0 and nodes_to_review is None:
            eligible = [idx for idx in uncertain_nodes if llm_cooldown.get(idx, 0) <= sim_step]
            if not fallback_to_rl:
                for idx in uncertain_nodes:
                    if idx not in eligible:
                        final_actions[idx] = _current_legal_phase(env, idx)
            nodes_to_review = eligible[: min(max_nodes_per_step, llm_budget_remaining)]
            llm_budget_remaining -= len(nodes_to_review)
            logger.info(
                f"[SafeGAT-CityFlow] step={sim_step:>4} review={nodes_to_review} "
                f"delta_min={margins[uncertain_nodes[0]]:.4f} budget_left={llm_budget_remaining}"
            )

            for node_idx in nodes_to_review:
                tls_id = env.controlled_tls[node_idx]
                if reactive_flags[node_idx]:
                    reason = "anomaly"
                elif forecast_flags[node_idx]:
                    reason = "forecast_anomaly"
                else:
                    reason = "uncertain"
                neighbor_summary = env.neighbor_summary(node_idx, rl_actions, obs)
                current_queue = float(obs[node_idx, 6])
                baseline_queue = float(queue_baseline[node_idx])
                rl_info = mods["RLDecisionInfo"](
                    intersection_id=tls_id,
                    observation=obs[node_idx].tolist(),
                    phase=int(last_phase[node_idx]) if last_phase[node_idx] >= 0 else int(env.phases.get(tls_id, 0)),
                    rl_action=int(rl_actions[node_idx]),
                    action_scores=[float(x) for x in q_values[node_idx].tolist()],
                    confidence_margin=float(margins[node_idx]),
                    legal_actions=env.network.legal_actions[tls_id],
                    neighbor_summary=neighbor_summary,
                    anomaly_tags=[],
                    metadata={
                        "phase_runtime": int(phase_runtime[node_idx]),
                        "sim_step": int(sim_step),
                        "sim_time_s": int(env.elapsed_seconds),
                        "time_of_day": _time_of_day(env.elapsed_seconds),
                        "transition_model_source": "cityflow_live",
                        "transition_model_confidence": 1.0,
                        "forecast_anomaly_prob": float(forecast_probs[node_idx]),
                        "forecast_horizon": int(llm_cfg.get("forecast_horizon_steps", 3)),
                        "emergency_vehicle": False,
                        "information_missing": False,
                        "historical_baseline_queue": round(baseline_queue, 4),
                        "current_queue": round(current_queue, 4),
                        "queue_pressure": round(max(current_queue, current_queue - baseline_queue), 4),
                        "waiting_pressure": round(float(obs[node_idx, 7]), 4),
                        "event_details": reason,
                        "propagation_status": _propagation_status(neighbor_summary),
                        "yellow_phase": False,
                        "observation_summary": (
                            f"occ={np.round(obs[node_idx, 2:6], 3).tolist()} "
                            f"queue={obs[node_idx, 6]:.3f} waiting={obs[node_idx, 7]:.3f}"
                        ),
                    },
                )
                stats.record_call(tls_id, float(margins[node_idx]), reason)
                try:
                    result = refiner.refine(rl_info)
                    final_actions[node_idx] = result.final_action
                    stats.record_result(
                        tls_id=tls_id,
                        rl_action=int(rl_actions[node_idx]),
                        final_action=int(result.final_action),
                        safety_adjusted=result.safety_adjusted,
                        confidence=float(result.llm_decision.parsed.get("confidence", 0.5) if result.llm_decision else 0.5),
                        quality=result.debug.get("override_quality"),
                    )
                except Exception as exc:
                    if fallback_to_rl:
                        logger.warning(f"[SafeGAT-CityFlow] refine failed for {tls_id}: {exc!r}; RL fallback")
                        stats.record_llm_failure(held_current_phase=False)
                    else:
                        hold_action = _current_legal_phase(env, node_idx)
                        final_actions[node_idx] = hold_action
                        stats.record_llm_failure(held_current_phase=True)
                        logger.warning(
                            f"[SafeGAT-CityFlow] refine failed for {tls_id}: {exc!r}; "
                            f"holding current phase={hold_action} instead of RL"
                        )
                    llm_cooldown[node_idx] = sim_step + 30
                    if "rate-limit" in str(exc).lower() or "rate_limit" in str(exc).lower() or "429" in str(exc):
                        llm_global_cooldown_until = sim_step + rate_limit_cooldown_steps
                        llm_global_cooldown_until_ts = time.time() + rate_limit_cooldown_s
                        logger.warning(
                            f"[SafeGAT-CityFlow] global LLM cooldown set until "
                            f"step={llm_global_cooldown_until} / {rate_limit_cooldown_s}s "
                            f"after provider rate limit"
                        )

        final_actions = env.legalize_actions(final_actions)
        obs, rewards, done, infos = env.step(final_actions)
        total_rewards += rewards
        stats.total_steps += 1
        queue_baseline = 0.98 * queue_baseline + 0.02 * obs[:, 6]
        for idx, tls_id in enumerate(env.controlled_tls):
            stats.record_reward(tls_id, float(rewards[idx]))
            cur = int(env.phases.get(tls_id, 0))
            if cur == last_phase[idx]:
                phase_runtime[idx] += 1
            else:
                last_phase[idx] = cur
                phase_runtime[idx] = 1

        step_log.append({
            "step": sim_step,
            "sim_time_s": env.elapsed_seconds,
            "mean_reward": float(rewards.mean()),
            "mean_occ": float(obs[:, 2:6].mean()),
            "mean_queue": float(obs[:, 6].mean()),
            "mean_waiting": float(obs[:, 7].mean()),
            "mean_margin": float(margins.mean()),
            "mean_forecast_anomaly_prob": float(forecast_probs.mean()),
            "n_uncertain": len(uncertain_nodes),
            "n_forecast": int(forecast_flags.sum()),
            "llm_calls": stats.llm_calls,
            "budget_left": llm_budget_remaining,
        })
        if sim_step % 50 == 0:
            logger.info(
                f"step={sim_step:>4} | mean_rew={rewards.mean():.4f} "
                f"| llm_calls={stats.llm_calls} | overrides={stats.llm_overrides} "
                f"| queue={obs[:, 6].mean():.4f}"
            )
        sim_step += 1

    env.close()
    wall = time.time() - wall_start
    summary = stats.summary()
    summary.update({
        "intervention_rate": intervention_rate,
        "risk_weights": risk_weights,
        "failure_injection_rate": failure_rate,
        "sim_seconds": args.sim_seconds,
        "action_interval": args.action_interval,
        "llm_budget_used": llm_budget - llm_budget_remaining,
        "fallback_to_rl": fallback_to_rl,
        "llm_config_path": str(llm_config_path) if not args.mock_llm else None,
        "llm_model": config.get("OPENAI_API_MODEL") if not args.mock_llm else "mock",
        "llm_api_base": config.get("OPENAI_API_BASE") if not args.mock_llm else None,
        "total_reward": float(total_rewards.sum()),
        "mean_reward": float(total_rewards.mean()),
        "wall_clock_s": round(wall, 1),
        "model_path": str(model_path),
    })
    per_junction = stats.per_junction_summary()
    (output_dir / "step_log.json").write_text(json.dumps(step_log, indent=2), encoding="utf-8")
    (output_dir / "intervention_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "combined_results.json").write_text(
        json.dumps({"inference_steps": step_log, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    _save_per_junction_csv(per_junction, output_dir / "per_junction_results.csv")
    logger.info(f"Live CityFlow SafeGAT complete | reward={total_rewards.sum():.2f} | outputs={output_dir}")


if __name__ == "__main__":
    main()
