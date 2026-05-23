"""
Offline SafeGAT adapter for CityFlow-style datasets.

This runner lets the Hangzhou, Jinan, and NewYork roadnet/flow folders exercise
the modified SafeGAT control stack without requiring a SUMO conversion first.
It builds graph metadata from roadnet JSON, creates lightweight demand-derived
observations from anon flow files, and applies the updated LLM-risk and safety
modules in audit mode.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class CityFlowNetwork:
    controlled_tls: List[str]
    neighbors: Dict[str, List[str]]
    incoming_roads: Dict[str, List[str]]
    legal_actions: Dict[str, List[int]]
    edge_index: List[Tuple[int, int]]


def load_roadnet(path: Path) -> CityFlowNetwork:
    data = json.loads(path.read_text(encoding="utf-8"))
    intersections = data.get("intersections", [])
    roads = data.get("roads", [])

    controlled = [
        item["id"]
        for item in intersections
        if not item.get("virtual", False)
    ]
    controlled_set = set(controlled)
    index = {tls_id: idx for idx, tls_id in enumerate(controlled)}

    incoming: Dict[str, List[str]] = {tls_id: [] for tls_id in controlled}
    neighbors: Dict[str, set[str]] = {tls_id: set() for tls_id in controlled}

    for road in roads:
        road_id = road.get("id")
        start = road.get("startIntersection")
        end = road.get("endIntersection")
        if end in controlled_set and road_id:
            incoming[end].append(road_id)
        if start in controlled_set and end in controlled_set:
            neighbors[start].add(end)
            neighbors[end].add(start)

    legal_actions: Dict[str, List[int]] = {}
    for item in intersections:
        tls_id = item.get("id")
        if tls_id not in controlled_set:
            continue
        phases = item.get("trafficLight", {}).get("lightphases", [])
        legal = [
            phase_idx
            for phase_idx, phase in enumerate(phases)
            if phase.get("availableRoadLinks")
        ]
        legal_actions[tls_id] = legal or list(range(min(4, max(1, len(phases)))))

    edge_index = []
    for src, nbs in neighbors.items():
        for dst in sorted(nbs):
            edge_index.append((index[src], index[dst]))

    return CityFlowNetwork(
        controlled_tls=controlled,
        neighbors={key: sorted(value) for key, value in neighbors.items()},
        incoming_roads=incoming,
        legal_actions=legal_actions,
        edge_index=edge_index,
    )


def load_flow_counts(path: Path, bucket_seconds: int) -> Dict[int, Counter[str]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    buckets: Dict[int, Counter[str]] = defaultdict(Counter)
    for record in records:
        route = record.get("route") or []
        if not route:
            continue
        start = int(float(record.get("startTime", 0)))
        end = int(float(record.get("endTime", start)))
        interval = max(float(record.get("interval", 1.0)), 1.0)
        bucket = start // bucket_seconds
        count = max(1, int((end - start) / interval) + 1)
        buckets[bucket][route[0]] += count
    return buckets


def choose_default_impl(city_dir: Path, num_nodes: int) -> Path:
    root = city_dir.parents[1]
    return root / ("7x28" if num_nodes > 64 else "4x4")


def import_safegat_modules(impl_dir: Path):
    sys.path.insert(0, str(impl_dir))
    from llm.anomaly_forecaster import GRUAnomalyForecaster
    from llm.corridor_context import CorridorContextCache
    from llm.intervention_gate import InterventionGate
    from llm.safety_shield import SafetyShield
    from llm.scenario_detector import ScenarioDetector
    from llm.traffic_prompt_builder import TrafficPromptBuilder
    from llm.types import RLDecisionInfo
    from utils.margin import compute_margin_from_scores

    return {
        "GRUAnomalyForecaster": GRUAnomalyForecaster,
        "CorridorContextCache": CorridorContextCache,
        "InterventionGate": InterventionGate,
        "SafetyShield": SafetyShield,
        "ScenarioDetector": ScenarioDetector,
        "TrafficPromptBuilder": TrafficPromptBuilder,
        "RLDecisionInfo": RLDecisionInfo,
        "compute_margin_from_scores": compute_margin_from_scores,
    }


def build_observation(
    tls_id: str,
    network: CityFlowNetwork,
    road_counts: Counter[str],
    phase: int,
) -> np.ndarray:
    incoming = network.incoming_roads.get(tls_id, [])
    counts = np.asarray([road_counts.get(road_id, 0) for road_id in incoming], dtype=float)
    if counts.size == 0:
        occ = np.zeros(4, dtype=float)
        queue = 0.0
    else:
        normalized = np.clip(counts / max(10.0, counts.max()), 0.0, 1.0)
        occ = np.zeros(4, dtype=float)
        for idx, value in enumerate(normalized):
            occ[idx % 4] = max(occ[idx % 4], value)
        queue = float(np.clip(counts.sum() / 40.0, 0.0, 1.0))
    return np.asarray([phase / 3.0, len(incoming) / 4.0, *occ, queue, 0.0], dtype=float)


def heuristic_scores(observation: np.ndarray, num_actions: int) -> List[float]:
    occ = observation[2:6]
    scores = np.full(num_actions, -0.05, dtype=float)
    for action in range(num_actions):
        scores[action] = float(occ[action % 4] - 0.03 * action)
    return scores.tolist()


def neighbor_summary(tls_id: str, network: CityFlowNetwork, phases: Dict[str, int]) -> Dict[str, Any]:
    return {
        nb: {"mean_occ": 0.0, "rl_action": int(phases.get(nb, 0))}
        for nb in network.neighbors.get(tls_id, [])
    }


def time_of_day(step: int, bucket_seconds: int) -> str:
    hour = ((step * bucket_seconds) // 3600) % 24
    if 7 <= hour < 10:
        return "morning_peak"
    if 16 <= hour < 19:
        return "evening_peak"
    if 10 <= hour < 16:
        return "midday"
    return "off_peak"


def propagation_status(summary: Dict[str, Any]) -> str:
    if not summary:
        return "isolated_or_unknown"
    occupancies = [
        float(value.get("mean_occ", 0.0))
        for value in summary.values()
        if isinstance(value, dict)
    ]
    if not occupancies:
        return "neighbor_pressure_unknown"
    max_occ = max(occupancies)
    mean_occ = float(np.mean(occupancies))
    if max_occ >= 0.70:
        return f"downstream_congested max_neighbor_occ={max_occ:.3f}"
    return f"stable_neighbors mean_neighbor_occ={mean_occ:.3f}"


def run_offline(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).resolve()
    roadnet_path = dataset_dir / args.roadnet
    flow_path = dataset_dir / args.flow
    output_dir = dataset_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    network = load_roadnet(roadnet_path)
    impl_dir = Path(args.impl_dir).resolve() if args.impl_dir else choose_default_impl(dataset_dir, len(network.controlled_tls))
    mods = import_safegat_modules(impl_dir)
    max_nodes_per_step = args.max_nodes_per_step
    if args.intervention_rate is not None:
        max_nodes_per_step = max(
            1,
            min(
                len(network.controlled_tls),
                int(math.ceil(args.intervention_rate * len(network.controlled_tls))),
            ),
        )
    intervention_rate = max_nodes_per_step / max(1, len(network.controlled_tls))

    detector = mods["ScenarioDetector"](
        queue_spike_threshold=args.queue_spike_threshold,
        zero_fraction_corruption_threshold=args.zero_fraction_corruption_threshold,
    )
    gate = mods["InterventionGate"](
        confidence_threshold=args.confidence_threshold,
        intervention_budget=max_nodes_per_step,
        forecast_threshold=args.forecast_anomaly_threshold,
        low_conf_weight=args.weight_uncertainty,
        anomaly_weight=args.weight_anomaly,
        queue_weight=args.weight_queue,
        wait_weight=args.weight_waiting,
        safety_weight=args.weight_safety,
    )
    forecaster = mods["GRUAnomalyForecaster"](
        threshold=args.forecast_anomaly_threshold,
        horizon=args.forecast_horizon_steps,
        history_len=args.forecast_history_len,
        queue_spike_threshold=args.queue_spike_threshold,
        zero_fraction_corruption_threshold=args.zero_fraction_corruption_threshold,
    )
    shield = mods["SafetyShield"](
        min_green_hold=args.min_green_hold,
        transition_confidence_threshold=args.transition_confidence_threshold,
    )
    prompt_builder = mods["TrafficPromptBuilder"]()
    corridor_cache = mods["CorridorContextCache"]()
    RLDecisionInfo = mods["RLDecisionInfo"]
    margin_fn = mods["compute_margin_from_scores"]

    flow_counts = load_flow_counts(flow_path, args.bucket_seconds)
    phases = {tls_id: 0 for tls_id in network.controlled_tls}
    runtimes = {tls_id: 0 for tls_id in network.controlled_tls}
    queue_baseline = {tls_id: 0.0 for tls_id in network.controlled_tls}
    summary = Counter()
    step_log = []
    decisions_path = output_dir / "safegat_cityflow_decisions.jsonl"
    failure_rng = random.Random(args.failure_seed)
    all_step_queues: List[float] = []

    with decisions_path.open("w", encoding="utf-8") as decisions:
        for step in range(args.steps):
            road_counts = flow_counts.get(step, Counter())
            candidates = []
            payloads = []

            for tls_id in network.controlled_tls:
                obs = build_observation(tls_id, network, road_counts, phases[tls_id])
                scores = heuristic_scores(obs, max(1, max(network.legal_actions[tls_id]) + 1))
                rl_action = int(np.argmax(scores))
                margin = float(margin_fn(scores))
                current_queue = float(obs[6])
                baseline_queue = float(queue_baseline[tls_id])
                metadata = {
                    "transition_model_source": "cityflow",
                    "transition_model_confidence": args.transition_model_confidence,
                    "phase_runtime": int(runtimes[tls_id]),
                    "sim_step": int(step),
                    "sim_time_s": int(step * args.bucket_seconds),
                    "time_of_day": time_of_day(step, args.bucket_seconds),
                    "historical_baseline_queue": round(baseline_queue, 4),
                    "current_queue": round(current_queue, 4),
                    "queue_pressure": round(max(current_queue, current_queue - baseline_queue), 4),
                    "waiting_pressure": round(float(obs[2:6].mean()), 4),
                    "event_details": "offline_cityflow_demand_bucket",
                }
                scenario = detector.detect(obs, metadata)
                forecast_prob = forecaster.update_and_predict(tls_id, obs)
                gate_result = gate.score(
                    confidence_margin=margin,
                    anomaly_tags=scenario["tags"],
                    corrupted=scenario["corrupted"],
                    forecast_probability=forecast_prob,
                    metadata=metadata,
                )
                if gate_result.should_intervene:
                    candidates.append((tls_id, gate_result))
                payloads.append((tls_id, obs, scores, rl_action, margin, scenario, forecast_prob, gate_result, metadata))

            selected = set(gate.select_top_k(candidates))
            step_overrides = 0
            step_adjusted = 0
            step_queues = []

            for tls_id, obs, scores, rl_action, margin, scenario, forecast_prob, gate_result, metadata in payloads:
                step_queues.append(float(obs[6]))
                if tls_id not in selected:
                    final_action = rl_action
                    source = "rl"
                    llm_called = False
                    llm_decision = None
                    prompt_preview = None
                else:
                    llm_called = True
                    context = corridor_cache.build_token(tls_id, network.neighbors.get(tls_id, []))
                    nb_summary = neighbor_summary(tls_id, network, phases)
                    metadata = {
                        **metadata,
                        "forecast_anomaly_prob": float(forecast_prob),
                        "forecast_horizon": args.forecast_horizon_steps,
                        "corridor_context": context,
                        "propagation_status": propagation_status(nb_summary),
                        "observation_summary": f"offline_cityflow_obs={np.round(obs, 3).tolist()}",
                    }
                    info = RLDecisionInfo(
                        intersection_id=tls_id,
                        observation=obs.tolist(),
                        phase=int(phases[tls_id]),
                        rl_action=rl_action,
                        action_scores=[float(item) for item in scores],
                        confidence_margin=margin,
                        legal_actions=network.legal_actions[tls_id],
                        neighbor_summary=nb_summary,
                        anomaly_tags=list(scenario["tags"]),
                        metadata=metadata,
                    )
                    prompt_preview = prompt_builder.build(info)[:500]
                    final_action = rl_action
                    source = "llm_audit_accept"
                    llm_decision = {
                        "decision": "accept",
                        "final_phase": int(rl_action),
                        "reason": "offline audit mode; no external LLM call",
                    }
                    if failure_rng.random() < args.failure_injection_rate:
                        source = "llm_failure_fallback"
                        summary["llm_failures"] += 1
                        llm_decision = {
                            "decision": "accept",
                            "final_phase": int(rl_action),
                            "reason": "injected failure; fallback to RL action",
                        }

                shield_result = shield.validate(
                    proposed_action=final_action,
                    legal_actions=network.legal_actions[tls_id],
                    phase_runtime=int(runtimes[tls_id]),
                    current_phase=int(phases[tls_id]),
                    metadata={
                        "transition_model_source": "cityflow",
                        "transition_model_confidence": args.transition_model_confidence,
                    },
                )

                if shield_result.adjusted:
                    step_adjusted += 1
                if shield_result.action != rl_action:
                    step_overrides += 1

                old_phase = phases[tls_id]
                phases[tls_id] = int(shield_result.action)
                runtimes[tls_id] = runtimes[tls_id] + 1 if phases[tls_id] == old_phase else 1

                if llm_called:
                    corridor_cache.update(
                        intersection_id=tls_id,
                        rl_action=rl_action,
                        final_action=shield_result.action,
                        source=source,
                        reason=shield_result.reason,
                        anomaly_tags=scenario["tags"],
                        step=step,
                    )
                    summary["llm_calls"] += 1
                    decisions.write(json.dumps({
                        "step": step,
                        "intersection_id": tls_id,
                        "phase": old_phase,
                        "rl_action": rl_action,
                        "final_action": shield_result.action,
                        "source": source,
                        "confidence_margin": margin,
                        "forecast_anomaly_prob": forecast_prob,
                        "anomaly_tags": scenario["tags"],
                        "gate": gate_result.score_breakdown,
                        "risk_weights": gate.risk_weights,
                        "safety_adjusted": shield_result.adjusted,
                        "shield_reason": shield_result.reason,
                        "llm_decision": llm_decision,
                        "prompt_preview": prompt_preview,
                    }) + "\n")

            summary["steps"] += 1
            summary["safety_adjustments"] += step_adjusted
            summary["rl_overrides"] += step_overrides
            summary["queue_samples"] += len(step_queues)
            summary["queue_sum"] += float(np.sum(step_queues))
            all_step_queues.append(float(np.mean(step_queues)) if step_queues else 0.0)
            for tls_id, obs, *_ in payloads:
                queue_baseline[tls_id] = 0.98 * queue_baseline[tls_id] + 0.02 * float(obs[6])
            step_log.append({
                "step": step,
                "active_flow_roads": len(road_counts),
                "selected_for_llm_audit": len(selected),
                "safety_adjustments": step_adjusted,
                "rl_overrides": step_overrides,
                "mean_queue": float(np.mean(step_queues)) if step_queues else 0.0,
                "intervention_rate": intervention_rate,
            })

    graph_payload = {
        "num_nodes": len(network.controlled_tls),
        "num_edges": len(network.edge_index),
        "controlled_tls": network.controlled_tls,
        "neighbors": network.neighbors,
        "edge_index": network.edge_index,
    }
    (output_dir / "cityflow_graph.json").write_text(json.dumps(graph_payload, indent=2), encoding="utf-8")
    (output_dir / "step_log.json").write_text(json.dumps(step_log, indent=2), encoding="utf-8")
    summary_payload = dict(summary)
    queue_samples = max(1, int(summary_payload.pop("queue_samples", 0)))
    queue_sum = float(summary_payload.pop("queue_sum", 0.0))
    summary_payload["mean_queue"] = queue_sum / queue_samples
    summary_payload["att_proxy"] = (
        float(np.mean(all_step_queues)) * args.bucket_seconds
        if all_step_queues else 0.0
    )
    summary_payload["intervention_rate"] = intervention_rate
    summary_payload["max_nodes_per_step"] = max_nodes_per_step
    summary_payload["risk_weights"] = gate.risk_weights
    summary_payload["confidence_threshold"] = args.confidence_threshold
    summary_payload["failure_injection_rate"] = args.failure_injection_rate
    (output_dir / "intervention_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Loaded {len(network.controlled_tls)} controlled intersections from {roadnet_path.name}")
    print(f"Using modified SafeGAT modules from: {impl_dir}")
    print(f"Saved offline audit outputs to: {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run modified SafeGAT audit on CityFlow JSON data.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--roadnet", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--impl-dir", default=None, help="SafeGAT implementation folder, e.g. ../../4x4")
    parser.add_argument("--output-dir", default="safegat_cityflow_output")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--bucket-seconds", type=int, default=30)
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--max-nodes-per-step", type=int, default=4)
    parser.add_argument("--intervention-rate", type=float, default=None, help="Optional K/N rate; overrides --max-nodes-per-step")
    parser.add_argument("--failure-injection-rate", type=float, default=0.0, help="Randomly reject this fraction of selected LLM audit responses")
    parser.add_argument("--failure-seed", type=int, default=42)
    parser.add_argument("--weight-uncertainty", type=float, default=2.0)
    parser.add_argument("--weight-anomaly", type=float, default=1.5)
    parser.add_argument("--weight-queue", type=float, default=1.0)
    parser.add_argument("--weight-waiting", type=float, default=0.5)
    parser.add_argument("--weight-safety", type=float, default=1.0)
    parser.add_argument("--min-green-hold", type=int, default=3)
    parser.add_argument("--transition-model-confidence", type=float, default=0.5)
    parser.add_argument("--transition-confidence-threshold", type=float, default=0.75)
    parser.add_argument("--queue-spike-threshold", type=float, default=0.85)
    parser.add_argument("--zero-fraction-corruption-threshold", type=float, default=0.90)
    parser.add_argument("--forecast-anomaly-threshold", type=float, default=0.65)
    parser.add_argument("--forecast-horizon-steps", type=int, default=3)
    parser.add_argument("--forecast-history-len", type=int, default=4)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_offline(args)


if __name__ == "__main__":
    main()
