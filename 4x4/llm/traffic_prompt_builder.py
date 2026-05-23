"""
Build structured SafeGAT LLM prompts from RLDecisionInfo.
"""

from __future__ import annotations

from .types import RLDecisionInfo


_OBJECTIVE = (
    "Objective: minimize network average travel time, queue spillback, and "
    "avoidable stopping while preserving hard safety constraints. Override the "
    "RL action only when the context gives a traffic-engineering reason that a "
    "different legal phase will reduce downstream risk or clear a safety event."
)

_RULES = (
    "Rules: emergency vehicle -> serve its phase; "
    "neighbor occ>0.70 -> release pressure away from congested direction; "
    "yellow phases ONLY change for emergencies; "
    "never starve any direction >3 consecutive cycles."
)


class TrafficPromptBuilder:
    def build(self, info: RLDecisionInfo) -> str:
        neighbor_lines = [
            f"  - {key}: {value}"
            for key, value in info.neighbor_summary.items()
        ] or ["  - none"]

        anomaly_text = ", ".join(info.anomaly_tags) if info.anomaly_tags else "none"
        action_scores = ", ".join(
            f"{i}:{score:.4f}" for i, score in enumerate(info.action_scores)
        )
        legal_actions = ", ".join(str(x) for x in info.legal_actions)
        obs_summary = info.metadata.get("observation_summary", str(info.observation))
        corridor_context = info.metadata.get("corridor_context", "none")
        forecast_prob = float(info.metadata.get("forecast_anomaly_prob", 0.0))

        baseline_queue = info.metadata.get("historical_baseline_queue", "unknown")
        current_queue = info.metadata.get("current_queue", "unknown")
        queue_pressure = info.metadata.get("queue_pressure", "unknown")
        time_of_day = info.metadata.get("time_of_day", "unknown")
        event_details = info.metadata.get("event_details", "none")
        propagation_status = info.metadata.get("propagation_status", "unknown")
        waiting_pressure = info.metadata.get("waiting_pressure", "unknown")

        return (
            "You are a traffic-signal control verifier for a multi-intersection "
            "graph RL controller.\n"
            "Return valid JSON ONLY - no markdown, no extra text.\n\n"
            f"{_OBJECTIVE}\n"
            f"{_RULES}\n\n"
            "Semantic context:\n"
            f"  - time_of_day: {time_of_day}\n"
            f"  - historical_baseline_queue: {baseline_queue}\n"
            f"  - current_queue: {current_queue}\n"
            f"  - queue_pressure: {queue_pressure}\n"
            f"  - waiting_pressure: {waiting_pressure}\n"
            f"  - event_details: {event_details}\n"
            f"  - propagation_status: {propagation_status}\n\n"
            f"Intersection ID  : {info.intersection_id}\n"
            f"Current phase    : {info.phase}\n"
            f"RL proposed phase: {info.rl_action}\n"
            f"Legal phases     : [{legal_actions}]\n"
            f"Confidence margin: {info.confidence_margin:.4f}\n"
            f"Action scores    : {action_scores}\n"
            f"Anomaly tags     : {anomaly_text}\n"
            f"Forecast p(+2-3) : {forecast_prob:.3f}\n"
            f"Observation      : {obs_summary}\n"
            "Neighbor summary:\n" + "\n".join(neighbor_lines) + "\n"
            "Corridor context (cached recent LLM decisions; avoid conflicts): "
            f"{corridor_context}\n\n"
            "Output schema:\n"
            '{"decision": "accept" or "override", '
            '"final_phase": integer, '
            '"reason": "brief technical reason"}'
        ).strip()
