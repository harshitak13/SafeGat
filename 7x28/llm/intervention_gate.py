"""
llm/intervention_gate.py

Decision gate for selective LLM intervention.

The risk score exposes the five SafeGAT Eq. 31 weights explicitly:
uncertainty, anomaly, queue pressure, waiting pressure, and safety pressure.
The score ranks candidates under the top-K budget; hard low-confidence,
anomaly, queue, waiting, corruption, forecast, or safety triggers open the gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_RISK_WEIGHTS: Dict[str, float] = {
    "uncertainty": 2.0,
    "anomaly": 1.5,
    "queue": 1.0,
    "waiting": 0.5,
    "safety": 1.0,
}


@dataclass
class GateDecision:
    should_intervene: bool
    reasons: List[str]
    score_breakdown: Dict[str, float]


class InterventionGate:
    def __init__(
        self,
        confidence_threshold: float = 0.05,
        anomaly_weight: float = DEFAULT_RISK_WEIGHTS["anomaly"],
        corruption_weight: float = 1.0,
        low_conf_weight: float = DEFAULT_RISK_WEIGHTS["uncertainty"],
        queue_weight: float = DEFAULT_RISK_WEIGHTS["queue"],
        wait_weight: float = DEFAULT_RISK_WEIGHTS["waiting"],
        safety_weight: float = DEFAULT_RISK_WEIGHTS["safety"],
        forecast_weight: float = 1.0,
        forecast_threshold: float = 0.65,
        intervention_budget: int = 8,
        queue_trigger_threshold: float = 0.85,
        wait_trigger_threshold: float = 0.85,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.anomaly_weight = anomaly_weight
        self.corruption_weight = corruption_weight
        self.low_conf_weight = low_conf_weight
        self.queue_weight = queue_weight
        self.wait_weight = wait_weight
        self.safety_weight = safety_weight
        self.forecast_weight = forecast_weight
        self.forecast_threshold = forecast_threshold
        self.intervention_budget = intervention_budget
        self.queue_trigger_threshold = queue_trigger_threshold
        self.wait_trigger_threshold = wait_trigger_threshold

    @property
    def risk_weights(self) -> Dict[str, float]:
        return {
            "uncertainty": self.low_conf_weight,
            "anomaly": self.anomaly_weight,
            "queue": self.queue_weight,
            "waiting": self.wait_weight,
            "safety": self.safety_weight,
        }

    @staticmethod
    def _bounded(value: Any, default: float = 0.0) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    def score(
        self,
        confidence_margin: float,
        anomaly_tags: Iterable[str],
        corrupted: bool,
        forecast_probability: float = 0.0,
        metadata: Dict[str, Any] | None = None,
    ) -> GateDecision:
        metadata = metadata or {}
        anomaly_tags = list(anomaly_tags)
        reasons: List[str] = []
        breakdown: Dict[str, float] = {}

        low_conf = float(confidence_margin < self.confidence_threshold)
        uncertainty = (
            (self.confidence_threshold - confidence_margin) / self.confidence_threshold
            if self.confidence_threshold > 0.0 and low_conf
            else 0.0
        )
        breakdown["uncertainty"] = self._bounded(uncertainty) * self.low_conf_weight
        if low_conf:
            reasons.append("low_confidence")

        anomaly_count = float(len(anomaly_tags))
        breakdown["anomaly"] = min(1.0, anomaly_count) * self.anomaly_weight
        if anomaly_count > 0:
            reasons.append("anomaly_detected")

        queue_pressure = self._bounded(
            metadata.get("queue_pressure", metadata.get("current_queue", 0.0))
        )
        wait_pressure = self._bounded(
            metadata.get("wait_pressure", metadata.get("waiting_pressure", 0.0))
        )
        transition_confidence = self._bounded(
            metadata.get("transition_model_confidence", 1.0),
            default=1.0,
        )
        safety_pressure = max(
            float(bool(metadata.get("emergency_vehicle", False))),
            float(bool(metadata.get("accident_flag", False))),
            float(bool(metadata.get("yellow_phase", False))),
            1.0 - transition_confidence,
        )

        breakdown["queue_pressure"] = queue_pressure * self.queue_weight
        breakdown["waiting_pressure"] = wait_pressure * self.wait_weight
        breakdown["safety_pressure"] = safety_pressure * self.safety_weight

        if queue_pressure >= self.queue_trigger_threshold:
            reasons.append("queue_pressure")
        if wait_pressure >= self.wait_trigger_threshold:
            reasons.append("waiting_pressure")
        if safety_pressure > 0.0:
            reasons.append("safety_pressure")

        breakdown["corrupted_observation"] = float(corrupted) * self.corruption_weight
        if corrupted:
            reasons.append("corrupted_observation")

        forecast_probability = self._bounded(forecast_probability)
        breakdown["forecast_anomaly"] = forecast_probability * self.forecast_weight
        if forecast_probability >= self.forecast_threshold:
            reasons.append("forecast_anomaly")

        total = sum(breakdown.values())
        return GateDecision(
            should_intervene=bool(reasons),
            reasons=reasons,
            score_breakdown={
                **breakdown,
                "total": total,
                "risk_weights": self.risk_weights,
            },
        )

    def select_top_k(
        self,
        candidate_items: List[Tuple[str, GateDecision]],
    ) -> List[str]:
        ranked = sorted(
            candidate_items,
            key=lambda x: x[1].score_breakdown.get("total", 0.0),
            reverse=True,
        )
        return [
            item_id
            for item_id, gate in ranked[: self.intervention_budget]
            if gate.should_intervene
        ]
