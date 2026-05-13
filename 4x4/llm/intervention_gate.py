"""
llm/intervention_gate.py

InterventionGate — decides when the LLM should intervene.

The gate scores three signals:
    1. Low confidence  (Q-margin Δ below threshold)
    2. Anomaly tags    (from ScenarioDetector)
    3. Corrupted obs   (NaN / packet-loss / empty)

Any non-zero total triggers LLM intervention.  The gate also provides
``select_top_k`` for budget-constrained multi-intersection selection.

Source: SafeGAT-LLM scaffold (llm/intervention_gate.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class GateDecision:
    """
    Attributes
    ----------
    should_intervene : bool  — True if LLM should be called
    reasons          : List[str]  — human-readable trigger reasons
    score_breakdown  : Dict[str, float]  — per-signal scores + total
    """
    should_intervene: bool
    reasons:          List[str]
    score_breakdown:  Dict[str, float]


class InterventionGate:
    """
    Decision-theoretic gate for selective LLM intervention.

    Parameters
    ----------
    confidence_threshold  : float — Δ below this → low_confidence trigger
    anomaly_weight        : float — weight applied to anomaly tag count
    corruption_weight     : float — weight applied to corruption flag
    low_conf_weight       : float — weight applied to low-confidence flag
    intervention_budget   : int   — max nodes to return from select_top_k
    """

    def __init__(
        self,
        confidence_threshold: float = 0.15,
        anomaly_weight:       float = 1.0,
        corruption_weight:    float = 1.0,
        low_conf_weight:      float = 1.0,
        intervention_budget:  int   = 8,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.anomaly_weight       = anomaly_weight
        self.corruption_weight    = corruption_weight
        self.low_conf_weight      = low_conf_weight
        self.intervention_budget  = intervention_budget

    def score(
        self,
        confidence_margin: float,
        anomaly_tags:      Iterable[str],
        corrupted:         bool,
    ) -> GateDecision:
        """
        Score one intersection and decide whether the LLM should intervene.

        Parameters
        ----------
        confidence_margin : float — Q(a*) − Q(a_2nd); lower = more uncertain
        anomaly_tags      : iterable of str — tags from ScenarioDetector
        corrupted         : bool — True if obs integrity is suspect

        Returns
        -------
        GateDecision
        """
        reasons:   List[str]        = []
        breakdown: Dict[str, float] = {}

        low_conf = float(confidence_margin < self.confidence_threshold)
        breakdown["low_confidence"] = low_conf * self.low_conf_weight
        if low_conf:
            reasons.append("low_confidence")

        anomaly_count = float(len(list(anomaly_tags)))
        breakdown["anomaly_tags"] = anomaly_count * self.anomaly_weight
        if anomaly_count > 0:
            reasons.append("anomaly_detected")

        breakdown["corrupted_observation"] = float(corrupted) * self.corruption_weight
        if corrupted:
            reasons.append("corrupted_observation")

        total = sum(breakdown.values())
        return GateDecision(
            should_intervene = total > 0.0,
            reasons          = reasons,
            score_breakdown  = {**breakdown, "total": total},
        )

    def select_top_k(
        self,
        candidate_items: List[Tuple[str, GateDecision]],
    ) -> List[str]:
        """
        From a list of (intersection_id, GateDecision) pairs,
        return the top-k intersection IDs ranked by gate total score,
        up to ``intervention_budget``.
        """
        ranked = sorted(
            candidate_items,
            key     = lambda x: x[1].score_breakdown.get("total", 0.0),
            reverse = True,
        )
        return [
            item_id
            for item_id, gate in ranked[: self.intervention_budget]
            if gate.should_intervene
        ]
