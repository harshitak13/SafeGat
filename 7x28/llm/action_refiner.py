"""
llm/action_refiner.py

SafeGATRefiner — orchestrates the full SafeGAT-LLM pipeline for one intersection.

Pipeline per call to refine()
------------------------------
1. ScenarioDetector   → extend anomaly_tags with detected conditions
2. InterventionGate   → score confidence + anomaly severity
3. If gate opens:
       TrafficPromptBuilder → build prompt
       LLMGateway           → query LLM, parse JSON
       (apply override or accept)
4. SafetyShield       → validate / repair final action
5. DecisionLogger     → append JSONL record

Source: SafeGAT-LLM scaffold (llm/action_refiner.py).
"""

from __future__ import annotations

from typing import Optional

from .decision_logger    import DecisionLogger
from .intervention_gate  import InterventionGate
from .llm_gateway        import LLMGateway
from .safety_shield      import SafetyShield
from .scenario_detector  import ScenarioDetector
from .traffic_prompt_builder import TrafficPromptBuilder
from .types              import LLMDecision, RLDecisionInfo, RefineResult


class SafeGATRefiner:
    """
    Entry point for the SafeGAT-LLM refinement pipeline.

    Instantiate once per training / inference run and call
    ``refine(info)`` for each flagged intersection per step.

    Parameters
    ----------
    detector         : ScenarioDetector     — anomaly detection
    gate             : InterventionGate     — uncertainty/anomaly gating
    prompt_builder   : TrafficPromptBuilder — LLM prompt construction
    llm_gateway      : LLMGateway           — LLM backend wrapper
    safety_shield    : SafetyShield         — post-LLM hard constraints
    decision_logger  : DecisionLogger | None — JSONL audit logger (optional)
    """

    def __init__(
        self,
        detector:        ScenarioDetector,
        gate:            InterventionGate,
        prompt_builder:  TrafficPromptBuilder,
        llm_gateway:     LLMGateway,
        safety_shield:   SafetyShield,
        decision_logger: Optional[DecisionLogger] = None,
    ) -> None:
        self.detector        = detector
        self.gate            = gate
        self.prompt_builder  = prompt_builder
        self.llm_gateway     = llm_gateway
        self.safety_shield   = safety_shield
        self.decision_logger = decision_logger

    def refine(self, info: RLDecisionInfo) -> RefineResult:
        """
        Run the full pipeline for one intersection at one simulation step.

        Parameters
        ----------
        info : RLDecisionInfo — complete decision context for this intersection

        Returns
        -------
        RefineResult — final action + audit metadata
        """
        # ── 1. Scenario detection ──────────────────────────────────────────────
        scenario = self.detector.detect(info.observation, info.metadata)
        # Merge newly detected tags into the info (dedup, sorted for stable logging)
        info.anomaly_tags = sorted(set(info.anomaly_tags + scenario["tags"]))

        # ── 2. Intervention gate ───────────────────────────────────────────────
        gate_result = self.gate.score(
            confidence_margin = info.confidence_margin,
            anomaly_tags      = info.anomaly_tags,
            corrupted         = scenario["corrupted"],
        )

        # ── 3. Conditional LLM call ────────────────────────────────────────────
        llm_called    = False
        llm_decision: Optional[LLMDecision] = None
        chosen_action = info.rl_action
        source        = "rl"
        trigger_reason = ",".join(gate_result.reasons) if gate_result.reasons else "none"

        if gate_result.should_intervene:
            llm_called = True
            prompt     = self.prompt_builder.build(info)
            llm_decision = self.llm_gateway.query(
                prompt, label=info.intersection_id
            )
            if llm_decision.decision == "override":
                chosen_action = llm_decision.final_phase
                source        = "llm_override"
            else:
                source = "llm_accept"

        # ── 4. Safety shield ───────────────────────────────────────────────────
        shield = self.safety_shield.validate(
            proposed_action = chosen_action,
            legal_actions   = info.legal_actions,
            phase_runtime   = int(info.metadata.get("phase_runtime", 0)),
            current_phase   = info.phase,
            metadata        = info.metadata,
        )

        result = RefineResult(
            final_action    = shield.action,
            source          = source,
            trigger_reason  = trigger_reason,
            safety_adjusted = shield.adjusted,
            llm_called      = llm_called,
            llm_decision    = llm_decision,
            debug={
                "gate":           gate_result.score_breakdown,
                "shield_reason":  shield.reason,
                "scenario_tags":  info.anomaly_tags,
            },
        )

        # ── 5. Audit logging ───────────────────────────────────────────────────
        if self.decision_logger is not None:
            self.decision_logger.log({
                "intersection_id":   info.intersection_id,
                "phase":             info.phase,
                "rl_action":         info.rl_action,
                "final_action":      result.final_action,
                "source":            result.source,
                "trigger_reason":    result.trigger_reason,
                "safety_adjusted":   result.safety_adjusted,
                "confidence_margin": info.confidence_margin,
                "action_scores":     info.action_scores,
                "anomaly_tags":      info.anomaly_tags,
                "llm_called":        result.llm_called,
                "llm_decision":      (
                    None
                    if result.llm_decision is None
                    else result.llm_decision.parsed
                ),
                "debug": result.debug,
            })

        return result
