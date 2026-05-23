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

from .anomaly_forecaster import GRUAnomalyForecaster
from .corridor_context import CorridorContextCache
from .decision_logger    import DecisionLogger
from .intervention_gate  import InterventionGate
from .llm_gateway        import LLMGateway
from .safety_shield      import SafetyShield
from .scenario_detector  import ScenarioDetector
from .traffic_prompt_builder import TrafficPromptBuilder
from .types              import LLMDecision, RLDecisionInfo, RefineResult


def _override_quality(info: RLDecisionInfo, final_action: int) -> dict:
    """Q-value proxy for whether an override improved on the RL proposal."""
    scores = list(info.action_scores or [])
    rl_action = int(info.rl_action)
    final_action = int(final_action)
    if (
        rl_action < 0
        or final_action < 0
        or rl_action >= len(scores)
        or final_action >= len(scores)
    ):
        return {
            "available": False,
            "metric": "current_state_q_proxy",
            "reason": "action index outside action_scores",
        }
    q_rl = float(scores[rl_action])
    q_final = float(scores[final_action])
    return {
        "available": True,
        "metric": "current_state_q_proxy",
        "rl_action": rl_action,
        "final_action": final_action,
        "q_rl_action": q_rl,
        "q_final_action": q_final,
        "q_delta_final_minus_rl": q_final - q_rl,
        "helped": bool(q_final >= q_rl),
    }


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
        corridor_cache:  Optional[CorridorContextCache] = None,
        anomaly_forecaster: Optional[GRUAnomalyForecaster] = None,
    ) -> None:
        self.detector        = detector
        self.gate            = gate
        self.prompt_builder  = prompt_builder
        self.llm_gateway     = llm_gateway
        self.safety_shield   = safety_shield
        self.decision_logger = decision_logger
        self.corridor_cache  = corridor_cache or CorridorContextCache()
        self.anomaly_forecaster = anomaly_forecaster or GRUAnomalyForecaster()

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
        if "forecast_anomaly_prob" not in info.metadata:
            info.metadata["forecast_anomaly_prob"] = (
                self.anomaly_forecaster.update_and_predict(
                    info.intersection_id,
                    info.observation,
                )
            )
        forecast_probability = float(info.metadata.get("forecast_anomaly_prob", 0.0))

        # ── 2. Intervention gate ───────────────────────────────────────────────
        gate_result = self.gate.score(
            confidence_margin = info.confidence_margin,
            anomaly_tags      = info.anomaly_tags,
            corrupted         = scenario["corrupted"],
            forecast_probability = forecast_probability,
            metadata          = info.metadata,
        )

        # ── 3. Conditional LLM call ────────────────────────────────────────────
        llm_called    = False
        llm_decision: Optional[LLMDecision] = None
        chosen_action = info.rl_action
        source        = "rl"
        trigger_reason = ",".join(gate_result.reasons) if gate_result.reasons else "none"
        corridor_context = "none"

        if gate_result.should_intervene:
            llm_called = True
            corridor_context = self.corridor_cache.build_token(
                info.intersection_id,
                info.neighbor_summary.keys(),
            )
            info.metadata["corridor_context"] = corridor_context
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
        quality = _override_quality(info, shield.action)

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
                "transition_model_confidence": (
                    self.safety_shield._transition_confidence(info.metadata)
                ),
                "scenario_tags":  info.anomaly_tags,
                "forecast_anomaly_prob": forecast_probability,
                "corridor_context": corridor_context,
                "override_quality": quality,
            },
        )

        # ── 5. Audit logging ───────────────────────────────────────────────────
        if llm_called and llm_decision is not None:
            self.corridor_cache.update(
                intersection_id = info.intersection_id,
                rl_action       = info.rl_action,
                final_action    = result.final_action,
                source          = result.source,
                reason          = llm_decision.reason,
                anomaly_tags    = info.anomaly_tags,
                step            = info.metadata.get("sim_step"),
            )

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
                "forecast_anomaly_prob": forecast_probability,
                "action_scores":     info.action_scores,
                "anomaly_tags":      info.anomaly_tags,
                "llm_called":        result.llm_called,
                "llm_decision":      (
                    None
                    if result.llm_decision is None
                    else result.llm_decision.parsed
                ),
                "override_quality":  quality,
                "debug": result.debug,
            })

        return result
