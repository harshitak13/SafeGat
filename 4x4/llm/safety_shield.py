"""
llm/safety_shield.py

SafetyShield — hard rule-based post-processing layer.

Applied AFTER the LLM decision to enforce:
    1. Legal action check:   proposed action must be in legal_actions
    2. Minimum green hold:   a green phase must be held >= min_green_hold steps
                             before switching is permitted

Extend this class with:
    - Phase-conflict matrices
    - Starvation prevention counters
    - Emergency vehicle priority logic

Source: SafeGAT-LLM scaffold (llm/safety_shield.py), extended with yellow-lock
logic from iLLM-TSC2 (run_grid_llm.py SafetyLayer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Set

# Phase indices treated as yellow (must not be switched away from)
_YELLOW_PHASES: Set[int] = {1, 3}


@dataclass
class ShieldResult:
    """
    Attributes
    ----------
    action   : int  — final safe action to execute
    adjusted : bool — True if the shield changed the proposed action
    reason   : str  — short explanation of the adjustment (or "accepted")
    """
    action:   int
    adjusted: bool
    reason:   str


class SafetyShield:
    """
    Validates and, if necessary, repairs a proposed action.

    Parameters
    ----------
    min_green_hold : int — minimum steps a green phase must be held (default 3)
    """

    def __init__(
        self,
        min_green_hold: int = 3,
        transition_confidence_threshold: float = 0.75,
    ) -> None:
        self.min_green_hold = min_green_hold
        self.transition_confidence_threshold = transition_confidence_threshold

    def validate(
        self,
        proposed_action: int,
        legal_actions:   Iterable[int],
        phase_runtime:   int                = 0,
        current_phase:   Optional[int]      = None,
        metadata:        Optional[Dict]     = None,
    ) -> ShieldResult:
        """
        Validate ``proposed_action`` against safety constraints.

        Parameters
        ----------
        proposed_action : int           — action from LLM (or RL fallback)
        legal_actions   : Iterable[int] — phases allowed at this step
        phase_runtime   : int           — steps the current phase has been active
        current_phase   : int | None    — currently active phase index
        metadata        : dict | None   — extra context (unused here; reserved)

        Returns
        -------
        ShieldResult
        """
        metadata      = metadata or {}
        legal_actions = list(legal_actions)
        transition_confidence = self._transition_confidence(metadata)

        # Rule -1: missing or empty T_i. On real/offline datasets, do not repair
        # from a SUMO-derived default when the controller constraints are absent.
        if not legal_actions:
            if current_phase is not None:
                return ShieldResult(
                    action   = int(current_phase),
                    adjusted = proposed_action != current_phase,
                    reason   = "transition_model_missing_hold_current",
                )
            return ShieldResult(
                action   = int(proposed_action),
                adjusted = False,
                reason   = "accepted_no_transition_model",
            )

        # Rule 0: Yellow-phase lock — never switch away during yellow
        if current_phase is not None and current_phase in _YELLOW_PHASES:
            # If proposed action differs from current yellow phase, force hold
            if proposed_action != current_phase:
                return ShieldResult(
                    action   = int(current_phase),
                    adjusted = True,
                    reason   = "yellow_phase_lock",
                )

        # Rule 1: Conservative T_i fallback. If the transition/action set came
        # from incomplete real firmware or misspecified offline data, preserve
        # the safety guarantee by extending the current phase instead of switching.
        if (
            current_phase is not None
            and proposed_action != current_phase
            and transition_confidence < self.transition_confidence_threshold
        ):
            return ShieldResult(
                action   = int(current_phase),
                adjusted = True,
                reason   = (
                    "low_transition_confidence_hold_current "
                    f"({transition_confidence:.2f}<{self.transition_confidence_threshold:.2f})"
                ),
            )

        # Rule 2: Illegal action repair
        if proposed_action not in legal_actions:
            fallback = (
                current_phase
                if current_phase is not None and current_phase in legal_actions
                else legal_actions[0]
            )
            return ShieldResult(
                action   = int(fallback),
                adjusted = True,
                reason   = "illegal_action_repaired",
            )

        # Rule 3: Minimum green hold
        if (
            current_phase is not None
            and proposed_action != current_phase
            and phase_runtime < self.min_green_hold
            and current_phase not in _YELLOW_PHASES
        ):
            return ShieldResult(
                action   = int(current_phase),
                adjusted = True,
                reason   = f"minimum_green_hold ({phase_runtime}/{self.min_green_hold})",
            )

        return ShieldResult(action=int(proposed_action), adjusted=False, reason="accepted")

    def _transition_confidence(self, metadata: Dict[str, Any]) -> float:
        """Return confidence in T_i, defaulting to high confidence for SUMO."""
        confidence_keys = (
            "transition_model_confidence",
            "ti_confidence",
            "constraint_confidence",
            "legal_action_confidence",
        )
        for key in confidence_keys:
            if key in metadata and metadata[key] is not None:
                return max(0.0, min(1.0, float(metadata[key])))

        low_confidence_flags = (
            "transition_model_missing",
            "transition_constraints_missing",
            "controller_firmware_unknown",
            "misspecified_constraints",
            "information_missing",
        )
        if any(bool(metadata.get(flag, False)) for flag in low_confidence_flags):
            return 0.0

        source = str(metadata.get("transition_model_source", "sumo")).lower()
        if source in {"real", "offline", "cityflow", "firmware", "unknown"}:
            return 0.0
        return 1.0
