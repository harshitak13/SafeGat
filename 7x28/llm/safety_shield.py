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
from typing import Dict, Iterable, Optional, Set

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

    def __init__(self, min_green_hold: int = 3) -> None:
        self.min_green_hold = min_green_hold

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

        # Rule 0: Yellow-phase lock — never switch away during yellow
        if current_phase is not None and current_phase in _YELLOW_PHASES:
            # If proposed action differs from current yellow phase, force hold
            if proposed_action != current_phase:
                return ShieldResult(
                    action   = int(current_phase),
                    adjusted = True,
                    reason   = "yellow_phase_lock",
                )

        # Rule 1: Illegal action repair
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

        # Rule 2: Minimum green hold
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
