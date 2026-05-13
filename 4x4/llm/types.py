"""
llm/types.py

Shared dataclasses for the SafeGAT-LLM pipeline.

RLDecisionInfo  — input bundle passed to SafeGATRefiner.refine()
LLMDecision     — parsed LLM response
RefineResult    — output from SafeGATRefiner.refine()

Source: SafeGAT-LLM scaffold (llm/types.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RLDecisionInfo:
    """
    All context needed by the refiner for one intersection at one step.

    Attributes
    ----------
    intersection_id   : str         — e.g. "J5"
    observation       : Any         — raw obs array (num_obs_dim,)
    phase             : int         — current active phase index
    rl_action         : int         — action proposed by GAT-DQN
    action_scores     : List[float] — Q-values for all legal actions
    confidence_margin : float       — Δ = Q(a*) − Q(a_2nd); low → uncertain
    legal_actions     : List[int]   — phase indices allowed at this step
    neighbor_summary  : Dict        — occupancy / action info for neighbouring junctions
    anomaly_tags      : List[str]   — pre-populated tags (extended by ScenarioDetector)
    metadata          : Dict        — e.g. phase_runtime, emergency_vehicle flag
    """
    intersection_id:   str
    observation:       Any
    phase:             int
    rl_action:         int
    action_scores:     List[float]
    confidence_margin: float
    legal_actions:     List[int]
    neighbor_summary:  Dict[str, Any] = field(default_factory=dict)
    anomaly_tags:      List[str]      = field(default_factory=list)
    metadata:          Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMDecision:
    """
    Parsed response from the LLM backend.

    Attributes
    ----------
    decision    : str   — "accept" or "override"
    final_phase : int   — the phase the LLM recommends
    reason      : str   — brief textual explanation
    raw_text    : str   — full raw LLM output (for logging)
    parsed      : Dict  — the parsed JSON dict
    """
    decision:    str
    final_phase: int
    reason:      str
    raw_text:    str          = ""
    parsed:      Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefineResult:
    """
    Output from SafeGATRefiner.refine().

    Attributes
    ----------
    final_action    : int   — action to execute (post-shield)
    source          : str   — "rl" | "llm_accept" | "llm_override"
    trigger_reason  : str   — comma-joined gate reasons, or "none"
    safety_adjusted : bool  — True if SafetyShield modified the action
    llm_called      : bool  — True if LLM was invoked
    llm_decision    : Optional[LLMDecision]
    debug           : Dict  — gate score breakdown, shield reason, scenario tags
    """
    final_action:    int
    source:          str
    trigger_reason:  str
    safety_adjusted: bool
    llm_called:      bool
    llm_decision:    Optional[LLMDecision]  = None
    debug:           Dict[str, Any]         = field(default_factory=dict)
