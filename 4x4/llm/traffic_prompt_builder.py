"""
llm/traffic_prompt_builder.py

TrafficPromptBuilder — constructs the LLM prompt from an RLDecisionInfo bundle.

The prompt is kept compact to minimise token usage (~250 tokens per call).
It includes:
    - Intersection context (ID, current phase, legal phases)
    - RL proposal and confidence margin
    - Action Q-scores
    - Anomaly tags
    - Observation summary
    - Neighbour occupancy / attention summary

Output schema required from the LLM::

    {"decision": "accept" | "override", "final_phase": int, "reason": "..."}

Sources
-------
- SafeGAT-LLM scaffold (llm/traffic_prompt_builder.py)
- iLLM-TSC2 (llm_agents/grid_tsc_agent.py) — prompt style guidance
"""

from __future__ import annotations

from .types import RLDecisionInfo

# System-level rules appended to every prompt.
_RULES = (
    "Rules: emergency vehicle → serve its phase; "
    "neighbour occ>0.70 → release pressure away from congested direction; "
    "yellow phases ONLY change for emergencies; "
    "never starve any direction >3 consecutive cycles."
)


class TrafficPromptBuilder:
    """
    Builds a structured, compact LLM prompt for a single intersection decision.

    Usage::

        builder = TrafficPromptBuilder()
        prompt  = builder.build(info)   # info: RLDecisionInfo
    """

    def build(self, info: RLDecisionInfo) -> str:
        """
        Construct and return the prompt string.

        Parameters
        ----------
        info : RLDecisionInfo — full decision context for one intersection

        Returns
        -------
        str — prompt ready to send to the LLM
        """
        # Neighbour summary lines
        neighbor_lines = [
            f"  - {key}: {value}"
            for key, value in info.neighbor_summary.items()
        ] or ["  - none"]

        anomaly_text  = ", ".join(info.anomaly_tags) if info.anomaly_tags else "none"
        action_scores = ", ".join(
            f"{i}:{score:.4f}" for i, score in enumerate(info.action_scores)
        )
        legal_actions = ", ".join(str(x) for x in info.legal_actions)
        obs_summary   = info.metadata.get("observation_summary", str(info.observation))

        return (
            f"You are a traffic-signal control verifier for a multi-intersection "
            f"graph RL controller.\n"
            f"Return valid JSON ONLY — no markdown, no extra text.\n\n"
            f"{_RULES}\n\n"
            f"Intersection ID  : {info.intersection_id}\n"
            f"Current phase    : {info.phase}\n"
            f"RL proposed phase: {info.rl_action}\n"
            f"Legal phases     : [{legal_actions}]\n"
            f"Confidence margin: {info.confidence_margin:.4f}\n"
            f"Action scores    : {action_scores}\n"
            f"Anomaly tags     : {anomaly_text}\n"
            f"Observation      : {obs_summary}\n"
            f"Neighbour summary:\n" + "\n".join(neighbor_lines) + "\n\n"
            f'Output schema:\n'
            f'{{"decision": "accept" or "override", '
            f'"final_phase": integer, '
            f'"reason": "brief technical reason"}}'
        ).strip()
