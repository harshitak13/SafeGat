"""
utils/margin.py

Helpers for computing Q-value confidence margins.

The margin Δ_i = Q(a*) − Q(a_2nd) measures how decisively the GAT-DQN
chose its greedy action at junction i.  A small Δ_i means the agent is
near-indifferent between its top two choices → good candidate for LLM review.

Sources
-------
- iLLM-TSC2 (run_grid_llm.py)
- SafeGAT-LLM scaffold (integration/agent_colight_hook_example.py)
"""

from __future__ import annotations

from typing import List

import numpy as np


def compute_q_margins(q_values: np.ndarray) -> np.ndarray:
    """
    Compute per-node Q-margin Δ_i = Q(a*) − Q(a_2nd).

    Parameters
    ----------
    q_values : (num_nodes, num_actions) float array

    Returns
    -------
    margins : (num_nodes,) float array — lower values indicate more uncertainty
    """
    sorted_q = np.sort(q_values, axis=1)[:, ::-1]   # descending
    if sorted_q.shape[1] < 2:
        return sorted_q[:, 0]
    return sorted_q[:, 0] - sorted_q[:, 1]


def compute_margin_from_scores(scores) -> float:
    """
    Compute the scalar margin for a single node's action-score vector.

    Parameters
    ----------
    scores : array-like (num_actions,)

    Returns
    -------
    float — margin between top-2 scores, or just max if only one score
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be a 1-D array")
    top2 = np.sort(scores)[-2:]
    return float(top2[-1] - top2[-2]) if len(top2) == 2 else float(top2[-1])


def select_uncertain_nodes(
    margins:       np.ndarray,
    anomaly_flags: np.ndarray,
    tau:           float,
) -> List[int]:
    """
    Return indices of nodes that should be sent to the LLM.

    A node is flagged if its margin is below ``tau`` OR if its anomaly
    flag is set.  The returned list is sorted by ascending margin so that
    the most uncertain nodes are at the front (useful for budget trimming).

    Parameters
    ----------
    margins       : (num_nodes,) float — per-node Δ_i values
    anomaly_flags : (num_nodes,) bool  — True if anomaly detected
    tau           : float              — uncertainty threshold

    Returns
    -------
    List[int] — sorted by ascending margin (most uncertain first)
    """
    uncertain = (margins < tau) | anomaly_flags.astype(bool)
    flagged   = [int(i) for i in range(len(margins)) if uncertain[i]]
    flagged.sort(key=lambda i: margins[i])
    return flagged
