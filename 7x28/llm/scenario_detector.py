"""
llm/scenario_detector.py

ScenarioDetector — flags corrupted, rare, or safety-critical observations.

Detects
-------
- nan_observation      : any NaN in the obs array
- possible_packet_loss : fraction of zeros exceeds threshold
- queue_spike          : max obs value exceeds queue spike threshold
- empty_observation    : empty obs array
- emergency_vehicle    : metadata flag
- accident_flag        : metadata flag
- sensor_delay_flag    : metadata flag

Source: SafeGAT-LLM scaffold (llm/scenario_detector.py).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class ScenarioDetector:
    """
    Simulator-agnostic observation anomaly detector.

    Parameters
    ----------
    queue_spike_threshold            : float — max(obs) >= this triggers queue_spike
    nan_is_corruption                : bool  — NaN → nan_observation tag
    zero_fraction_corruption_threshold : float — frac of zeros >= this triggers packet_loss
    """

    def __init__(
        self,
        queue_spike_threshold: float = 0.8,
        nan_is_corruption: bool = True,
        zero_fraction_corruption_threshold: float = 0.9,
    ) -> None:
        self.queue_spike_threshold            = queue_spike_threshold
        self.nan_is_corruption                = nan_is_corruption
        self.zero_fraction_corruption_threshold = zero_fraction_corruption_threshold

    def detect(
        self,
        observation: Any,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Analyse the observation and metadata for anomalies.

        Parameters
        ----------
        observation : array-like — the junction's obs vector
        metadata    : dict       — optional flags from the environment info

        Returns
        -------
        dict with keys:
            tags       : List[str]  — detected anomaly tags
            corrupted  : bool       — True for data-integrity issues
            zero_fraction : float   — fraction of zero elements
        """
        metadata = metadata or {}
        obs      = np.asarray(observation, dtype=float)
        tags: List[str] = []

        if self.nan_is_corruption and np.isnan(obs).any():
            tags.append("nan_observation")

        if obs.size > 0:
            zero_frac = float(np.mean(obs == 0.0))
            if zero_frac >= self.zero_fraction_corruption_threshold:
                tags.append("possible_packet_loss")
            if float(np.max(obs)) >= self.queue_spike_threshold:
                tags.append("queue_spike")
        else:
            zero_frac = 1.0
            tags.append("empty_observation")

        if metadata.get("emergency_vehicle", False):
            tags.append("emergency_vehicle")
        if metadata.get("accident_flag", False):
            tags.append("accident_flag")
        if metadata.get("sensor_delay_flag", False):
            tags.append("sensor_delay_flag")

        corrupted = any(
            t in {"nan_observation", "possible_packet_loss", "empty_observation"}
            for t in tags
        )

        return {
            "tags":          tags,
            "corrupted":     corrupted,
            "zero_fraction": zero_frac if obs.size > 0 else 1.0,
        }
