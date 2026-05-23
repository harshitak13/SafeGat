"""
Lightweight GRU-based anomaly forecaster for proactive SafeGAT gating.

The forecaster keeps a short per-intersection observation history and predicts
the probability that a current reactive anomaly detector will fire 2-3 steps
ahead. It is intentionally small enough to run online for every node.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, Iterable

import numpy as np
import torch
from torch import nn


class GRUAnomalyForecaster(nn.Module):
    """
    Online GRU forecaster with a calibrated temporal-risk fallback.

    No extra LLM calls are made. The probability produced here is consumed by
    InterventionGate as an additional risk term.
    """

    def __init__(
        self,
        obs_dim: int = 8,
        hidden_dim: int = 12,
        history_len: int = 4,
        horizon: int = 3,
        threshold: float = 0.65,
        queue_spike_threshold: float = 0.85,
        zero_fraction_corruption_threshold: float = 0.90,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.history_len = history_len
        self.horizon = horizon
        self.threshold = threshold
        self.queue_spike_threshold = queue_spike_threshold
        self.zero_fraction_corruption_threshold = zero_fraction_corruption_threshold
        self.feature_dim = 5
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)
        self._history: Dict[str, Deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=history_len)
        )
        self._init_stable_weights()
        self.eval()

    def update_and_predict(self, intersection_id: str, observation: Any) -> float:
        """Append one observation and return p(anomaly in the next 2-3 steps)."""
        obs = self._clean_observation(observation)
        self._history[intersection_id].append(obs)
        return self.predict(intersection_id)

    def update_many(self, intersection_ids: Iterable[str], observations: Any) -> np.ndarray:
        """Vector-friendly helper for inference loops."""
        return np.asarray(
            [
                self.update_and_predict(intersection_id, observation)
                for intersection_id, observation in zip(intersection_ids, observations)
            ],
            dtype=np.float32,
        )

    def predict(self, intersection_id: str) -> float:
        history = list(self._history.get(intersection_id, ()))
        if not history:
            return 0.0
        sequence = self._feature_sequence(history)
        with torch.no_grad():
            _, hidden = self.gru(sequence.unsqueeze(0))
            gru_prob = torch.sigmoid(self.head(hidden[-1])).item()
        temporal_prob = self._temporal_projection(sequence.numpy())
        return float(np.clip(0.35 * gru_prob + 0.65 * temporal_prob, 0.0, 1.0))

    def is_forecast_anomaly(self, probability: float) -> bool:
        return probability >= self.threshold

    def _feature_sequence(self, history: list[np.ndarray]) -> torch.Tensor:
        padded = history[-self.history_len :]
        while len(padded) < self.history_len:
            padded.insert(0, padded[0])
        features = np.asarray([self._features(obs) for obs in padded], dtype=np.float32)
        return torch.from_numpy(features)

    def _features(self, obs: np.ndarray) -> np.ndarray:
        max_obs = float(np.max(obs)) if obs.size else 0.0
        mean_occ = float(np.mean(obs[2:6])) if obs.size >= 6 else max_obs
        queue = float(obs[6]) if obs.size > 6 else max_obs
        emergency = float(obs[7] > 0.5) if obs.size > 7 else 0.0
        zero_frac = float(np.mean(obs == 0.0)) if obs.size else 1.0
        return np.asarray([max_obs, mean_occ, queue, emergency, zero_frac], dtype=np.float32)

    def _temporal_projection(self, features: np.ndarray) -> float:
        max_obs = features[:, 0]
        mean_occ = features[:, 1]
        queue = features[:, 2]
        emergency = features[:, 3]
        zero_frac = features[:, 4]

        span = max(1, len(features) - 1)
        max_slope = max(0.0, float(max_obs[-1] - max_obs[0]) / span)
        queue_slope = max(0.0, float(queue[-1] - queue[0]) / span)

        current_pressure = max(
            float(max_obs[-1]) / max(self.queue_spike_threshold, 1e-6),
            float(queue[-1]) / max(self.queue_spike_threshold, 1e-6),
            float(mean_occ[-1]),
            float(emergency[-1]),
            float(zero_frac[-1]) / max(self.zero_fraction_corruption_threshold, 1e-6),
        )
        projected_pressure = current_pressure + self.horizon * max(max_slope, queue_slope)
        return float(np.clip(projected_pressure, 0.0, 1.0))

    def _clean_observation(self, observation: Any) -> np.ndarray:
        obs = np.asarray(observation, dtype=float).reshape(-1)
        if obs.size == 0:
            return np.zeros(self.obs_dim, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=1.0, posinf=1.0, neginf=0.0)
        if obs.size < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - obs.size))
        return obs[: self.obs_dim].astype(np.float32)

    def _init_stable_weights(self) -> None:
        for _, param in self.gru.named_parameters():
            nn.init.zeros_(param)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -3.0)
