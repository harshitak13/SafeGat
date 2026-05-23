"""
Cached corridor context for SafeGAT LLM calls.

The cache stores compact summaries of recent neighboring LLM decisions so a
new prompt can reason with corridor-level consistency without making another
LLM call.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class _DecisionRecord:
    intersection_id: str
    step: int
    rl_action: int
    final_action: int
    source: str
    reason: str
    anomaly_tags: Tuple[str, ...]


class CorridorContextCache:
    """
    Thread-safe cache of recent LLM decisions keyed by intersection.

    The generated token is intentionally short: it is appended to future prompts
    as extra context, but it never triggers an additional LLM request.
    """

    def __init__(
        self,
        max_records_per_intersection: int = 3,
        max_neighbors: int = 4,
        max_token_chars: int = 420,
    ) -> None:
        self.max_records_per_intersection = max_records_per_intersection
        self.max_neighbors = max_neighbors
        self.max_token_chars = max_token_chars
        self._records: Dict[str, Deque[_DecisionRecord]] = defaultdict(
            lambda: deque(maxlen=max_records_per_intersection)
        )
        self._clock = 0
        self._lock = Lock()

    def build_token(
        self,
        intersection_id: str,
        neighbor_ids: Iterable[str],
    ) -> str:
        """Return a compact token for the current intersection corridor."""
        with self._lock:
            keys = self._ordered_keys(intersection_id, neighbor_ids)
            records: List[_DecisionRecord] = []
            for key in keys:
                records.extend(
                    list(self._records[key])[-self.max_records_per_intersection:]
                )

        if not records:
            return "none"

        parts: List[str] = []
        for record in sorted(records, key=lambda item: item.step, reverse=True):
            src = self._source_label(record.source)
            if record.rl_action == record.final_action:
                action = f"keep:{record.final_action}"
            else:
                action = f"{record.rl_action}->{record.final_action}"
            reason = self._shorten(record.reason, 48)
            tags = ",".join(record.anomaly_tags[:2]) or "no-tags"
            parts.append(
                f"{record.intersection_id}@{record.step}:{src}:{action};{tags};{reason}"
            )

        token = " | ".join(parts)
        return self._shorten(token, self.max_token_chars)

    def update(
        self,
        intersection_id: str,
        rl_action: int,
        final_action: int,
        source: str,
        reason: str = "",
        anomaly_tags: Optional[Iterable[str]] = None,
        step: Optional[int] = None,
    ) -> None:
        """Store one completed LLM refinement decision."""
        with self._lock:
            if step is None:
                self._clock += 1
                step = self._clock
            else:
                self._clock = max(self._clock, int(step))
            self._records[intersection_id].append(
                _DecisionRecord(
                    intersection_id=intersection_id,
                    step=int(step),
                    rl_action=int(rl_action),
                    final_action=int(final_action),
                    source=source,
                    reason=str(reason or ""),
                    anomaly_tags=tuple(anomaly_tags or ()),
                )
            )

    def _ordered_keys(
        self,
        intersection_id: str,
        neighbor_ids: Iterable[str],
    ) -> List[str]:
        keys = [intersection_id]
        for neighbor_id in neighbor_ids:
            if neighbor_id not in keys:
                keys.append(str(neighbor_id))
            if len(keys) > self.max_neighbors:
                break
        return keys

    @staticmethod
    def _source_label(source: str) -> str:
        if source == "llm_override":
            return "override"
        if source == "llm_accept":
            return "accept"
        return source

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."
