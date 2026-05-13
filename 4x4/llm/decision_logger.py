"""
llm/decision_logger.py

DecisionLogger — appends one JSON record per refine() call to a JSONL file.

Each record contains:
    intersection_id, phase, rl_action, final_action, source,
    trigger_reason, safety_adjusted, confidence_margin, action_scores,
    anomaly_tags, llm_called, llm_decision, debug

Source: SafeGAT-LLM scaffold (llm/decision_logger.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class DecisionLogger:
    """
    Append-only JSONL logger for SafeGAT refine decisions.

    Parameters
    ----------
    filepath : str — path to the output .jsonl file
                     (parent directories are created automatically)
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        """Append ``record`` as a single JSON line."""
        with self.filepath.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
