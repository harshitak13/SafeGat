"""Run live SafeGAT-LLM control on NewYork CityFlow datasets."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cityflow_safegat.live_runner import main


def default_args() -> list[str]:
    dataset_dir = Path(__file__).resolve().parent / "28_7"
    return [
        "--dataset-dir", str(dataset_dir),
        "--roadnet", "roadnet_28_7.json",
        "--flow", "anon_28_7_newyork_real_double.json",
        "--impl-dir", str(ROOT / "7x28"),
        "--llm-config", str(ROOT / "7x28" / "configs" / "config.yaml"),
        "--output-dir", "safegat_cityflow_live_output",
        "--sim-seconds", "1800",
        "--action-interval", "5",
        "--gat-heads", "2",
    ]


if __name__ == "__main__":
    main(default_args() + sys.argv[1:])
