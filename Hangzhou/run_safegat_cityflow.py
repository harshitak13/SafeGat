"""Run live SafeGAT-LLM control on Hangzhou CityFlow datasets."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cityflow_safegat.live_runner import main


def default_args() -> list[str]:
    dataset_dir = Path(__file__).resolve().parent / "4_4"
    return [
        "--dataset-dir", str(dataset_dir),
        "--roadnet", "roadnet_4_4.json",
        "--flow", "anon_4_4_hangzhou_real.json",
        "--impl-dir", str(ROOT / "4x4"),
        "--llm-config", str(ROOT / "4x4" / "configs" / "config.yaml"),
        "--output-dir", "safegat_cityflow_live_output",
        "--sim-seconds", "1800",
        "--action-interval", "5",
    ]


if __name__ == "__main__":
    main(default_args() + sys.argv[1:])
