"""Run modified SafeGAT offline audit on Hangzhou CityFlow datasets."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cityflow_safegat.offline_runner import main


if __name__ == "__main__":
    dataset_dir = Path(__file__).resolve().parent / "4_4"
    default_args = [
        "--dataset-dir", str(dataset_dir),
        "--roadnet", "roadnet_4_4.json",
        "--flow", "anon_4_4_hangzhou_real.json",
        "--impl-dir", str(ROOT / "4x4"),
        "--output-dir", "safegat_cityflow_output",
        "--max-nodes-per-step", "4",
    ]
    main(default_args + sys.argv[1:])
