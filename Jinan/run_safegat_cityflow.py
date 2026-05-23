"""Run modified SafeGAT offline audit on Jinan CityFlow datasets."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cityflow_safegat.offline_runner import main


if __name__ == "__main__":
    dataset_dir = Path(__file__).resolve().parent / "3_4"
    default_args = [
        "--dataset-dir", str(dataset_dir),
        "--roadnet", "roadnet_3_4.json",
        "--flow", "anon_3_4_jinan_real.json",
        "--impl-dir", str(ROOT / "4x4"),
        "--output-dir", "safegat_cityflow_output",
        "--max-nodes-per-step", "4",
    ]
    main(default_args + sys.argv[1:])
