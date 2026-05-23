"""Run SafeGAT CityFlow sweeps on NewYork."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cityflow_safegat.offline_sweeps import main


if __name__ == "__main__":
    dataset_dir = Path(__file__).resolve().parent / "28_7"
    sys.argv[1:1] = [
        "--dataset-dir", str(dataset_dir),
        "--roadnet", "roadnet_28_7.json",
        "--flow", "anon_28_7_newyork_real_double.json",
        "--impl-dir", str(ROOT / "7x28"),
    ]
    main()
