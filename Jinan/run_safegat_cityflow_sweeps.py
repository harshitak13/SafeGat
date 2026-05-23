"""Run SafeGAT CityFlow sweeps on Jinan."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cityflow_safegat.offline_sweeps import main


if __name__ == "__main__":
    dataset_dir = Path(__file__).resolve().parent / "3_4"
    sys.argv[1:1] = [
        "--dataset-dir", str(dataset_dir),
        "--roadnet", "roadnet_3_4.json",
        "--flow", "anon_3_4_jinan_real.json",
        "--impl-dir", str(ROOT / "4x4"),
    ]
    main()
