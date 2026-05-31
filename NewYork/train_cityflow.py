"""Train SafeGAT GAT-DQN on the NewYork CityFlow dataset."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cityflow_safegat.train_live import main


def default_args() -> list[str]:
    dataset_dir = Path(__file__).resolve().parent / "28_7"
    return [
        "--dataset-dir", str(dataset_dir),
        "--roadnet", "roadnet_28_7.json",
        "--flow", "anon_28_7_newyork_real_double.json",
        "--impl-dir", str(ROOT / "7x28"),
        "--output-dir", "safegat_cityflow_live_training",
        "--model-dir", "models",
        "--gat-heads", "2",
    ]


if __name__ == "__main__":
    main(default_args() + sys.argv[1:])
