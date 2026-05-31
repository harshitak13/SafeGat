"""Train SafeGAT GAT-DQN on the Hangzhou CityFlow dataset."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cityflow_safegat.train_live import main


def default_args() -> list[str]:
    dataset_dir = Path(__file__).resolve().parent / "4_4"
    return [
        "--dataset-dir", str(dataset_dir),
        "--roadnet", "roadnet_4_4.json",
        "--flow", "anon_4_4_hangzhou_real.json",
        "--impl-dir", str(ROOT / "4x4"),
        "--output-dir", "safegat_cityflow_live_training",
        "--model-dir", "models",
    ]


if __name__ == "__main__":
    main(default_args() + sys.argv[1:])
