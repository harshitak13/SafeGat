"""
per_intersection_analysis.py — Per-Junction Analysis for 7x28 SafeGAT-iLLM
============================================================================
Reads the output logs from run_safegat.py / ablation runs and produces
per-junction breakdowns across all 196 controlled nodes of the 7x28 grid.

Analyses
--------
1. Per-junction mean reward, total reward, LLM calls, override rate
2. Spatial heatmap of junction performance (grid layout)
3. Top/bottom 20 junctions by reward
4. LLM intervention distribution across the network

Input files (produced by run_safegat.py)
-----------------------------------------
    output/per_junction_results.csv
    output/step_log.json
    output/intervention_summary.json

Output
------
    output/per_junction_analysis/
        per_junction_summary.csv
        top20_junctions.csv
        bottom20_junctions.csv
        reward_heatmap.png
        llm_calls_heatmap.png

Run
---
    python latency_per_intersection_robustness/per_intersection_analysis.py
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
import sys
sys.path.insert(0, str(_ROOT))

from network.net_config import CONTROLLED_TLS, TLS_GRID_POS, NUM_NODES, GRID_ROWS

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_DIR  = _ROOT / "output"
OUT_DIR    = _ROOT / "output" / "per_junction_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_per_junction_csv(path: Path) -> dict:
    """Load per_junction_results.csv -> dict keyed by junction_id."""
    data = {}
    if not path.exists():
        print(f"  [WARN] Not found: {path}  — using synthetic data for demo.")
        rng = np.random.default_rng(42)
        for tls_id in CONTROLLED_TLS:
            data[tls_id] = {
                "junction_id":   tls_id,
                "llm_calls":     int(rng.integers(0, 50)),
                "llm_overrides": int(rng.integers(0, 15)),
                "mean_reward":   float(rng.uniform(-0.8, -0.1)),
                "total_reward":  float(rng.uniform(-2500, -300)),
            }
        return data
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jid = row["junction_id"]
            data[jid] = {
                "junction_id":   jid,
                "llm_calls":     int(row.get("llm_calls", 0)),
                "llm_overrides": int(row.get("llm_overrides", 0)),
                "mean_reward":   float(row.get("mean_reward", 0.0)),
                "total_reward":  float(row.get("total_reward", 0.0)),
            }
    return data


def build_grid_matrix(data: dict, field: str) -> tuple:
    """
    Build a 2D numpy matrix for plotting.
    Returns (matrix, row_labels, col_labels) where matrix[r][c] = value.
    The 7x28 grid is irregular, so missing cells are NaN.
    """
    # Find max row and col
    max_row = max(pos[0] for pos in TLS_GRID_POS.values()) + 1
    max_col = max(pos[1] for pos in TLS_GRID_POS.values()) + 1

    matrix = np.full((max_row, max_col), np.nan)
    for jid, jdata in data.items():
        if jid in TLS_GRID_POS:
            r, c = TLS_GRID_POS[jid]
            matrix[r, c] = float(jdata.get(field, 0.0))

    return matrix, list(range(max_row)), list(range(max_col))


def plot_heatmap(matrix: np.ndarray, title: str, out_path: str,
                 cmap: str = "RdYlGn") -> None:
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(20, 6))
        # Mask NaN cells
        import numpy.ma as ma
        masked = ma.masked_invalid(matrix)
        im = ax.imshow(masked, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Column (West -> East)")
        ax.set_ylabel("Row (North -> South)")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")
    except ImportError:
        print("  matplotlib not available — skipping heatmap")


def main() -> None:
    print(f"Per-Junction Analysis — 7x28 ({NUM_NODES} junctions)")
    print(f"Reading from: {INPUT_DIR}")

    # ── Load data ─────────────────────────────────────────────────────────────
    per_jct_path = INPUT_DIR / "per_junction_results.csv"
    data = load_per_junction_csv(per_jct_path)

    # Ensure all CONTROLLED_TLS junctions are represented
    for tls_id in CONTROLLED_TLS:
        if tls_id not in data:
            data[tls_id] = {
                "junction_id": tls_id, "llm_calls": 0, "llm_overrides": 0,
                "mean_reward": 0.0, "total_reward": 0.0,
            }

    rows = [data[jid] for jid in CONTROLLED_TLS]

    # ── Add derived fields ─────────────────────────────────────────────────────
    for row in rows:
        calls = max(row["llm_calls"], 1)
        row["override_rate_%"] = round(100 * row["llm_overrides"] / calls, 2)
        r, c = TLS_GRID_POS.get(row["junction_id"], (0, 0))
        row["grid_row"] = r
        row["grid_col"] = c

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_path = OUT_DIR / "per_junction_summary.csv"
    fieldnames = ["junction_id", "grid_row", "grid_col", "mean_reward",
                  "total_reward", "llm_calls", "llm_overrides", "override_rate_%"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary written -> {summary_path}")

    # ── Top/bottom 20 by mean reward ──────────────────────────────────────────
    sorted_rows = sorted(rows, key=lambda r: r["mean_reward"], reverse=True)
    top20 = sorted_rows[:20]
    bot20 = sorted_rows[-20:]

    def write_subset(subset, path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(subset)
        print(f"Written -> {path}")

    write_subset(top20, OUT_DIR / "top20_junctions.csv")
    write_subset(bot20, OUT_DIR / "bottom20_junctions.csv")

    # ── Grid statistics ────────────────────────────────────────────────────────
    mean_rewards = [r["mean_reward"] for r in rows]
    llm_calls    = [r["llm_calls"] for r in rows]
    print(f"\nNetwork statistics across {NUM_NODES} junctions:")
    print(f"  Mean reward   : {np.mean(mean_rewards):.4f} ± {np.std(mean_rewards):.4f}")
    print(f"  Best junction : {top20[0]['junction_id']} (mean={top20[0]['mean_reward']:.4f})")
    print(f"  Worst junction: {bot20[-1]['junction_id']} (mean={bot20[-1]['mean_reward']:.4f})")
    print(f"  Total LLM calls across network: {sum(llm_calls)}")
    print(f"  Mean LLM calls per junction   : {np.mean(llm_calls):.1f}")

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    print("\nGenerating heatmaps ...")
    reward_matrix, _, _ = build_grid_matrix(data, "mean_reward")
    plot_heatmap(
        reward_matrix,
        "Per-Junction Mean Reward — 7x28 SafeGAT (greener = better)",
        str(OUT_DIR / "reward_heatmap.png"),
        cmap="RdYlGn",
    )

    calls_matrix, _, _ = build_grid_matrix(data, "llm_calls")
    plot_heatmap(
        calls_matrix,
        "Per-Junction LLM Calls — 7x28 SafeGAT (brighter = more interventions)",
        str(OUT_DIR / "llm_calls_heatmap.png"),
        cmap="YlOrRd",
    )

    print(f"\nPer-junction analysis complete. All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
