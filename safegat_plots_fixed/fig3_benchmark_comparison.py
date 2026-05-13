"""
fig3_benchmark_comparison.py
============================
Figure 3 — Traffic Efficiency & Benchmark Comparison
Produces:
  fig3_combined  — grouped bars 4×4 vs 7×28 for each method/metric
  fig3_4x4       — 4×4 only bar chart
  fig3_7x28      — 7×28 only bar chart
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from _shared import *

METHODS_SHORT = [
    "Webster\n(Fixed)",
    "Actuated\n/Adaptive",
    "Plain\nDQN",
    "GAT-DQN\n(no LLM)",
    "SafeGAT\n-iLLM",
]

METRICS = [
    ("att",        "ATT (s)",              True),
    ("avg_queue",  "Queue Length (s)",     True),
    ("avg_delay",  "Delay / Time-Loss (s)",True),
    ("throughput", "Throughput (veh)",     False),
]

BAR_COLORS = ["#e41a1c", "#ff7f00", "#377eb8", "#4daf4a", "#984ea3"]


def _single_grid_chart(bench: dict, title: str, fig_size=(15, 5)):
    fig, axes = plt.subplots(1, 4, figsize=fig_size)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

    methods  = list(bench.keys())
    x        = np.arange(len(methods))
    width    = 0.55

    for ax, (metric, ylabel, lower_better) in zip(axes, METRICS):
        vals = [bench[m][metric] for m in methods]
        bars = ax.bar(x, vals, width=width, color=BAR_COLORS, edgecolor="white", linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{v:.0f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS_SHORT, fontsize=9, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(ylabel, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        # Highlight best bar
        best_idx = np.argmin(vals) if lower_better else np.argmax(vals)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.5)

    fig.tight_layout()
    return fig


def _combined_chart(bench4: dict, bench7: dict, fig_size=(16, 12)):
    """4-metric plot, each metric has grouped 4x4 vs 7x28 bars."""
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    fig.suptitle(
        "Traffic Signal Control — Method Comparison (4×4 vs 7×28 Grid)",
        fontsize=16, fontweight="bold", y=1.01
    )

    methods = list(bench4.keys())
    x       = np.arange(len(methods))
    w       = 0.35

    for ax, (metric, ylabel, lower_better) in zip(axes.flat, METRICS):
        vals4 = [bench4[m][metric] for m in methods]
        vals7 = [bench7[m][metric] for m in methods]

        bars4 = ax.bar(x - w/2, vals4, w, label="4×4 Grid",  color=C["4x4"],
                       alpha=0.85, edgecolor="white")
        bars7 = ax.bar(x + w/2, vals7, w, label="7×28 Grid", color=C["7x28"],
                       alpha=0.85, edgecolor="white")

        for bars, vals in [(bars4, vals4), (bars7, vals7)]:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(max(vals4), max(vals7)) * 0.008,
                        f"{v:.0f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(METHODS_SHORT, fontsize=9, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(ylabel, fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_fig3():
    bench4 = load_4x4_benchmark()
    comb7  = load_7x28_combined()
    bench7 = make_7x28_benchmark(comb7)

    fig = _single_grid_chart(
        bench4,
        "Traffic Signal Control — Method Comparison (4×4 Grid, 1 800 veh/h/lane)"
    )
    save(fig, "fig3_4x4")

    fig = _single_grid_chart(
        bench7,
        "Traffic Signal Control — Method Comparison (7×28 Grid, 1 800 veh/h/lane)"
    )
    save(fig, "fig3_7x28")

    fig = _combined_chart(bench4, bench7)
    save(fig, "fig3_combined")
    print("  ok fig3 done")


if __name__ == "__main__":
    plot_fig3()
