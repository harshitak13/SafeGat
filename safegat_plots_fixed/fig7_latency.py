"""
fig7_latency.py
===============
Figure 7 — Latency & Deployment Feasibility
Produces:
  fig7_4x4       — 3-panel latency analysis (original layout)
  fig7_7x28      — 3-panel latency analysis (scaled for 196 nodes)
  fig7_combined  — 2-row layout, 4×4 on top, 7×28 on bottom
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from _shared import *

# ── Latency model constants ───────────────────────────────────────────────────
GROQ_MEAN_MS  = 250
GROQ_P95_MS   = 600
GROQ_P99_MS   = 1200
STEP_BUDGET_MS = 1000

RATES   = [0.10, 0.25, 0.50, 1.00, 1.50, 2.00, 2.50]
BUDGETS = [200, 500, 1000, 2000, 5000]


def _latency_panels(ax_dist, ax_sync, ax_heat,
                    calls_per_step: float, total_calls: int, label: str):
    """Draw the 3 standard latency panels into given axes."""

    # Panel 1 — latency distribution
    mu    = math.log(GROQ_MEAN_MS)
    sigma = 0.5
    xs    = np.linspace(50, 2000, 500)
    pdf   = (1 / (xs * sigma * math.sqrt(2 * math.pi))) * \
            np.exp(-(np.log(xs) - mu)**2 / (2 * sigma**2))
    ax_dist.plot(xs, pdf, color="#2196F3", lw=2.2)
    ax_dist.axvline(GROQ_MEAN_MS,   color="blue",   ls="--", lw=1.5,
                    label=f"Mean={GROQ_MEAN_MS}ms")
    ax_dist.axvline(GROQ_P95_MS,    color="orange", ls="--", lw=1.5,
                    label=f"p95={GROQ_P95_MS}ms")
    ax_dist.axvline(GROQ_P99_MS,    color="red",    ls="--", lw=1.5,
                    label=f"p99={GROQ_P99_MS}ms")
    ax_dist.axvline(STEP_BUDGET_MS, color="green",  ls="-",  lw=2.0,
                    label=f"Step budget={STEP_BUDGET_MS}ms")
    ax_dist.fill_betweenx([0, max(pdf)], 0, STEP_BUDGET_MS, alpha=0.07, color="green")
    ax_dist.set_xlabel("Latency (ms)", fontweight="bold")
    ax_dist.set_ylabel("Probability density", fontweight="bold")
    ax_dist.set_title(f"LLM Call Latency\n(Groq llama-3.1-8b-instant) {label}",
                      fontweight="bold")
    ax_dist.legend(fontsize=9); ax_dist.grid(alpha=0.3)

    # Panel 2 — sync vs async overhead
    all_rates = sorted(set(RATES + [calls_per_step]))
    sync_oh   = [r * GROQ_MEAN_MS for r in all_rates]
    async_oh  = [GROQ_MEAN_MS     for _ in all_rates]
    ax_sync.plot(all_rates, sync_oh,  "o-",  color="#F44336", lw=2.2, label="Synchronous")
    ax_sync.plot(all_rates, async_oh, "s--", color="#4CAF50", lw=2.2, label="Async (non-blocking)")
    ax_sync.axhline(STEP_BUDGET_MS, color="black", ls=":", lw=1.8, label="1s step budget")
    ax_sync.axvline(calls_per_step, color="purple", ls="--", lw=1.5,
                    label=f"Actual rate={calls_per_step:.2f} calls/step")
    ax_sync.fill_between(all_rates, 0, STEP_BUDGET_MS, alpha=0.06, color="green",
                         label="Viable zone")
    ax_sync.set_xlabel("LLM calls per simulation step", fontweight="bold")
    ax_sync.set_ylabel("Latency overhead (ms)", fontweight="bold")
    ax_sync.set_title(f"Sync vs Async Overhead vs Step Budget {label}", fontweight="bold")
    ax_sync.legend(fontsize=9); ax_sync.grid(alpha=0.3)

    # Panel 3 — deployment viability heatmap
    headroom = np.array([
        [b - (r * GROQ_MEAN_MS) for b in BUDGETS]
        for r in RATES
    ])
    im = ax_heat.imshow(headroom, aspect="auto", origin="lower",
                        cmap="RdYlGn", vmin=-1000, vmax=1000)
    plt.colorbar(im, ax=ax_heat, label="Headroom (ms)")
    ax_heat.set_xticks(range(len(BUDGETS)))
    ax_heat.set_xticklabels([f"{b}ms" for b in BUDGETS], fontsize=9, fontweight="bold")
    ax_heat.set_yticks(range(len(RATES)))
    ax_heat.set_yticklabels([f"{r:.2f}" for r in RATES], fontsize=9, fontweight="bold")
    ax_heat.set_xlabel("Step latency budget", fontweight="bold")
    ax_heat.set_ylabel("LLM calls / step", fontweight="bold")
    ax_heat.set_title(f"Deployment Viability\n(headroom ms, green=OK) {label}",
                      fontweight="bold")
    for i, r in enumerate(RATES):
        for j, b in enumerate(BUDGETS):
            h = headroom[i, j]
            ax_heat.text(j, i, f"{h:.0f}", ha="center", va="center",
                         fontsize=8, fontweight="bold",
                         color="black" if abs(h) < 800 else "white")


def plot_fig7():
    # ── 4×4 stats ─────────────────────────────────────────────────────────────
    sl4  = load_4x4_steplog()
    sum4 = load_4x4_summary()
    cps4 = sum4["llm_calls"] / sum4["total_sim_steps"]

    # ── 7×28 stats ────────────────────────────────────────────────────────────
    sum7 = load_7x28_summary()
    cps7 = sum7["llm_calls"] / sum7["total_sim_steps"]

    # ── Fig 7A: 4×4 only ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Latency Analysis & Deployment Feasibility — 4×4 Grid",
                 fontsize=16, fontweight="bold", y=1.02)
    _latency_panels(axes[0], axes[1], axes[2], cps4, sum4["llm_calls"], "(4×4)")
    fig.tight_layout()
    save(fig, "fig7_4x4")

    # ── Fig 7B: 7×28 only ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Latency Analysis & Deployment Feasibility — 7×28 Grid",
                 fontsize=16, fontweight="bold", y=1.02)
    _latency_panels(axes[0], axes[1], axes[2], cps7, sum7["llm_calls"], "(7×28)")
    fig.tight_layout()
    save(fig, "fig7_7x28")

    # ── Fig 7C: Combined — 2 rows ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Latency Analysis & Deployment Feasibility — 4×4 vs 7×28",
                 fontsize=16, fontweight="bold", y=1.01)
    _latency_panels(axes[0, 0], axes[0, 1], axes[0, 2], cps4, sum4["llm_calls"], "(4×4)")
    _latency_panels(axes[1, 0], axes[1, 1], axes[1, 2], cps7, sum7["llm_calls"], "(7×28)")
    fig.tight_layout()
    save(fig, "fig7_combined")
    print("  ok fig7 done")


if __name__ == "__main__":
    plot_fig7()
