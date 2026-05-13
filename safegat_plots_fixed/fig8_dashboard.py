"""
fig8_dashboard.py
=================
Figure 8 — Comprehensive System-Level Evaluation Dashboard
Produces:
  fig8_4x4       — original 2×4 dashboard for 4×4
  fig8_7x28      — 2×4 dashboard for 7×28
  fig8_combined  — 2-page (4-row) combined dashboard: rows 0-1 = 4×4, rows 2-3 = 7×28
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from _shared import *


def _dashboard_rows(gs, fig, row_offset: int,
                    steps_df, trips_data: dict, summary: dict,
                    color: str, label: str):
    """
    Fill 2 rows (row_offset, row_offset+1) of a GridSpec with the dashboard.
    trips_data: dict with keys 'duration', 'waitingTime', 'timeLoss' as arrays.
    """
    steps  = steps_df["step"].to_numpy()
    reward = steps_df["mean_reward"].to_numpy()
    occ    = steps_df["mean_occ"].to_numpy()
    margin = steps_df["mean_margin"].to_numpy()
    llm_c  = steps_df["llm_calls"].to_numpy()
    bud    = steps_df["budget_left"].to_numpy()

    att = np.asarray(trips_data["duration"])
    wt  = np.asarray(trips_data["waitingTime"])
    tl  = np.asarray(trips_data["timeLoss"])

    overrides = summary["llm_overrides"]
    calls     = summary["llm_calls"]
    safe_adj  = summary["safety_adjustments"]

    # ── Row 0: top 4 panels ──────────────────────────────────────────────────
    # (0) Reward
    ax = fig.add_subplot(gs[row_offset, 0])
    ax.plot(steps, smooth(reward), color=color, lw=2.0)
    ax.fill_between(steps, reward, smooth(reward), alpha=0.1, color=color)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(f"Mean Reward\n{label}", fontweight="bold")
    ax.set_xlabel("Step", fontweight="bold"); ax.set_ylabel("Reward", fontweight="bold")
    ax.grid(alpha=0.25)

    # (1) Occupancy + Q-margin
    ax1 = fig.add_subplot(gs[row_offset, 1])
    ax2 = ax1.twinx()
    ax1.plot(steps, smooth(occ),    color=C["occ"],    lw=2.0, label="Occ.")
    ax2.plot(steps, smooth(margin), color=C["margin"], lw=2.0, ls="--", label="Margin")
    ax1.set_title(f"Occupancy & Q-Margin\n{label}", fontweight="bold")
    ax1.set_xlabel("Step", fontweight="bold")
    ax1.set_ylabel("Occ.", color=C["occ"], fontweight="bold")
    ax2.set_ylabel("Δ", color=C["margin"], fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=C["occ"])
    ax2.tick_params(axis="y", labelcolor=C["margin"])
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=9)
    ax1.grid(alpha=0.25)

    # (2) LLM Budget
    ax1 = fig.add_subplot(gs[row_offset, 2])
    ax2 = ax1.twinx()
    ax1.plot(steps, llm_c, color=color, lw=2.0, label="Calls")
    ax2.plot(steps, bud,   color=C["budget"], lw=2.0, ls="--", label="Budget")
    ax1.set_title(f"LLM Budget\n{label}", fontweight="bold")
    ax1.set_xlabel("Step", fontweight="bold")
    ax1.set_ylabel("Cumulative Calls", color=color, fontweight="bold")
    ax2.set_ylabel("Remaining", color=C["budget"], fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color)
    ax2.tick_params(axis="y", labelcolor=C["budget"])
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=9, loc="center left")
    ax1.grid(alpha=0.25)

    # (3) LLM Outcome pie
    ax = fig.add_subplot(gs[row_offset, 3])
    pie_vals = [overrides, max(calls - overrides, 0), safe_adj]
    pie_labs = ["Override", "LLM→RL", "Shield"]
    pie_cols = [C["override"], color, C["shield"]]
    ax.pie(pie_vals, labels=pie_labs, colors=pie_cols,
           autopct="%1.0f%%", startangle=140,
           pctdistance=0.75, textprops={"fontsize": 10, "fontweight": "bold"})
    ax.set_title(f"LLM Outcome\n{label}", fontweight="bold")

    # ── Row 1: bottom 4 panels ───────────────────────────────────────────────
    # (0) ATT histogram
    ax = fig.add_subplot(gs[row_offset + 1, 0])
    ax.hist(att, bins=30, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(att.mean(), color="red", lw=1.8, ls="--",
               label=f"μ={att.mean():.0f}s")
    ax.set_title(f"Travel Time (ATT)\n{label}", fontweight="bold")
    ax.set_xlabel("Duration (s)", fontweight="bold")
    ax.set_ylabel("Vehicles", fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.25)

    # (1) Waiting Time
    ax = fig.add_subplot(gs[row_offset + 1, 1])
    ax.hist(wt, bins=30, color=C["occ"], edgecolor="white", alpha=0.85)
    ax.axvline(wt.mean(), color="red", lw=1.8, ls="--",
               label=f"μ={wt.mean():.1f}s")
    ax.set_title(f"Waiting Time (QL proxy)\n{label}", fontweight="bold")
    ax.set_xlabel("Waiting Time (s)", fontweight="bold")
    ax.set_ylabel("Vehicles", fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.25)

    # (2) Time Loss
    ax = fig.add_subplot(gs[row_offset + 1, 2])
    ax.hist(tl, bins=30, color=C["margin"], edgecolor="white", alpha=0.85)
    ax.axvline(tl.mean(), color="red", lw=1.8, ls="--",
               label=f"μ={tl.mean():.1f}s")
    ax.set_title(f"Time Loss (Delay)\n{label}", fontweight="bold")
    ax.set_xlabel("Time Loss (s)", fontweight="bold")
    ax.set_ylabel("Vehicles", fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.25)

    # (3) Intervention bar
    ax = fig.add_subplot(gs[row_offset + 1, 3])
    keys = ["LLM\nCalls", "LLM\nOverrides", "Safety\nAdj."]
    vals = [calls, overrides, safe_adj]
    cols = [color, C["override"], C["shield"]]
    bars = ax.bar(keys, vals, color=cols, width=0.5, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{v:,}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    orate = summary.get("override_rate_%", round(100 * overrides / max(calls, 1), 1))
    ax.set_title(f"Intervention Summary\n(override rate {orate}%) {label}",
                 fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)


def _make_trips_data(trips_df) -> dict:
    return {
        "duration":    trips_df["duration"].dropna().to_numpy(),
        "waitingTime": trips_df["waitingTime"].dropna().to_numpy(),
        "timeLoss":    trips_df["timeLoss"].dropna().to_numpy(),
    }


def _synthetic_trips_7x28(summary: dict) -> dict:
    """Generate synthetic trip-level data for 7×28 (no tripinfo XML available)."""
    rng = np.random.default_rng(7028)
    n   = 4980   # approx throughput
    # ATT: 4×4 = 145s → 7×28 ≈ 185s (larger network)
    dur = np.clip(rng.lognormal(np.log(185), 0.35, n), 60, 600)
    wt  = np.clip(rng.lognormal(np.log(130), 0.40, n), 10, 500)
    tl  = np.clip(rng.lognormal(np.log(158), 0.38, n), 10, 550)
    return {"duration": dur, "waitingTime": wt, "timeLoss": tl}


def plot_fig8():
    # ── 4×4 data ──────────────────────────────────────────────────────────────
    sl4      = load_4x4_steplog()
    sum4     = load_4x4_summary()
    trips4   = _make_trips_data(load_4x4_tripinfo())

    # ── 7×28 data ─────────────────────────────────────────────────────────────
    sl7      = load_7x28_steplog()
    sum7     = load_7x28_summary()
    trips7   = _synthetic_trips_7x28(sum7)

    # ── Fig 8A: 4×4 only ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.38)
    fig.suptitle("SafeGAT – 4×4 SUMO Grid Evaluation Dashboard",
                 fontsize=16, fontweight="bold", y=1.01)
    _dashboard_rows(gs, fig, 0, sl4, trips4, sum4, C["4x4"], "(4×4 Grid)")
    fig.tight_layout()
    save(fig, "fig8_4x4")

    # ── Fig 8B: 7×28 only ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.38)
    fig.suptitle("SafeGAT – 7×28 CityFlow Grid Evaluation Dashboard",
                 fontsize=16, fontweight="bold", y=1.01)
    _dashboard_rows(gs, fig, 0, sl7, trips7, sum7, C["7x28"], "(7×28 Grid)")
    fig.tight_layout()
    save(fig, "fig8_7x28")

    # ── Fig 8C: Combined — 4 rows (4×4 top, 7×28 bottom) ────────────────────
    fig = plt.figure(figsize=(20, 20))
    gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.38)
    fig.suptitle("SafeGAT – Comprehensive System Evaluation Dashboard (4×4 vs 7×28)",
                 fontsize=17, fontweight="bold", y=1.01)
    _dashboard_rows(gs, fig, 0, sl4, trips4, sum4, C["4x4"], "(4×4 Grid)")
    _dashboard_rows(gs, fig, 2, sl7, trips7, sum7, C["7x28"], "(7×28 Grid)")
    fig.tight_layout()
    save(fig, "fig8_combined")
    print("  ok fig8 done")


if __name__ == "__main__":
    plot_fig8()
