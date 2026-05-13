"""
plot_safegat_metrics.py
-----------------------
Generates all evaluation figures for the SafeGAT paper.

Inputs:
  data/output/step_log.json              – per-step simulation telemetry
  data/output/intervention_summary.json  – aggregate intervention stats
  data/output/safegat.tripinfo.xml       – SUMO trip-level metrics (ATT, delay, etc.)

Outputs (saved to data/output/figures/):
  1. reward_curve.pdf           – mean reward over simulation steps
  2. occupancy_margin.pdf       – lane occupancy + Q-value margin
  3. llm_budget.pdf             – cumulative LLM calls vs. remaining budget
  4. att_distribution.pdf       – Average Travel Time distribution (trip-level)
  5. delay_distribution.pdf     – waiting-time / time-loss distribution
  6. intervention_pie.pdf       – breakdown of override vs. safe execution
  7. safety_adjustments_bar.pdf – safety shield activation summary
  8. combined_dashboard.pdf     – all key metrics in one 2×4 grid (paper-ready)

Usage:
  python plot_safegat_metrics.py                    # uses default paths
  python plot_safegat_metrics.py --result_dir <dir> # custom output directory
"""

import os
import json
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

# ── Color palette (colorblind-friendly) ──────────────────────────────────────
C = {
    "safegat":   "#1f77b4",   # blue
    "reward":    "#2ca02c",   # green
    "occ":       "#ff7f0e",   # orange
    "margin":    "#9467bd",   # purple
    "budget":    "#8c564b",   # brown
    "override":  "#d62728",   # red
    "safe":      "#17becf",   # cyan
    "shield":    "#e377c2",   # pink
}

SMOOTH = 10   # rolling-average window for reward / occupancy curves


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def smooth(series: np.ndarray, w: int = SMOOTH) -> np.ndarray:
    """Simple rolling mean with edge handling."""
    return pd.Series(series).rolling(w, min_periods=1).mean().to_numpy()


def parse_tripinfo(xml_path: str) -> pd.DataFrame:
    """Parse SUMO tripinfo XML into a DataFrame with numeric columns."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    for trip in root.findall("tripinfo"):
        rows.append(trip.attrib)
    df = pd.DataFrame(rows)
    numeric_cols = [
        "duration", "waitingTime", "timeLoss", "departDelay",
        "routeLength", "arrivalSpeed", "departSpeed"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save(fig, path: str):
    """Save figure as both PDF and PNG."""
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png")
    plt.close(fig)
    print(f"  saved → {path}.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────────
# Individual plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_reward_curve(steps_df: pd.DataFrame, out: str):
    """Figure 1 – Mean reward over simulation steps."""
    fig, ax = plt.subplots(figsize=(7, 4))
    raw = steps_df["mean_reward"].to_numpy()
    s   = smooth(raw)
    ax.plot(steps_df["step"], s, color=C["reward"], lw=1.8, label="Smoothed reward")
    ax.fill_between(steps_df["step"], raw, s, alpha=0.15, color=C["reward"])
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Mean Reward over Simulation Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, out)


def plot_occupancy_margin(steps_df: pd.DataFrame, out: str):
    """Figure 2 – Lane occupancy and Q-value confidence margin (dual axis)."""
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    occ_s    = smooth(steps_df["mean_occ"].to_numpy())
    margin_s = smooth(steps_df["mean_margin"].to_numpy())

    ax1.plot(steps_df["step"], occ_s,    color=C["occ"],    lw=1.8, label="Lane Occ.")
    ax2.plot(steps_df["step"], margin_s, color=C["margin"], lw=1.8, ls="--", label="Q-margin")

    ax1.set_xlabel("Simulation Step")
    ax1.set_ylabel("Mean Lane Occupancy", color=C["occ"])
    ax2.set_ylabel("Mean Q-value Margin Δ", color=C["margin"])
    ax1.tick_params(axis="y", labelcolor=C["occ"])
    ax2.tick_params(axis="y", labelcolor=C["margin"])

    lines  = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")
    ax1.set_title("Lane Occupancy and Q-Value Confidence Margin")
    ax1.grid(True, alpha=0.3)
    save(fig, out)


def plot_llm_budget(steps_df: pd.DataFrame, out: str):
    """Figure 3 – Cumulative LLM calls vs. remaining budget."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax2 = ax.twinx()

    ax.plot(steps_df["step"], steps_df["llm_calls"],
            color=C["safegat"], lw=1.8, label="Cumulative LLM calls")
    ax2.plot(steps_df["step"], steps_df["budget_left"],
             color=C["budget"], lw=1.8, ls="--", label="Budget remaining")

    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("LLM Calls (cumulative)", color=C["safegat"])
    ax2.set_ylabel("Budget Remaining", color=C["budget"])
    ax.tick_params(axis="y", labelcolor=C["safegat"])
    ax2.tick_params(axis="y", labelcolor=C["budget"])

    lines  = ax.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="center left")
    ax.set_title("LLM Call Budget Consumption")
    ax.grid(True, alpha=0.3)
    save(fig, out)


def plot_att_distribution(trips_df: pd.DataFrame, out: str):
    """Figure 4 – Average Travel Time (duration) distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    data = trips_df["duration"].dropna()
    ax.hist(data, bins=40, color=C["safegat"], edgecolor="white", alpha=0.85)
    ax.axvline(data.mean(), color="red", lw=1.5, ls="--",
               label=f"Mean ATT = {data.mean():.1f} s")
    ax.axvline(data.median(), color="orange", lw=1.5, ls=":",
               label=f"Median = {data.median():.1f} s")
    ax.set_xlabel("Trip Duration (s) — Average Travel Time")
    ax.set_ylabel("Number of Vehicles")
    ax.set_title("Average Travel Time Distribution (SafeGAT)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, out)


def plot_delay_distribution(trips_df: pd.DataFrame, out: str):
    """Figure 5 – Waiting time and time-loss distributions side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, col, label, color in [
        (axes[0], "waitingTime", "Waiting Time (s)", C["occ"]),
        (axes[1], "timeLoss",    "Time Loss (s)",    C["margin"]),
    ]:
        data = trips_df[col].dropna()
        ax.hist(data, bins=35, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(), color="red", lw=1.5, ls="--",
                   label=f"Mean = {data.mean():.1f} s")
        ax.set_xlabel(label)
        ax.set_ylabel("Vehicle Count")
        ax.set_title(f"{label} Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Delay Metrics — SafeGAT (4×4 SUMO Grid)", fontsize=13)
    fig.tight_layout()
    save(fig, out)


def plot_intervention_pie(summary: dict, out: str):
    """Figure 6 – Intervention decision breakdown pie chart.
    
    Note: llm_calls in the summary is CUMULATIVE over all steps (each step
    may invoke the LLM for multiple nodes). We use raw counts directly here
    to show the breakdown of what happened to each LLM invocation.
    """
    overrides   = summary["llm_overrides"]
    calls       = summary["llm_calls"]
    safe_adj    = summary["safety_adjustments"]
    no_override = calls - overrides   # LLM was called but RL action kept

    # Pie: what happened to each LLM invocation?
    sizes  = [overrides, no_override, safe_adj]
    labels = [
        f"LLM Override\n({overrides})",
        f"LLM → RL kept\n({no_override})",
        f"Safety Shield\nAdj. ({safe_adj})",
    ]
    colors  = [C["override"], C["safegat"], C["shield"]]
    explode = (0.06, 0, 0.06)

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        explode=explode, pctdistance=0.78,
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title("LLM Invocation Outcome Breakdown\n(SafeGAT – 4×4 Grid)")
    save(fig, out)


def plot_safety_summary_bar(summary: dict, out: str):
    """Figure 7 – Bar chart of key intervention counts."""
    keys   = ["LLM Calls", "LLM Overrides", "Safety Adjustments"]
    vals   = [summary["llm_calls"], summary["llm_overrides"], summary["safety_adjustments"]]
    colors = [C["safegat"], C["override"], C["shield"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(keys, vals, color=colors, width=0.5, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(v), ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title(f"Intervention & Safety Statistics\n"
                 f"(Override rate: {summary['override_rate_%']}%  |  "
                 f"Mean margin @ call: {summary['mean_margin_at_call']:.4f})")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, out)


def plot_combined_dashboard(steps_df: pd.DataFrame,
                            trips_df: pd.DataFrame,
                            summary: dict,
                            out: str):
    """Figure 8 – Paper-ready 2×4 dashboard combining all key metrics."""
    fig = plt.figure(figsize=(18, 9))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    # ── Row 0 ─────────────────────────────────────────────────────────────────

    # (0,0) Reward
    ax = fig.add_subplot(gs[0, 0])
    raw = steps_df["mean_reward"].to_numpy()
    ax.plot(steps_df["step"], smooth(raw), color=C["reward"], lw=1.5)
    ax.fill_between(steps_df["step"], raw, smooth(raw), alpha=0.12, color=C["reward"])
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_title("Mean Reward")
    ax.set_xlabel("Step"); ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.25)

    # (0,1) Occupancy + Margin
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = ax1.twinx()
    ax1.plot(steps_df["step"], smooth(steps_df["mean_occ"].to_numpy()),
             color=C["occ"], lw=1.5, label="Occ.")
    ax2.plot(steps_df["step"], smooth(steps_df["mean_margin"].to_numpy()),
             color=C["margin"], lw=1.5, ls="--", label="Margin")
    ax1.set_title("Occupancy & Q-Margin")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Occ.", color=C["occ"])
    ax2.set_ylabel("Δ", color=C["margin"])
    ax1.tick_params(axis="y", labelcolor=C["occ"])
    ax2.tick_params(axis="y", labelcolor=C["margin"])
    ax1.grid(True, alpha=0.25)

    # (0,2) LLM Budget
    ax1 = fig.add_subplot(gs[0, 2])
    ax2 = ax1.twinx()
    ax1.plot(steps_df["step"], steps_df["llm_calls"],
             color=C["safegat"], lw=1.5, label="Calls")
    ax2.plot(steps_df["step"], steps_df["budget_left"],
             color=C["budget"], lw=1.5, ls="--", label="Budget")
    ax1.set_title("LLM Budget")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Calls", color=C["safegat"])
    ax2.set_ylabel("Remaining", color=C["budget"])
    ax1.tick_params(axis="y", labelcolor=C["safegat"])
    ax2.tick_params(axis="y", labelcolor=C["budget"])
    ax1.grid(True, alpha=0.25)

    # (0,3) Intervention Pie — breakdown of LLM invocation outcomes
    ax = fig.add_subplot(gs[0, 3])
    overrides   = summary["llm_overrides"]
    calls       = summary["llm_calls"]
    safe_adj    = summary["safety_adjustments"]
    ax.pie(
        [overrides, calls - overrides, safe_adj],
        labels=["Override", "LLM→RL", "Shield"],
        colors=[C["override"], C["safegat"], C["shield"]],
        autopct="%1.0f%%", startangle=140,
        pctdistance=0.75, textprops={"fontsize": 8},
    )
    ax.set_title("LLM Outcome")

    # ── Row 1 ─────────────────────────────────────────────────────────────────

    # (1,0) ATT histogram
    ax = fig.add_subplot(gs[1, 0])
    att = trips_df["duration"].dropna()
    ax.hist(att, bins=30, color=C["safegat"], edgecolor="white", alpha=0.85)
    ax.axvline(att.mean(), color="red", lw=1.3, ls="--", label=f"μ={att.mean():.0f}s")
    ax.set_title("Travel Time (ATT)")
    ax.set_xlabel("Duration (s)"); ax.set_ylabel("Vehicles")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    # (1,1) Waiting Time
    ax = fig.add_subplot(gs[1, 1])
    wt = trips_df["waitingTime"].dropna()
    ax.hist(wt, bins=30, color=C["occ"], edgecolor="white", alpha=0.85)
    ax.axvline(wt.mean(), color="red", lw=1.3, ls="--", label=f"μ={wt.mean():.1f}s")
    ax.set_title("Waiting Time (QL proxy)")
    ax.set_xlabel("Waiting Time (s)"); ax.set_ylabel("Vehicles")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    # (1,2) Time Loss
    ax = fig.add_subplot(gs[1, 2])
    tl = trips_df["timeLoss"].dropna()
    ax.hist(tl, bins=30, color=C["margin"], edgecolor="white", alpha=0.85)
    ax.axvline(tl.mean(), color="red", lw=1.3, ls="--", label=f"μ={tl.mean():.1f}s")
    ax.set_title("Time Loss (Delay)")
    ax.set_xlabel("Time Loss (s)"); ax.set_ylabel("Vehicles")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    # (1,3) Intervention bar summary
    ax = fig.add_subplot(gs[1, 3])
    keys = ["LLM\nCalls", "LLM\nOverrides", "Safety\nAdj."]
    vals = [summary["llm_calls"], summary["llm_overrides"], summary["safety_adjustments"]]
    cols = [C["safegat"], C["override"], C["shield"]]
    bars = ax.bar(keys, vals, color=cols, width=0.5, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(v), ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Intervention Summary\n(override rate {summary['override_rate_%']}%)")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("SafeGAT – 4×4 SUMO Grid Evaluation Dashboard", fontsize=14, y=1.01)
    save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate SafeGAT paper figures.")
    parser.add_argument("--result_dir", default="data/output",
                        help="Path to the result directory (default: data/output)")
    args = parser.parse_args()

    base    = args.result_dir
    fig_dir = os.path.join(base, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    step_log_path     = os.path.join(base, "step_log.json")
    summary_path      = os.path.join(base, "intervention_summary.json")
    tripinfo_path     = os.path.join(base, "safegat.tripinfo.xml")

    print("Loading step_log.json …")
    with open(step_log_path) as f:
        steps_df = pd.DataFrame(json.load(f))

    print("Loading intervention_summary.json …")
    with open(summary_path) as f:
        summary = json.load(f)

    print("Loading tripinfo XML …")
    trips_df = parse_tripinfo(tripinfo_path)

    print(f"\nSimulation steps: {len(steps_df)}")
    print(f"Trips completed : {len(trips_df)}")
    print(f"Summary         : {summary}\n")

    # ── Generate figures ───────────────────────────────────────────────────────
    print("Generating figures …")
    plot_reward_curve       (steps_df,                          os.path.join(fig_dir, "1_reward_curve"))
    plot_occupancy_margin   (steps_df,                          os.path.join(fig_dir, "2_occupancy_margin"))
    plot_llm_budget         (steps_df,                          os.path.join(fig_dir, "3_llm_budget"))
    plot_att_distribution   (trips_df,                          os.path.join(fig_dir, "4_att_distribution"))
    plot_delay_distribution (trips_df,                          os.path.join(fig_dir, "5_delay_distribution"))
    plot_intervention_pie   (summary,                           os.path.join(fig_dir, "6_intervention_pie"))
    plot_safety_summary_bar (summary,                           os.path.join(fig_dir, "7_safety_summary_bar"))
    plot_combined_dashboard (steps_df, trips_df, summary,       os.path.join(fig_dir, "8_combined_dashboard"))

    # ── Print key metrics for the paper ───────────────────────────────────────
    att   = trips_df["duration"].dropna()
    wt    = trips_df["waitingTime"].dropna()
    tl    = trips_df["timeLoss"].dropna()
    print("\n=== KEY METRICS FOR PAPER TABLE ===")
    print(f"ATT  (mean ± std): {att.mean():.2f} ± {att.std():.2f} s")
    print(f"ATT  (median)    : {att.median():.2f} s")
    print(f"Wait (mean ± std): {wt.mean():.2f} ± {wt.std():.2f} s")
    print(f"TimeLoss (mean)  : {tl.mean():.2f} s")
    print(f"Throughput       : {len(trips_df)} vehicles completed")
    print(f"LLM calls        : {summary['llm_calls']} / {summary['total_sim_steps']} steps")
    print(f"Override rate    : {summary['override_rate_%']}%")
    print(f"Safety adj.      : {summary['safety_adjustments']}")
    print(f"Mean margin@call : {summary['mean_margin_at_call']}")
    print("===================================\n")
    print(f"All figures saved to: {os.path.abspath(fig_dir)}/")


if __name__ == "__main__":
    main()
