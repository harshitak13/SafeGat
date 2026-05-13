"""
fig2_training_convergence.py
============================
Figure 2 — Training Convergence & Inference Stability
Compares SafeGAT-iLLM vs CoLight vs LLMLight training curves.

Produces:
  fig2_4x4       — 4x4: training convergence (3 methods) + inference panels
  fig2_7x28      — 7x28: training convergence (3 methods) + inference panels
  fig2_combined  — side-by-side 4x4 vs 7x28 convergence + inference comparison
"""

from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from _shared import *

COL_SG = "#1f77b4"   # SafeGAT-iLLM  blue
COL_CL = "#2ca02c"   # CoLight        green
COL_LL = "#d62728"   # LLMLight       red


def _normalise(arr):
    """Min-max normalise so all methods share a [0,1] y-axis (higher = better)."""
    a = np.array(arr, dtype=float)
    lo, hi = a.min(), a.max()
    return np.zeros_like(a) if hi == lo else (a - lo) / (hi - lo)


def _load_baselines(grid: str):
    base = ROOT / "data" / grid
    with open(base / "colight_rewards.json")  as f: col = json.load(f)
    with open(base / "llmlight_rewards.json") as f: llm = json.load(f)
    return col, llm


def _load_safegat_4x4():
    tc = load_4x4_training()
    return tc["episode_rewards_raw"], tc["episode_rewards_smooth"], tc["epsilon_per_episode"]


def _load_safegat_7x28():
    tc  = load_7x28_training()
    raw = [e["total_reward"] for e in tc]
    eps = [e["epsilon"]      for e in tc]
    return raw, smooth(raw, 7).tolist(), eps


def _training_figure(sg_raw, sg_smooth, epsilons,
                     col_raw, llm_raw,
                     inf_rewards, inf_occ, inf_margins, inf_budget,
                     title, grid_label):
    fig = plt.figure(figsize=(18, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.36)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)

    n_cmp  = min(len(sg_raw), len(col_raw), len(llm_raw))
    ep_cmp = np.arange(1, n_cmp + 1)
    sg_n   = _normalise(sg_raw [:n_cmp])
    col_n  = _normalise(col_raw[:n_cmp])
    llm_n  = _normalise(llm_raw[:n_cmp])
    n_sg   = len(sg_raw)
    ep_all = np.arange(1, n_sg + 1)

    # (0,0) Normalised convergence — all 3 methods
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ep_cmp, smooth(sg_n,  7), color=COL_SG, lw=2.4,        label="SafeGAT-iLLM (ours)")
    ax.plot(ep_cmp, smooth(col_n, 7), color=COL_CL, lw=2.0, ls="--", label="CoLight")
    ax.plot(ep_cmp, smooth(llm_n, 7), color=COL_LL, lw=2.0, ls=":",  label="LLMLight")
    ax.set_title(f"Training Convergence ({grid_label})\nNormalised reward — all methods", fontweight="bold")
    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_ylabel("Normalised Reward (0→worst, 1→best)", fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.15)

    # (0,1) Raw rewards — all methods + smoothed
    ax = fig.add_subplot(gs[0, 1])
    n_max = max(len(sg_raw), len(col_raw), len(llm_raw))
    for raw, cc, lbl2 in [(sg_raw, COL_SG, "SafeGAT-iLLM"),
                           (col_raw, COL_CL, "CoLight"),
                           (llm_raw, COL_LL, "LLMLight")]:
        ep_r = np.arange(1, len(raw)+1)
        ax.plot(ep_r, raw,           color=cc, lw=0.7, alpha=0.3)
        ax.plot(ep_r, smooth(raw, 7), color=cc, lw=2.2, label=lbl2)
    ax.set_title(f"Raw Episode Rewards ({grid_label})\n7-episode moving average", fontweight="bold")
    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_ylabel("Episode Reward", fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # (0,2) Epsilon schedule
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(ep_all, epsilons, color=COL_SG, lw=2.4)
    ax.axhline(0.05, color="red", lw=1.5, ls="--", label="ε_min = 0.05")
    ax.set_title(f"SafeGAT Epsilon Schedule\n({grid_label})", fontweight="bold")
    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_ylabel("ε (exploration rate)", fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

    # (1,0) Inference reward + occupancy
    steps = np.arange(len(inf_rewards))
    ax1   = fig.add_subplot(gs[1, 0])
    ax2   = ax1.twinx()
    ax1.plot(steps, smooth(inf_rewards, 20), color=COL_SG, lw=2.2, label="Reward (20-step MA)")
    ax2.plot(steps, smooth(inf_occ,     10), color=C["occ"], lw=2.0, alpha=0.85, label="Mean occ")
    ax1.set_title(f"Inference: Reward & Occupancy\n({grid_label})", fontweight="bold")
    ax1.set_xlabel("Step", fontweight="bold")
    ax1.set_ylabel("Mean Step Reward", color=COL_SG, fontweight="bold")
    ax2.set_ylabel("Mean Occupancy",   color=C["occ"], fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=COL_SG)
    ax2.tick_params(axis="y", labelcolor=C["occ"])
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=9, loc="lower right")
    ax1.grid(alpha=0.3)

    # (1,1) Q-confidence margin + LLM budget
    ax1 = fig.add_subplot(gs[1, 1])
    ax2 = ax1.twinx()
    ax1.plot(steps, smooth(inf_margins, 5), color=C["occ"], lw=2.0, label="Confidence margin")
    ax1.axhline(0.05, color="red", lw=1.5, ls="--", label="T_c = 0.05")
    budget_pct = (1 - np.array(inf_budget) / max(max(inf_budget), 1)) * 100
    ax2.plot(steps, budget_pct, color="#8c564b", lw=2.0, label="Budget used %")
    ax1.set_title(f"Q-Confidence Margin & LLM Budget\n({grid_label})", fontweight="bold")
    ax1.set_xlabel("Step", fontweight="bold")
    ax1.set_ylabel("Mean Confidence Margin", color=C["occ"],    fontweight="bold")
    ax2.set_ylabel("LLM Budget Used %",      color="#8c564b",   fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=C["occ"])
    ax2.tick_params(axis="y", labelcolor="#8c564b")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=9, loc="upper left")
    ax1.grid(alpha=0.3)

    # (1,2) Final convergence bar — mean of last 10 episodes (normalised)
    ax = fig.add_subplot(gs[1, 2])
    last_n = min(10, n_cmp)
    means  = [float(np.mean(_normalise(r)[-last_n:])) for r in [sg_raw, col_raw, llm_raw]]
    stds   = [float(np.std( _normalise(r)[-last_n:])) for r in [sg_raw, col_raw, llm_raw]]
    bars = ax.bar(["SafeGAT-iLLM", "CoLight", "LLMLight"], means,
                  yerr=stds, capsize=6,
                  color=[COL_SG, COL_CL, COL_LL], edgecolor="white", width=0.5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(f"Mean Normalised Reward\nLast {last_n} Episodes ({grid_label})", fontweight="bold")
    ax.set_ylabel("Normalised Reward (↑ better)", fontweight="bold")
    ax.set_ylim(0, 1.3); ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


def _combined_figure(sg4_raw, col4_raw, llm4_raw,
                     sg7_raw, col7_raw, llm7_raw,
                     inf4_r, inf4_occ, inf7_r, inf7_occ):
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        "SafeGAT-iLLM vs CoLight vs LLMLight — Training Convergence (4×4 & 7×28 Grids)",
        fontsize=16, fontweight="bold", y=1.01
    )

    for row, (sg_raw, col_raw, llm_raw, inf_r, inf_occ, lbl) in enumerate([
        (sg4_raw, col4_raw, llm4_raw, inf4_r, inf4_occ, "4×4 Grid"),
        (sg7_raw, col7_raw, llm7_raw, inf7_r, inf7_occ, "7×28 Grid"),
    ]):
        n_cmp  = min(len(sg_raw), len(col_raw), len(llm_raw))
        ep_cmp = np.arange(1, n_cmp + 1)

        # Col 0 — normalised convergence
        ax = axes[row, 0]
        ax.plot(ep_cmp, smooth(_normalise(sg_raw [:n_cmp]), 7), color=COL_SG, lw=2.4, label="SafeGAT-iLLM")
        ax.plot(ep_cmp, smooth(_normalise(col_raw[:n_cmp]), 7), color=COL_CL, lw=2.0, ls="--", label="CoLight")
        ax.plot(ep_cmp, smooth(_normalise(llm_raw[:n_cmp]), 7), color=COL_LL, lw=2.0, ls=":",  label="LLMLight")
        ax.set_title(f"Training Convergence — {lbl}\n(normalised reward)", fontweight="bold")
        ax.set_xlabel("Episode", fontweight="bold")
        ax.set_ylabel("Normalised Reward (↑ better)", fontweight="bold")
        ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.15)

        # Col 1 — raw reward all methods
        ax = axes[row, 1]
        for raw, cc, lbl2 in [(sg_raw, COL_SG, "SafeGAT-iLLM"),
                               (col_raw, COL_CL, "CoLight"),
                               (llm_raw, COL_LL, "LLMLight")]:
            ep_r = np.arange(1, len(raw)+1)
            ax.plot(ep_r, raw,            color=cc, lw=0.7, alpha=0.3)
            ax.plot(ep_r, smooth(raw, 7), color=cc, lw=2.2, label=lbl2)
        ax.set_title(f"Raw Episode Rewards — {lbl}", fontweight="bold")
        ax.set_xlabel("Episode", fontweight="bold")
        ax.set_ylabel("Episode Reward", fontweight="bold")
        ax.legend(fontsize=10); ax.grid(alpha=0.3)

        # Col 2 — inference reward + occupancy
        steps = np.arange(len(inf_r))
        ax1   = axes[row, 2]
        ax2   = ax1.twinx()
        ax1.plot(steps, smooth(inf_r,   20), color=COL_SG, lw=2.2, label="SafeGAT Reward")
        ax2.plot(steps, smooth(inf_occ, 10), color=C["occ"], lw=2.0, alpha=0.85, label="Mean occ")
        ax1.set_title(f"SafeGAT Inference — {lbl}", fontweight="bold")
        ax1.set_xlabel("Step", fontweight="bold")
        ax1.set_ylabel("Mean Step Reward", color=COL_SG, fontweight="bold")
        ax2.set_ylabel("Mean Occupancy",   color=C["occ"], fontweight="bold")
        ax1.tick_params(axis="y", labelcolor=COL_SG)
        ax2.tick_params(axis="y", labelcolor=C["occ"])
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], fontsize=9)
        ax1.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_fig2():
    sg4_raw, sg4_sm, eps4 = _load_safegat_4x4()
    col4_raw, llm4_raw    = _load_baselines("4x4")
    tc4  = load_4x4_training()
    sl4  = load_4x4_steplog()
    inf4_r   = tc4["inference_step_reward"]
    inf4_occ = tc4["inference_mean_occ"]
    inf4_mar = tc4["inference_mean_margin"]
    inf4_bud = sl4["budget_left"].tolist()

    sg7_raw, sg7_sm, eps7 = _load_safegat_7x28()
    col7_raw, llm7_raw    = _load_baselines("7x28")
    comb7 = load_7x28_combined()
    inf7  = comb7["inference_steps"]
    inf7_r   = [s["mean_reward"]  for s in inf7]
    inf7_occ = [s["mean_occ"]     for s in inf7]
    inf7_mar = [s["mean_margin"]  for s in inf7]
    inf7_bud = [s["budget_left"]  for s in inf7]

    fig = _training_figure(sg4_raw, sg4_sm, eps4, col4_raw, llm4_raw,
                           inf4_r, inf4_occ, inf4_mar, inf4_bud,
                           "SafeGAT-iLLM vs CoLight vs LLMLight — Training Convergence (4×4 Grid)",
                           "4×4")
    save(fig, "fig2_4x4")

    fig = _training_figure(sg7_raw, sg7_sm, eps7, col7_raw, llm7_raw,
                           inf7_r, inf7_occ, inf7_mar, inf7_bud,
                           "SafeGAT-iLLM vs CoLight vs LLMLight — Training Convergence (7×28 Grid)",
                           "7×28")
    save(fig, "fig2_7x28")

    fig = _combined_figure(sg4_raw, col4_raw, llm4_raw,
                           sg7_raw, col7_raw, llm7_raw,
                           inf4_r, inf4_occ, inf7_r, inf7_occ)
    save(fig, "fig2_combined")
    print("  ok fig2 done")


if __name__ == "__main__":
    plot_fig2()
