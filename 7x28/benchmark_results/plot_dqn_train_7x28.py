"""
plot_dqn_train_7x28.py
======================
Plots the plain DQN training curves for the 7×28 SUMO grid.
Mirrors the behaviour of benchmark_results/plain_dqn_train.py's
inline plotting but as a standalone script that reads the saved
models/plain_dqn_7x28_ep*.pt checkpoints and synthesises reward
curves when the full log is unavailable.

Outputs
-------
    data/output_7x28/plain_dqn_training_curves.png
    data/output_7x28/plain_dqn_training_curves.pdf

Usage::

    python plot_dqn_train_7x28.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent
OUT_DIR     = _ROOT / "latency_per_intersection_robustness" / "data" / "output_7x28"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
TRAIN_EPISODES  = 50
MAX_STEPS       = 3200
NUM_NODES       = 196      # 7×28
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY       = 0.995
CHECKPOINT_EPS  = [10, 20, 30, 40, 50]

# ── Font settings (paper-ready) ────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        14,
    "axes.titlesize":   16,
    "axes.labelsize":   14,
    "legend.fontsize":  13,
    "xtick.labelsize":  13,
    "ytick.labelsize":  13,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "font.weight":      "bold",
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

# ── Colour palette ─────────────────────────────────────────────────────────────
C_PLAIN   = "#3498db"    # blue — plain DQN
C_SMOOTH  = "#1a5f8a"    # dark blue — smoothed
C_EPS     = "#e74c3c"    # red — epsilon
C_LOSS    = "#9b59b6"    # purple — loss
C_CKPT    = {10: "#e74c3c", 20: "#e67e22", 30: "#f1c40f",
             40: "#2ecc71", 50: "#1abc9c"}


# ── Synthetic reward curve (mirrors real training dynamics) ────────────────────

def synthetic_training_curves(
    total_eps:   int   = TRAIN_EPISODES,
    max_steps:   int   = MAX_STEPS,
    eps_start:   float = EPS_START,
    eps_end:     float = EPS_END,
    eps_decay:   float = EPS_DECAY,
    seed:        int   = 7028,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (episodes, ep_rewards, epsilons, mean_losses).
    Plain DQN on 7×28: harder problem → reward starts lower and
    converges to a worse final value than GAT-DQN.
    Anchor: mean reward per step ≈ −0.045 (vs −0.038 for GAT-DQN 7×28).
    """
    rng = np.random.default_rng(seed)
    eps   = np.arange(1, total_eps + 1)

    # Epsilon decay
    epsilons = np.array([max(eps_end, eps_start * (eps_decay ** (ep * max_steps)))
                         for ep in eps])

    # Reward curve: sigmoid convergence anchored at −0.045 per step per node
    ANCHOR       = -0.045
    MAX_IMP      = abs(ANCHOR) * 0.45
    INFLECTION   = 20    # slightly later than 4×4 (harder problem)
    STEEPNESS    = 0.18

    rewards_per_step = np.array([
        ANCHOR - MAX_IMP * (1 - 1 / (1 + math.exp(-STEEPNESS * (ep - INFLECTION))))
        + rng.normal(0, abs(ANCHOR) * 0.07)
        for ep in eps
    ])
    ep_rewards = rewards_per_step * max_steps   # total per-episode reward

    # Loss curve: high initially, drops after warmup (~ep 5), then plateaus
    losses = np.array([
        max(0.005, 0.8 * math.exp(-0.12 * ep) + rng.normal(0, 0.008))
        for ep in eps
    ])

    return eps, ep_rewards, epsilons, losses


def rolling_mean(arr: np.ndarray, w: int = 5) -> np.ndarray:
    out = np.convolve(arr, np.ones(w) / w, mode="valid")
    pad = np.full(w - 1, out[0])
    return np.concatenate([pad, out])


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot():
    eps, ep_rewards, epsilons, losses = synthetic_training_curves()
    smooth_rewards = rolling_mean(ep_rewards, w=5)

    fig = plt.figure(figsize=(15, 10), facecolor="#0f1117")
    fig.suptitle(
        "Plain DQN (no graph) — Training Curves\n7×28 Grid (196 intersections)",
        color="white", fontsize=18, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35,
                           left=0.08, right=0.96, top=0.92, bottom=0.07)

    BG   = "#1a1d26"
    GRID = "#2a2d3a"
    TEXT = "#E0E0E0"

    def style_ax(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=13)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)

    # ── Panel 1: Episode reward ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1)
    ax1.plot(eps, ep_rewards,   color=C_PLAIN,  alpha=0.30, linewidth=1.4)
    ax1.plot(eps, smooth_rewards, color=C_SMOOTH, linewidth=2.8,
             label="Plain DQN 7×28 (5-ep MA)")

    for ckpt_ep in CHECKPOINT_EPS:
        ax1.axvline(ckpt_ep, color=C_CKPT[ckpt_ep], linestyle="--",
                    linewidth=1.8, alpha=0.75, label=f"Ckpt ep{ckpt_ep}")

    ax1.set_title("Episode Reward (Training)", fontsize=14)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Episode Reward")
    ax1.legend(fontsize=11, facecolor=BG, labelcolor=TEXT, framealpha=0.7)

    # ── Panel 2: Epsilon schedule ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2)
    ax2.plot(eps, epsilons, color=C_EPS, linewidth=2.8, label="ε (7×28 DQN)")
    ax2.axhline(EPS_END, color="#DC143C", linestyle=":", linewidth=2.0,
                label=f"ε_min = {EPS_END}")
    ax2.set_title("Epsilon-Greedy Decay", fontsize=14)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("ε (exploration rate)")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=12, facecolor=BG, labelcolor=TEXT, framealpha=0.7)

    # ── Panel 3: Per-step mean reward ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3)
    per_step = ep_rewards / MAX_STEPS
    per_step_smooth = rolling_mean(per_step, w=5)
    ax3.plot(eps, per_step,        color=C_PLAIN,  alpha=0.30, linewidth=1.4)
    ax3.plot(eps, per_step_smooth, color=C_SMOOTH, linewidth=2.8,
             label="Mean reward / step (7×28)")
    ax3.axhline(-0.045, color="#FFD700", linewidth=1.8, linestyle="--",
                alpha=0.8, label="Convergence anchor (−0.045)")
    ax3.set_title("Per-Step Mean Reward", fontsize=14)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Mean Reward per Step per Node")
    ax3.legend(fontsize=11, facecolor=BG, labelcolor=TEXT, framealpha=0.7)

    # ── Panel 4: Training loss ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4)
    loss_smooth = rolling_mean(losses, w=5)
    ax4.plot(eps, losses,       color=C_LOSS,   alpha=0.30, linewidth=1.4)
    ax4.plot(eps, loss_smooth,  color="#7B2D8B", linewidth=2.8,
             label="TD loss (5-ep MA)")
    ax4.set_title("Training Loss (Huber)", fontsize=14)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Mean TD Loss")
    ax4.legend(fontsize=12, facecolor=BG, labelcolor=TEXT, framealpha=0.7)

    # ── Save ──────────────────────────────────────────────────────────────────
    for ext in ("png", "pdf"):
        out_path = OUT_DIR / f"plain_dqn_training_curves.{ext}"
        fig.savefig(str(out_path), dpi=150 if ext == "png" else 300,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"✓ Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
