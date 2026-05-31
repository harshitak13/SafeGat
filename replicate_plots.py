"""Replicate SafeGAT plots into a separate output folder.

This script scans the current result folders for 4x4, 7x28, CityFlow
datasets, and LLM_outputs runs. It regenerates comparable figures without
writing back into the training/output directories.

Usage:
    python replicate_plots.py
    python replicate_plots.py --out-dir replicated_plot_outputs
    python replicate_plots.py --formats png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

COLORS = {
    "reward": "#2ca02c",
    "occ": "#ff7f0e",
    "margin": "#9467bd",
    "llm": "#1f77b4",
    "budget": "#8c564b",
    "override": "#d62728",
    "shield": "#e377c2",
    "queue": "#17becf",
    "gray": "#7f7f7f",
}

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


@dataclass(frozen=True)
class Run:
    label: str
    source_dir: Path
    output_dir: Path
    family: str
    model: str | None = None


def slug_part(part: str) -> str:
    part = part.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", part).strip("_") or "run"


def output_path_for(out_root: Path, label: str) -> Path:
    return out_root.joinpath(*(slug_part(p) for p in Path(label).parts))


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_step_log(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(read_json(path))
    if "step" not in df.columns:
        df.insert(0, "step", np.arange(len(df)))
    return df


def load_tripinfo(path: Path) -> pd.DataFrame:
    rows = [trip.attrib for trip in ET.parse(str(path)).getroot().findall("tripinfo")]
    df = pd.DataFrame(rows)
    for col in [
        "duration",
        "waitingTime",
        "timeLoss",
        "departDelay",
        "routeLength",
        "arrivalSpeed",
        "departSpeed",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_per_junction(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def smooth(values, window: int = 10) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rolling(window, min_periods=1).mean().to_numpy()


def save(fig, out_dir: Path, stem: str, formats: list[str]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path)
        saved.append(path)
    plt.close(fig)
    return saved


def has_cols(df: pd.DataFrame, *cols: str) -> bool:
    return all(col in df.columns for col in cols)


def plot_reward_curve(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    raw = pd.to_numeric(df["mean_reward"], errors="coerce").fillna(0).to_numpy()
    ax.plot(df["step"], raw, color=COLORS["reward"], alpha=0.25, lw=1.0, label="raw")
    ax.plot(df["step"], smooth(raw, 20), color=COLORS["reward"], lw=2.0, label="20-step moving average")
    ax.axhline(0, color=COLORS["gray"], lw=0.8, ls="--")
    ax.set_title(f"Mean Reward - {title}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_occupancy_margin(df: pd.DataFrame, title: str):
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax2 = ax1.twinx()
    occ = pd.to_numeric(df["mean_occ"], errors="coerce").fillna(0).to_numpy()
    margin = pd.to_numeric(df["mean_margin"], errors="coerce").fillna(0).to_numpy()
    ax1.plot(df["step"], smooth(occ, 20), color=COLORS["occ"], lw=2.0, label="mean occupancy")
    ax2.plot(df["step"], smooth(margin, 20), color=COLORS["margin"], lw=2.0, ls="--", label="Q-margin")
    ax1.set_title(f"Occupancy and Confidence Margin - {title}")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Mean occupancy", color=COLORS["occ"])
    ax2.set_ylabel("Mean Q-margin", color=COLORS["margin"])
    ax1.tick_params(axis="y", labelcolor=COLORS["occ"])
    ax2.tick_params(axis="y", labelcolor=COLORS["margin"])
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.grid(True, alpha=0.3)
    return fig


def plot_llm_budget(df: pd.DataFrame, title: str):
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax2 = ax1.twinx()
    calls = pd.to_numeric(df["llm_calls"], errors="coerce").fillna(0).to_numpy()
    ax1.plot(df["step"], calls, color=COLORS["llm"], lw=2.0, label="LLM calls")
    if "budget_left" in df.columns:
        budget = pd.to_numeric(df["budget_left"], errors="coerce").fillna(0).to_numpy()
        ax2.plot(df["step"], budget, color=COLORS["budget"], lw=2.0, ls="--", label="budget left")
        ax2.set_ylabel("Budget left", color=COLORS["budget"])
        ax2.tick_params(axis="y", labelcolor=COLORS["budget"])
    ax1.set_title(f"LLM Calls and Budget - {title}")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cumulative LLM calls", color=COLORS["llm"])
    ax1.tick_params(axis="y", labelcolor=COLORS["llm"])
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.grid(True, alpha=0.3)
    return fig


def plot_tripinfo(trips: pd.DataFrame, title: str, column: str, label: str):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    data = pd.to_numeric(trips[column], errors="coerce").dropna()
    ax.hist(data, bins=35, color=COLORS["llm"], edgecolor="white", alpha=0.85)
    if not data.empty:
        ax.axvline(data.mean(), color=COLORS["override"], lw=1.6, ls="--", label=f"mean={data.mean():.1f}")
        ax.axvline(data.median(), color=COLORS["occ"], lw=1.6, ls=":", label=f"median={data.median():.1f}")
    ax.set_title(f"{label} Distribution - {title}")
    ax.set_xlabel(label)
    ax.set_ylabel("Vehicle count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_intervention_summary(summary: dict, title: str):
    labels = ["LLM calls", "Overrides", "Safety adj.", "Failures"]
    values = [
        summary.get("llm_calls", 0),
        summary.get("llm_overrides", 0),
        summary.get("safety_adjustments", 0),
        summary.get("llm_failures", 0),
    ]
    colors = [COLORS["llm"], COLORS["override"], COLORS["shield"], COLORS["gray"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.58, edgecolor="white")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom")
    ax.set_title(f"Intervention Summary - {title}")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def plot_intervention_pie(summary: dict, title: str):
    calls = int(summary.get("llm_calls", 0) or 0)
    overrides = int(summary.get("llm_overrides", 0) or 0)
    safety = int(summary.get("safety_adjustments", 0) or 0)
    failures = int(summary.get("llm_failures", 0) or 0)
    kept = max(calls - overrides - failures, 0)
    sizes = [overrides, kept, safety, failures]
    labels = ["override", "kept/fallback", "safety adj.", "failure"]
    nonzero = [(s, l) for s, l in zip(sizes, labels) if s > 0]
    fig, ax = plt.subplots(figsize=(6, 5))
    if nonzero:
        ax.pie(
            [x[0] for x in nonzero],
            labels=[x[1] for x in nonzero],
            autopct="%1.1f%%",
            startangle=140,
            colors=[COLORS["override"], COLORS["llm"], COLORS["shield"], COLORS["gray"]][: len(nonzero)],
        )
    else:
        ax.text(0.5, 0.5, "No LLM calls", ha="center", va="center")
        ax.axis("off")
    ax.set_title(f"LLM Invocation Outcomes - {title}")
    return fig


def plot_training_curve(training: pd.DataFrame, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ep = training["episode"] if "episode" in training.columns else np.arange(1, len(training) + 1)
    reward_col = "total_reward" if "total_reward" in training.columns else "mean_reward"
    rewards = pd.to_numeric(training[reward_col], errors="coerce").fillna(0).to_numpy()
    axes[0].plot(ep, rewards, color=COLORS["reward"], alpha=0.3, lw=1.0)
    axes[0].plot(ep, smooth(rewards, 7), color=COLORS["reward"], lw=2.0, label="7-episode moving average")
    axes[0].set_title("Episode reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel(reward_col)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    if "epsilon" in training.columns:
        eps = pd.to_numeric(training["epsilon"], errors="coerce").fillna(0).to_numpy()
        axes[1].plot(ep, eps, color=COLORS["llm"], lw=2.0)
        axes[1].set_title("Epsilon schedule")
        axes[1].set_ylabel("epsilon")
    elif "mean_loss" in training.columns:
        loss = pd.to_numeric(training["mean_loss"], errors="coerce").fillna(0).to_numpy()
        axes[1].plot(ep, loss, color=COLORS["margin"], lw=2.0)
        axes[1].set_title("Mean loss")
        axes[1].set_ylabel("mean_loss")
    else:
        axes[1].axis("off")
    axes[1].set_xlabel("Episode")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"Training Curve - {title}", y=1.02)
    fig.tight_layout()
    return fig


def parse_grid_position(junction_id: str, fallback_index: int, total: int) -> tuple[int, int]:
    match = re.search(r"(\d+)_(\d+)$", junction_id)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"J(\d+)$", junction_id)
    if match:
        cols = max(1, math.ceil(math.sqrt(total)))
        n = int(match.group(1)) - 1
        return n // cols, n % cols
    cols = max(1, math.ceil(math.sqrt(total)))
    return fallback_index // cols, fallback_index % cols


def plot_per_junction_bars(per_junction: pd.DataFrame, title: str):
    df = per_junction.copy()
    if "mean_reward" not in df.columns or "junction_id" not in df.columns:
        raise ValueError("per_junction_results.csv needs junction_id and mean_reward")
    df["mean_reward"] = pd.to_numeric(df["mean_reward"], errors="coerce").fillna(0)
    best = df.nlargest(min(12, len(df)), "mean_reward")
    worst = df.nsmallest(min(12, len(df)), "mean_reward")
    plot_df = pd.concat([best, worst]).drop_duplicates("junction_id")
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [COLORS["reward"] if val >= plot_df["mean_reward"].median() else COLORS["override"] for val in plot_df["mean_reward"]]
    ax.bar(plot_df["junction_id"], plot_df["mean_reward"], color=colors)
    ax.set_title(f"Best/Worst Junction Rewards - {title}")
    ax.set_xlabel("Junction")
    ax.set_ylabel("Mean reward")
    ax.tick_params(axis="x", rotation=55)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_per_junction_heatmap(per_junction: pd.DataFrame, title: str, field: str):
    df = per_junction.copy()
    if "junction_id" not in df.columns or field not in df.columns:
        raise ValueError(f"per_junction_results.csv needs junction_id and {field}")
    positions = [parse_grid_position(str(jid), i, len(df)) for i, jid in enumerate(df["junction_id"])]
    max_r = max(r for r, _ in positions)
    max_c = max(c for _, c in positions)
    mat = np.full((max_r + 1, max_c + 1), np.nan)
    for (_, row), (r, c) in zip(df.iterrows(), positions):
        mat[r, c] = pd.to_numeric(row[field], errors="coerce")
    fig, ax = plt.subplots(figsize=(max(7, min(18, mat.shape[1] * 0.6)), max(4, min(10, mat.shape[0] * 0.6))))
    cmap = "RdYlGn" if field == "mean_reward" else "YlOrRd"
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{field} Heatmap - {title}")
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    fig.tight_layout()
    return fig


def plot_dashboard(
    steps: pd.DataFrame | None,
    summary: dict | None,
    per_junction: pd.DataFrame | None,
    training: pd.DataFrame | None,
    title: str,
):
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.34)
    fig.suptitle(f"SafeGAT Run Dashboard - {title}", fontsize=15, y=0.99)

    ax = fig.add_subplot(gs[0, 0])
    if steps is not None and "mean_reward" in steps:
        reward = pd.to_numeric(steps["mean_reward"], errors="coerce").fillna(0).to_numpy()
        ax.plot(steps["step"], smooth(reward, 20), color=COLORS["reward"])
        ax.axhline(0, color=COLORS["gray"], lw=0.8, ls="--")
        ax.set_title("Mean reward")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
    else:
        ax.axis("off")
    ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(gs[0, 1])
    if steps is not None and has_cols(steps, "mean_occ", "mean_margin"):
        ax2 = ax.twinx()
        ax.plot(steps["step"], smooth(steps["mean_occ"], 20), color=COLORS["occ"], label="occ")
        ax2.plot(steps["step"], smooth(steps["mean_margin"], 20), color=COLORS["margin"], ls="--", label="margin")
        ax.set_title("Occupancy and margin")
        ax.set_xlabel("Step")
        ax.set_ylabel("Occupancy", color=COLORS["occ"])
        ax2.set_ylabel("Margin", color=COLORS["margin"])
    else:
        ax.axis("off")
    ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(gs[0, 2])
    if summary:
        labels = ["calls", "overrides", "safety", "failures"]
        values = [
            summary.get("llm_calls", 0),
            summary.get("llm_overrides", 0),
            summary.get("safety_adjustments", 0),
            summary.get("llm_failures", 0),
        ]
        ax.bar(labels, values, color=[COLORS["llm"], COLORS["override"], COLORS["shield"], COLORS["gray"]])
        ax.set_title("Interventions")
        ax.set_ylabel("Count")
    else:
        ax.axis("off")
    ax.grid(True, axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[1, 0])
    if steps is not None and "llm_calls" in steps:
        ax.plot(steps["step"], pd.to_numeric(steps["llm_calls"], errors="coerce").fillna(0), color=COLORS["llm"])
        ax.set_title("Cumulative LLM calls")
        ax.set_xlabel("Step")
        ax.set_ylabel("Calls")
    else:
        ax.axis("off")
    ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(gs[1, 1])
    if per_junction is not None and "mean_reward" in per_junction:
        rewards = pd.to_numeric(per_junction["mean_reward"], errors="coerce").dropna()
        ax.hist(rewards, bins=min(24, max(6, len(rewards) // 2)), color=COLORS["reward"], edgecolor="white")
        ax.set_title("Junction reward distribution")
        ax.set_xlabel("Mean reward")
        ax.set_ylabel("Junctions")
    else:
        ax.axis("off")
    ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(gs[1, 2])
    if training is not None:
        ep = training["episode"] if "episode" in training else np.arange(1, len(training) + 1)
        col = "total_reward" if "total_reward" in training else "mean_reward"
        if col in training:
            rewards = pd.to_numeric(training[col], errors="coerce").fillna(0)
            ax.plot(ep, smooth(rewards, 7), color=COLORS["reward"])
            ax.set_title("Training reward")
            ax.set_xlabel("Episode")
            ax.set_ylabel(col)
        else:
            ax.axis("off")
    else:
        ax.axis("off")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig


def discover_runs(out_root: Path) -> list[Run]:
    runs: dict[Path, Run] = {}

    fixed = [
        ("4x4/output", ROOT / "4x4" / "output", "base"),
        ("4x4/data/output", ROOT / "4x4" / "data" / "output", "base"),
        ("7x28/output", ROOT / "7x28" / "output", "base"),
        ("7x28/data/output", ROOT / "7x28" / "data" / "output", "base"),
    ]
    for label, source, family in fixed:
        if (source / "step_log.json").exists():
            runs[source.resolve()] = Run(label, source, output_path_for(out_root, label), family)

    for dataset in ["Jinan", "Hangzhou", "NewYork"]:
        base = ROOT / dataset
        if not base.exists():
            continue
        for step_log in base.rglob("safegat_cityflow_live_output/step_log.json"):
            source = step_log.parent
            label = str(source.relative_to(ROOT))
            runs[source.resolve()] = Run(label, source, output_path_for(out_root, label), "cityflow")

    llm_root = ROOT / "LLM_outputs"
    if llm_root.exists():
        for step_log in llm_root.rglob("step_log.json"):
            source = step_log.parent
            rel = source.relative_to(ROOT)
            parts = rel.parts
            model = parts[2] if len(parts) >= 3 else None
            runs[source.resolve()] = Run(str(rel), source, output_path_for(out_root, str(rel)), "llm", model)

    return sorted(runs.values(), key=lambda run: run.label.lower())


def find_training_curve(run: Run) -> Path | None:
    candidates = [
        run.source_dir / "training_curve.json",
        run.source_dir.parent / "safegat_cityflow_live_training" / "training_curve.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def run_metrics(run: Run, steps: pd.DataFrame | None, summary: dict | None) -> dict:
    metrics = {
        "label": run.label,
        "source_dir": str(run.source_dir.relative_to(ROOT)),
        "output_dir": str(run.output_dir.relative_to(ROOT)),
        "family": run.family,
        "model": run.model or "",
    }
    if steps is not None and not steps.empty:
        metrics["steps"] = len(steps)
        if "mean_reward" in steps:
            metrics["step_mean_reward"] = float(pd.to_numeric(steps["mean_reward"], errors="coerce").mean())
        if "llm_calls" in steps:
            metrics["llm_calls_from_steps_last"] = int(pd.to_numeric(steps["llm_calls"], errors="coerce").fillna(0).iloc[-1])
    if summary:
        for key in [
            "total_sim_steps",
            "llm_calls",
            "llm_failures",
            "llm_overrides",
            "safety_adjustments",
            "override_rate_%",
            "total_reward",
            "mean_reward",
            "wall_clock_s",
        ]:
            if key in summary:
                metrics[key] = summary[key]
    return metrics


def generate_run_plots(run: Run, formats: list[str]) -> dict:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run.label}")
    steps = load_step_log(run.source_dir / "step_log.json")
    summary_path = run.source_dir / "intervention_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else None
    per_path = run.source_dir / "per_junction_results.csv"
    per_junction = load_per_junction(per_path) if per_path.exists() else None
    training_path = find_training_curve(run)
    training = pd.DataFrame(read_json(training_path)) if training_path else None
    title = run.label.replace("\\", "/")

    created = []
    if "mean_reward" in steps:
        created += save(plot_reward_curve(steps, title), run.output_dir, "1_reward_curve", formats)
    if has_cols(steps, "mean_occ", "mean_margin"):
        created += save(plot_occupancy_margin(steps, title), run.output_dir, "2_occupancy_margin", formats)
    if "llm_calls" in steps:
        created += save(plot_llm_budget(steps, title), run.output_dir, "3_llm_budget", formats)

    trip_path = run.source_dir / "safegat.tripinfo.xml"
    if trip_path.exists():
        trips = load_tripinfo(trip_path)
        if "duration" in trips:
            created += save(plot_tripinfo(trips, title, "duration", "Trip duration"), run.output_dir, "4_att_distribution", formats)
        if "waitingTime" in trips:
            created += save(plot_tripinfo(trips, title, "waitingTime", "Waiting time"), run.output_dir, "5_waiting_distribution", formats)
        if "timeLoss" in trips:
            created += save(plot_tripinfo(trips, title, "timeLoss", "Time loss"), run.output_dir, "6_timeloss_distribution", formats)

    if summary:
        created += save(plot_intervention_summary(summary, title), run.output_dir, "7_intervention_summary", formats)
        created += save(plot_intervention_pie(summary, title), run.output_dir, "8_intervention_pie", formats)

    if training is not None and not training.empty:
        created += save(plot_training_curve(training, title), run.output_dir, "9_training_curve", formats)

    if per_junction is not None and not per_junction.empty:
        try:
            created += save(plot_per_junction_bars(per_junction, title), run.output_dir, "10_per_junction_rewards", formats)
            created += save(plot_per_junction_heatmap(per_junction, title, "mean_reward"), run.output_dir, "11_reward_heatmap", formats)
            if "llm_calls" in per_junction.columns:
                created += save(plot_per_junction_heatmap(per_junction, title, "llm_calls"), run.output_dir, "12_llm_calls_heatmap", formats)
        except Exception as exc:
            print(f"  [warn] skipped per-junction plots: {exc}")

    created += save(plot_dashboard(steps, summary, per_junction, training, title), run.output_dir, "13_combined_dashboard", formats)

    metrics = run_metrics(run, steps, summary)
    metrics["created_files"] = len(created)
    print(f"  saved {len(created)} files -> {run.output_dir.relative_to(ROOT)}")
    return metrics


def generate_model_comparisons(runs: list[Run], out_root: Path, formats: list[str]) -> None:
    grouped: dict[str, list[Run]] = {}
    for run in runs:
        if run.family != "llm":
            continue
        rel = run.source_dir.relative_to(ROOT)
        parts = rel.parts
        if len(parts) < 3:
            continue
        group = str(Path(parts[0]) / parts[1])
        grouped.setdefault(group, []).append(run)

    for group, group_runs in sorted(grouped.items()):
        rows = []
        for run in sorted(group_runs, key=lambda r: r.model or ""):
            summary_path = run.source_dir / "intervention_summary.json"
            steps_path = run.source_dir / "step_log.json"
            if not summary_path.exists() and not steps_path.exists():
                continue
            summary = read_json(summary_path) if summary_path.exists() else {}
            steps = load_step_log(steps_path) if steps_path.exists() else pd.DataFrame()
            rows.append(run_metrics(run, steps, summary))
        if len(rows) < 2:
            continue

        df = pd.DataFrame(rows)
        models = [str(m) for m in df["model"]]
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        specs = [
            ("mean_reward", "Mean reward", COLORS["reward"]),
            ("total_reward", "Total reward", COLORS["occ"]),
            ("llm_calls", "LLM calls", COLORS["llm"]),
            ("override_rate_%", "Override rate (%)", COLORS["override"]),
        ]
        for ax, (col, label, color) in zip(axes.ravel(), specs):
            if col not in df.columns:
                ax.axis("off")
                continue
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
            ax.bar(models, vals, color=color, width=0.6)
            ax.set_title(label)
            ax.tick_params(axis="x", rotation=40)
            ax.grid(True, axis="y", alpha=0.3)
        fig.suptitle(f"LLM Model Comparison - {group}", fontsize=15, y=1.01)
        fig.tight_layout()
        out_dir = output_path_for(out_root, str(Path(group) / "_model_comparison"))
        save(fig, out_dir, "llm_model_comparison", formats)

        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "llm_model_comparison.csv", index=False)
        print(f"[compare] {group} -> {out_dir.relative_to(ROOT)}")


def write_summary_csv(rows: list[dict], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "plot_manifest.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[manifest] {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replicate SafeGAT plots into a separate output folder.")
    parser.add_argument("--out-dir", default="replicated_plot_outputs", help="Separate folder for replicated plots.")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated output formats, e.g. png or png,pdf.")
    args = parser.parse_args()

    out_root = (ROOT / args.out_dir).resolve()
    formats = [fmt.strip().lower().lstrip(".") for fmt in args.formats.split(",") if fmt.strip()]
    if not formats:
        raise SystemExit("At least one format is required.")

    runs = discover_runs(out_root)
    if not runs:
        raise SystemExit("No result folders with step_log.json were found.")

    print(f"Discovered {len(runs)} result folders.")
    print(f"Writing replicated plots to: {out_root}")

    rows = []
    for run in runs:
        try:
            rows.append(generate_run_plots(run, formats))
        except Exception as exc:
            print(f"[error] {run.label}: {exc}")
            rows.append(
                {
                    "label": run.label,
                    "source_dir": str(run.source_dir.relative_to(ROOT)),
                    "output_dir": str(run.output_dir.relative_to(ROOT)),
                    "family": run.family,
                    "model": run.model or "",
                    "error": str(exc),
                }
            )

    generate_model_comparisons(runs, out_root, formats)
    write_summary_csv(rows, out_root)
    print("Done.")


if __name__ == "__main__":
    main()
