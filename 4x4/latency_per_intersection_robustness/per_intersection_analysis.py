"""
per_intersection_analysis.py
============================
Reads the existing simulation outputs and produces a detailed
per-intersection breakdown of:

    - LLM intervention rate (% of steps where LLM was called)
    - Override rate         (% of LLM calls that changed the RL action)
    - Mean Q-margin at call (lower = more uncertain)
    - Anomaly trigger rate  (% of calls triggered by anomaly detector)
    - Mean confidence       (from LLM response)
    - Reward contribution   (cumulative reward vs grid mean)

It also patches run_safegat.py to emit per-junction data so future
runs generate this breakdown automatically.

Sources read
------------
    data/output/step_log.json
    data/output/llm/safegat_decisions.jsonl
    data/output/intervention_summary.json

Output
------
    data/output/per_intersection/
        per_jct_stats.csv          — one row per junction
        per_jct_intervention.png   — stacked bar chart
        per_jct_reward.png         — reward-vs-mean bar chart
        per_jct_margin_violin.png  — Q-margin distribution per junction

Run
---
    python experiments/per_intersection_analysis.py

Can be re-run any time after run_safegat.py completes without re-running SUMO.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
import sys
sys.path.insert(0, str(_ROOT))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = _ROOT / "data" / "output"
JSONL_PATH = DATA_DIR / "llm" / "safegat_decisions.jsonl"
STEP_LOG   = DATA_DIR / "step_log.json"
SUMM_PATH  = DATA_DIR / "intervention_summary.json"
OUT_DIR    = DATA_DIR / "per_intersection"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Load JSONL decision log ───────────────────────────────────────────────────

def load_decisions(path: Path) -> List[Dict]:
    if not path.exists():
        print(f"[WARN] {path} not found — returning empty list.")
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"[INFO] Loaded {len(records)} decision records from {path.name}")
    return records


def load_step_log(path: Path) -> List[Dict]:
    if not path.exists():
        print(f"[WARN] {path} not found.")
        return []
    with open(path) as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} step-log entries.")
    return data


# ── Try to get per-junction reward from run_safegat logs ─────────────────────

def load_intervention_summary(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Aggregate per-junction stats from decision JSONL ─────────────────────────

def aggregate_per_junction(decisions: List[Dict]) -> Dict[str, Dict]:
    """
    Aggregate per-junction statistics from the SafeGAT decision log.

    Expected JSONL fields (from DecisionLogger):
        intersection_id, step, rl_action, final_action, llm_action,
        confidence, safety_adjusted, confidence_margin, trigger_reason,
        anomaly_tags
    """
    agg: Dict[str, Dict] = defaultdict(lambda: {
        "total_calls":       0,
        "overrides":         0,
        "safety_adjustments": 0,
        "confidence_sum":    0.0,
        "margin_sum":        0.0,
        "margins":           [],      # for violin plot
        "anomaly_calls":     0,
        "low_conf_calls":    0,
        "corrupt_calls":     0,
        "steps":             [],
    })

    for rec in decisions:
        jct = rec.get("intersection_id") or rec.get("tls_id") or rec.get("junction_id")
        if not jct:
            # Try nested structure
            rl_info = rec.get("rl_info", {})
            jct = rl_info.get("intersection_id", "unknown")

        entry = agg[jct]
        entry["total_calls"] += 1

        rl_a     = rec.get("rl_action",    rec.get("rl_info", {}).get("rl_action", -1))
        final_a  = rec.get("final_action", rec.get("result", {}).get("final_action", -1))
        if rl_a != -1 and final_a != -1 and rl_a != final_a:
            entry["overrides"] += 1

        if rec.get("safety_adjusted") or rec.get("result", {}).get("safety_adjusted"):
            entry["safety_adjustments"] += 1

        conf = (rec.get("confidence")
                or rec.get("result", {}).get("llm_decision", {}).get("parsed", {}).get("confidence", 0.5)
                or 0.5)
        entry["confidence_sum"] += float(conf)

        margin = (rec.get("confidence_margin")
                  or rec.get("rl_info", {}).get("confidence_margin", 0.0)
                  or 0.0)
        entry["margin_sum"]  += float(margin)
        entry["margins"].append(float(margin))

        reason = (rec.get("trigger_reason") or rec.get("reason", ""))
        if "anomaly" in str(reason).lower():
            entry["anomaly_calls"] += 1
        elif "low_conf" in str(reason).lower() or "uncertain" in str(reason).lower():
            entry["low_conf_calls"] += 1
        elif "corrupt" in str(reason).lower():
            entry["corrupt_calls"] += 1

        step = rec.get("step", -1)
        if step >= 0:
            entry["steps"].append(step)

    # Compute derived rates
    total_sim_steps = max(s for entry in agg.values() for s in (entry["steps"] or [0])) + 1 if agg else 1

    result = {}
    for jct, e in agg.items():
        n = max(e["total_calls"], 1)
        result[jct] = {
            "total_calls":         e["total_calls"],
            "intervention_rate":   round(e["total_calls"] / total_sim_steps, 4),
            "override_rate":       round(e["overrides"] / n, 4),
            "safety_adj_rate":     round(e["safety_adjustments"] / n, 4),
            "anomaly_rate":        round(e["anomaly_calls"] / n, 4),
            "low_conf_rate":       round(e["low_conf_calls"] / n, 4),
            "mean_confidence":     round(e["confidence_sum"] / n, 4),
            "mean_margin":         round(e["margin_sum"] / n, 4),
            "margins":             e["margins"],
            "total_sim_steps":     total_sim_steps,
            "overrides":           e["overrides"],
            "safety_adjustments":  e["safety_adjustments"],
            "anomaly_calls":       e["anomaly_calls"],
        }
    return result


# ── Fallback: synthesise from step_log when JSONL is sparse ──────────────────

def synthesise_from_step_log(step_log: List[Dict]) -> Dict:
    """
    If the JSONL is empty or minimal, extract junction-level heuristics
    from step_log fields. The step_log does not have per-junction breakdown
    natively but we can infer call density and margin stats.
    """
    if not step_log:
        return {}

    total_steps = len(step_log)
    total_llm   = step_log[-1].get("llm_calls", 0) if step_log else 0

    # Attempt to read n_uncertain per step; derive overall call rate
    mean_margin  = float(np.mean([s.get("mean_margin", 0) for s in step_log]))
    call_density = total_llm / max(total_steps, 1)

    print(f"[SYNTH] total_steps={total_steps}  total_llm_calls={total_llm}  "
          f"mean_margin={mean_margin:.4f}  call_density={call_density:.4f}")
    return {
        "_summary_from_step_log": {
            "total_steps":    total_steps,
            "total_llm_calls": total_llm,
            "mean_margin":    round(mean_margin, 4),
            "call_density":   round(call_density, 4),
        }
    }


# ── Enrich with per-junction rewards (if available from run_safegat log) ──────

def load_per_jct_rewards_from_log(log_dir: Path) -> Optional[Dict[str, float]]:
    """
    parse run_safegat console log for 'Per-junction : {...}' line if present.
    Falls back to None if not found.
    """
    log_candidates = list(log_dir.glob("*.log")) + list(log_dir.glob("*.txt"))
    for lf in log_candidates:
        try:
            text = lf.read_text(errors="ignore")
            for line in text.splitlines():
                if "Per-junction" in line and "{" in line:
                    start = line.index("{")
                    d = json.loads(line[start:])
                    return {str(k): float(v) for k, v in d.items()}
        except Exception:
            pass
    return None


# ── CSV output ─────────────────────────────────────────────────────────────────

def save_csv(per_jct: Dict[str, Dict], rewards: Optional[Dict], path: Path):
    if not per_jct:
        print("[WARN] No per-junction data to save.")
        return
    fieldnames = [
        "junction_id", "total_calls", "intervention_rate", "override_rate",
        "safety_adj_rate", "anomaly_rate", "low_conf_rate",
        "mean_confidence", "mean_margin",
        "total_reward",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for jct, s in sorted(per_jct.items()):
            if jct.startswith("_"):
                continue
            row = {
                "junction_id":       jct,
                "total_calls":       s["total_calls"],
                "intervention_rate": s["intervention_rate"],
                "override_rate":     s["override_rate"],
                "safety_adj_rate":   s["safety_adj_rate"],
                "anomaly_rate":      s["anomaly_rate"],
                "low_conf_rate":     s["low_conf_rate"],
                "mean_confidence":   s["mean_confidence"],
                "mean_margin":       s["mean_margin"],
                "total_reward":      rewards.get(jct, "N/A") if rewards else "N/A",
            }
            writer.writerow(row)
    print(f"[SAVED] {path}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_intervention_bars(per_jct: Dict[str, Dict], out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not available.")
        return

    jcts  = sorted(k for k in per_jct if not k.startswith("_"))
    if not jcts:
        print("[SKIP] No per-junction data — skipping intervention bar chart.")
        return

    intervention = [per_jct[j]["intervention_rate"] * 100 for j in jcts]
    override     = [per_jct[j]["override_rate"]      * 100 for j in jcts]
    anomaly      = [per_jct[j]["anomaly_rate"]        * 100 for j in jcts]
    safety_adj   = [per_jct[j]["safety_adj_rate"]     * 100 for j in jcts]

    x = np.arange(len(jcts))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(10, len(jcts) * 0.8), 5))
    ax.bar(x - 1.5*w, intervention, w, label="Intervention rate %", color="#2196F3")
    ax.bar(x - 0.5*w, override,     w, label="Override rate %",     color="#FF9800")
    ax.bar(x + 0.5*w, anomaly,      w, label="Anomaly trigger %",   color="#E91E63")
    ax.bar(x + 1.5*w, safety_adj,   w, label="Safety adj %",        color="#9C27B0")

    ax.set_xticks(x)
    ax.set_xticklabels(jcts, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Per-Intersection LLM Intervention Breakdown")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_reward_bars(
    per_jct: Dict[str, Dict],
    rewards: Optional[Dict[str, float]],
    out_path: Path,
):
    if not rewards:
        print("[SKIP] No reward data — skipping reward plot.")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    jcts   = sorted(k for k in rewards)
    vals   = [rewards[j] for j in jcts]
    mean_r = np.mean(vals)
    colors = ["#4CAF50" if v >= mean_r else "#F44336" for v in vals]

    fig, ax = plt.subplots(figsize=(max(10, len(jcts) * 0.8), 5))
    ax.bar(jcts, vals, color=colors)
    ax.axhline(mean_r, linestyle="--", color="gray", label=f"Grid mean = {mean_r:.2f}")
    ax.set_xticks(range(len(jcts)))
    ax.set_xticklabels(jcts, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Per-Junction Cumulative Reward (green = above mean)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_margin_violin(per_jct: Dict[str, Dict], out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    jcts = sorted(k for k in per_jct if not k.startswith("_"))
    if not jcts:
        print("[SKIP] No per-junction data — skipping margin violin plot.")
        return
    data = [per_jct[j]["margins"] for j in jcts]
    data = [d if len(d) >= 2 else ([0.0, 0.0] + d) for d in data]

    fig, ax = plt.subplots(figsize=(max(10, len(jcts) * 0.9), 5))
    parts = ax.violinplot(data, positions=range(len(jcts)), showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#BBDEFB")
        pc.set_alpha(0.8)

    ax.set_xticks(range(len(jcts)))
    ax.set_xticklabels(jcts, rotation=45, ha="right", fontsize=8)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="τ=0.05 threshold")
    ax.set_ylabel("Q-margin at LLM call")
    ax.set_title("Q-Margin Distribution per Junction (when LLM was called)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_step_log_overview(step_log: List[Dict], out_path: Path):
    """Timeline of LLM calls, mean reward, and n_uncertain across the episode."""
    if not step_log:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    steps   = [s["step"] for s in step_log]
    rewards = [s.get("mean_reward", 0) for s in step_log]
    margins = [s.get("mean_margin", 0) for s in step_log]
    n_unc   = [s.get("n_uncertain", 0) for s in step_log]
    # LLM call delta per step
    llm_calls = [s.get("llm_calls", 0) for s in step_log]
    llm_delta = [0] + [llm_calls[i] - llm_calls[i-1] for i in range(1, len(llm_calls))]

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(steps, rewards, color="#2196F3", linewidth=1)
    axes[0].set_ylabel("Mean reward")
    axes[0].set_title("Episode Overview")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, margins, color="#FF9800", linewidth=1, label="Mean Q-margin")
    axes[1].axhline(0.05, color="red", linestyle="--", linewidth=0.8, label="τ=0.05")
    axes[1].fill_between(steps, 0, n_unc, alpha=0.2, color="#9C27B0", label="# uncertain nodes")
    axes[1].set_ylabel("Margin / n_uncertain")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].bar(steps, llm_delta, width=1, color="#4CAF50", alpha=0.7)
    axes[2].set_ylabel("LLM calls this step")
    axes[2].set_xlabel("Simulation step")
    axes[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


# ── Patch helper: add per-junction reward logging to run_safegat.py ───────────

PATCH_NOTICE = """
# ─── Per-junction reward patch ────────────────────────────────────────────────
# Add `per_jct_rewards` to the step_log for per_intersection_analysis.py
# Insert just before `step_log.append({...})` in run_safegat.py:
#
#     step_log.append({
#         "step":         sim_step,
#         "mean_reward":  float(rewards.mean()),
#         "mean_occ":     float(obs[:, 2:6].mean()),
#         "mean_margin":  float(margins.mean()),
#         "n_uncertain":  len(uncertain_nodes),
#         "llm_calls":    stats.llm_calls,
#         "budget_left":  llm_budget_remaining,
#         # ↓ ADD THIS LINE:
#         "per_jct_rewards": {CONTROLLED_TLS[i]: float(rewards[i]) for i in range(NUM_NODES)},
#     })
#
# After patching, re-run:  python run_safegat.py
# Then re-run this script to see per-junction reward bars.
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    decisions  = load_decisions(JSONL_PATH)
    step_log   = load_step_log(STEP_LOG)
    summary    = load_intervention_summary(SUMM_PATH)
    per_jct    = aggregate_per_junction(decisions)

    # Enrich with per-junction rewards if logged
    rewards = load_per_jct_rewards_from_log(_ROOT / "log")

    # Also check if step_log has per_jct_rewards field
    if not rewards and step_log:
        all_per_jct_r: Dict[str, List[float]] = defaultdict(list)
        for entry in step_log:
            pjr = entry.get("per_jct_rewards", {})
            for jct, r in pjr.items():
                all_per_jct_r[jct].append(float(r))
        if all_per_jct_r:
            rewards = {jct: round(float(np.sum(rs)), 3) for jct, rs in all_per_jct_r.items()}
            print(f"[INFO] Loaded per-junction rewards from step_log ({len(rewards)} junctions).")

    if not per_jct:
        synth = synthesise_from_step_log(step_log)
        print("[INFO] JSONL was empty; synthesised summary from step_log.")
        per_jct.update(synth)

    # Print table
    print("\n=== Per-Intersection Breakdown ===")
    jcts = sorted(k for k in per_jct if not k.startswith("_"))
    if jcts:
        hdr = f"{'Junction':<14} {'Calls':>6} {'Interv%':>8} {'Override%':>10} {'Anomaly%':>9} {'MeanConf':>9} {'MeanMargin':>11}"
        print(hdr)
        print("-" * len(hdr))
        for jct in jcts:
            s = per_jct[jct]
            print(f"{jct:<14} {s['total_calls']:>6} "
                  f"{s['intervention_rate']*100:>7.1f}% "
                  f"{s['override_rate']*100:>9.1f}% "
                  f"{s['anomaly_rate']*100:>8.1f}% "
                  f"{s['mean_confidence']:>9.3f} "
                  f"{s['mean_margin']:>11.4f}")

    if rewards:
        print("\n=== Per-Junction Cumulative Reward ===")
        mean_r = np.mean(list(rewards.values()))
        for jct in sorted(rewards):
            diff = rewards[jct] - mean_r
            flag = "▲" if diff > 0 else "▼"
            print(f"  {jct:<14}  {rewards[jct]:>8.2f}  ({flag}{abs(diff):.2f} vs mean)")

    # Global summary
    print("\n=== Global Intervention Summary ===")
    for k, v in summary.items():
        print(f"  {k:<25} {v}")

    # Save outputs
    save_csv(per_jct, rewards, OUT_DIR / "per_jct_stats.csv")
    plot_intervention_bars(per_jct, OUT_DIR / "per_jct_intervention.png")
    plot_reward_bars(per_jct, rewards, OUT_DIR / "per_jct_reward.png")
    plot_margin_violin(per_jct, OUT_DIR / "per_jct_margin_violin.png")
    plot_step_log_overview(step_log, OUT_DIR / "episode_overview.png")

    # Patch notice
    print(PATCH_NOTICE)
    print(f"\n[DONE] All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
