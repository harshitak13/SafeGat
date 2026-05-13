"""
latency_analysis.py — LLM Latency Analysis for 7x28 SafeGAT-iLLM
==================================================================
Measures and models the real-world latency introduced by LLM calls in
SafeGAT for the 196-junction 7x28 network.

Analyses
--------
1. Empirical timing  — records wall-clock time per LLM call using the same
   LLMGateway / backend stack used in production.
2. Per-step budget   — compares RL-only step time vs SafeGAT step time across
   N mock steps, showing overhead distribution.
3. Per-junction breakdown — latency contribution broken down across all 196 nodes.
4. Deployment viability  — sweeps (call_rate, latency) space and marks the
   safe operating region for the 7x28 network.

Output
------
    output/latency/
        raw_call_latencies.json
        step_timing_log.json
        latency_summary.csv
        per_junction_latency.csv
        latency_distribution.png
        deployment_viability.png

Run
---
    python latency_per_intersection_robustness/latency_analysis.py
"""

from __future__ import annotations

import json
import os
import time
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
import sys
sys.path.insert(0, str(_ROOT))

from llm.llm_gateway  import LLMGateway
from utils.readConfig import read_config
from network.net_config import CONTROLLED_TLS, NUM_NODES

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = _ROOT / "output" / "latency"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Experiment parameters — scaled for 196 nodes ──────────────────────────────
N_LLM_CALLS        = 60       # real LLM calls to time
N_MOCK_STEPS       = 200      # steps for overhead measurement
STEP_DURATION_S    = 1.0      # SUMO step = 1 s real-time (headless)
# Deployment viability sweep
CALL_RATES         = [0.01, 0.02, 0.04, 0.08, 0.16, 0.3, 0.5, 1.0]  # calls/step
LATENCY_TARGETS_MS = [100, 200, 500, 1000, 2000, 5000]

# How many nodes can be reviewed per step (matches run_safegat.py)
MAX_NODES_PER_STEP = 8

# Sample prompt representative of a 7x28 junction query
_SAMPLE_PROMPT = """\
You are a traffic signal controller AI for a large 196-junction urban network.

Junction: J_47  |  Phase: 2  |  Phase elapsed: 18 s
Observation: occ=[0.78, 0.45, 0.22, 0.61]  queue=0.69  emergency=0
RL proposed action: 1  |  Q-margin: 0.031 (low confidence)
Neighbour actions: J_46->0, J_48->2, J_19->1, J_75->3

Choose the best traffic-signal phase (0-3) for this junction.
Respond ONLY with valid JSON, no other text:
{"action": <int 0-3>, "confidence": <float 0-1>, "reasoning": "<one sentence>"}
"""


def _build_backend(config: dict):
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    api_key  = config["OPENAI_API_KEY"]
    model    = config["OPENAI_API_MODEL"]
    base_url = config.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    chat = ChatOpenAI(
        model=model, temperature=0.0,
        openai_api_key=api_key, openai_api_base=base_url, request_timeout=30,
    )
    def _backend(prompt: str) -> str:
        return chat.invoke([HumanMessage(content=prompt)]).content
    return _backend


# ── 1. Empirical call timing ───────────────────────────────────────────────────

def measure_llm_latency(gateway: LLMGateway, n_calls: int) -> List[Dict]:
    print(f"\n[1] Measuring LLM latency: {n_calls} real calls ...")
    records = []
    for i in range(n_calls):
        t0 = time.perf_counter()
        try:
            decision = gateway.query(_SAMPLE_PROMPT, label=f"J_latency_{i}")
            success  = True
        except Exception as e:
            success = False
            print(f"  Call {i} failed: {e}")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        record = {
            "call_idx":    i,
            "latency_ms":  round(elapsed_ms, 2),
            "success":     success,
        }
        records.append(record)
        if (i + 1) % 10 == 0:
            recent = [r["latency_ms"] for r in records[-10:] if r["success"]]
            print(f"  calls={i+1:>3}  |  last-10 mean={np.mean(recent):.0f} ms  "
                  f"|  last-10 max={max(recent):.0f} ms")
    return records


# ── 2. Per-step overhead measurement ─────────────────────────────────────────

def measure_step_overhead(gateway: LLMGateway, n_steps: int) -> List[Dict]:
    print(f"\n[2] Measuring per-step overhead: {n_steps} mock steps ...")
    step_records = []
    for step in range(n_steps):
        t_rl = time.perf_counter()
        # Simulate RL action selection (cheap numpy ops)
        fake_obs = np.random.rand(NUM_NODES, 8).astype(np.float32)
        fake_q   = np.random.rand(NUM_NODES, 4).astype(np.float32)
        _actions = fake_q.argmax(axis=1)
        t_rl_done = time.perf_counter()

        # Simulate selective LLM calls (only on ~5% of nodes, up to MAX_NODES_PER_STEP)
        t_llm_start = time.perf_counter()
        n_flagged = min(
            int(NUM_NODES * 0.05),   # ~5% flagged
            MAX_NODES_PER_STEP,
        )
        llm_calls_this_step = 0
        if step % 20 == 0 and n_flagged > 0:  # Only actually call LLM every 20 steps
            try:
                gateway.query(_SAMPLE_PROMPT, label=f"step_{step}_node_0")
                llm_calls_this_step = 1
            except Exception:
                pass
        t_llm_done = time.perf_counter()

        step_records.append({
            "step":              step,
            "rl_ms":             round((t_rl_done - t_rl) * 1000, 3),
            "llm_ms":            round((t_llm_done - t_llm_start) * 1000, 3),
            "total_ms":          round((t_llm_done - t_rl) * 1000, 3),
            "llm_calls":         llm_calls_this_step,
            "nodes_flagged":     n_flagged,
        })

        if step % 50 == 0:
            print(f"  step={step:>3}  |  rl={step_records[-1]['rl_ms']:.1f}ms  "
                  f"|  llm={step_records[-1]['llm_ms']:.1f}ms")

    return step_records


# ── 3. Per-junction latency breakdown ─────────────────────────────────────────

def compute_per_junction_latency(call_records: List[Dict]) -> List[Dict]:
    """
    Distribute measured latencies across all 196 junctions to show expected
    per-junction overhead based on call frequency. Uses uniform allocation
    since latency measurements come from the same gateway (not junction-specific).
    """
    successful = [r for r in call_records if r["success"]]
    if not successful:
        return []

    mean_lat = np.mean([r["latency_ms"] for r in successful])
    std_lat  = np.std([r["latency_ms"] for r in successful])

    # Simulate call frequency: ~5% of nodes flagged each step on average
    # for a 3600-step episode with MAX_NODES_PER_STEP=8
    est_calls_per_jct = (3600 * MAX_NODES_PER_STEP * 0.05) / NUM_NODES

    rows = []
    for tls_id in CONTROLLED_TLS:
        rows.append({
            "junction_id":           tls_id,
            "est_calls_per_episode": round(est_calls_per_jct, 2),
            "mean_latency_ms":       round(mean_lat, 2),
            "std_latency_ms":        round(std_lat, 2),
            "est_total_lat_ms":      round(est_calls_per_jct * mean_lat, 1),
        })
    return rows


# ── 4. Deployment viability heatmap ───────────────────────────────────────────

def compute_deployment_viability() -> Dict:
    """
    For each (call_rate_per_step, mean_latency_ms) pair, compute the
    expected added delay per simulation step for the 196-junction network.
    A (call_rate, latency) combo is viable if added_delay < STEP_DURATION_S * 1000.
    """
    results = {}
    for call_rate in CALL_RATES:
        for lat_ms in LATENCY_TARGETS_MS:
            # Expected LLM delay per step = calls_per_step * mean_latency
            calls_per_step = min(call_rate * NUM_NODES, MAX_NODES_PER_STEP)
            added_delay_ms = calls_per_step * lat_ms
            viable = added_delay_ms < STEP_DURATION_S * 1000
            results[f"{call_rate:.3f}_{lat_ms}"] = {
                "call_rate":       call_rate,
                "latency_ms":      lat_ms,
                "calls_per_step":  round(calls_per_step, 2),
                "added_delay_ms":  round(added_delay_ms, 1),
                "viable":          viable,
            }
    return results


# ── 5. Plotting ────────────────────────────────────────────────────────────────

def plot_latency_distribution(call_records: List[Dict], out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        lats = [r["latency_ms"] for r in call_records if r["success"]]
        if not lats:
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # CDF
        sorted_lats = np.sort(lats)
        cdf = np.arange(1, len(sorted_lats) + 1) / len(sorted_lats)
        ax1.plot(sorted_lats, cdf, color="steelblue", lw=2)
        ax1.axvline(np.median(lats), color="red", ls="--", label=f"Median={np.median(lats):.0f}ms")
        ax1.axvline(np.percentile(lats, 95), color="orange", ls="--",
                    label=f"P95={np.percentile(lats, 95):.0f}ms")
        ax1.set_xlabel("Latency (ms)")
        ax1.set_ylabel("CDF")
        ax1.set_title("LLM Call Latency CDF (7x28)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Boxplot
        ax2.boxplot(lats, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.6))
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title(f"Latency Distribution\n(n={len(lats)}, mean={np.mean(lats):.0f}ms)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")
    except ImportError:
        print("  matplotlib not available — skipping plot")


def plot_deployment_viability(viability: Dict, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        unique_rates = sorted(set(v["call_rate"] for v in viability.values()))
        unique_lats  = sorted(set(v["latency_ms"] for v in viability.values()))

        Z = np.zeros((len(unique_lats), len(unique_rates)))
        for v in viability.values():
            ri = unique_rates.index(v["call_rate"])
            li = unique_lats.index(v["latency_ms"])
            Z[li, ri] = v["added_delay_ms"]

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(Z, aspect="auto", origin="lower",
                       cmap="RdYlGn_r", interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Added delay per step (ms)")
        ax.set_xticks(range(len(unique_rates)))
        ax.set_xticklabels([f"{r:.2f}" for r in unique_rates], rotation=45)
        ax.set_yticks(range(len(unique_lats)))
        ax.set_yticklabels([f"{l}" for l in unique_lats])
        ax.set_xlabel("LLM call rate (calls per node per step)")
        ax.set_ylabel("Mean LLM latency (ms)")
        ax.set_title(
            f"Deployment Viability: Added Delay — 7x28 ({NUM_NODES} nodes)\n"
            f"Green = viable (<1000ms/step), Red = too slow"
        )

        # Mark viable region boundary
        threshold_ms = STEP_DURATION_S * 1000
        for li, lat in enumerate(unique_lats):
            for ri, rate in enumerate(unique_rates):
                calls = min(rate * NUM_NODES, MAX_NODES_PER_STEP)
                if calls * lat < threshold_ms:
                    ax.add_patch(plt.Rectangle(
                        (ri - 0.5, li - 0.5), 1, 1,
                        fill=False, edgecolor="white", lw=1.5, linestyle="--"
                    ))

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")
    except ImportError:
        print("  matplotlib not available — skipping plot")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"SafeGAT-iLLM Latency Analysis — 7x28 ({NUM_NODES} junctions)")
    print(f"Output directory: {OUT_DIR}")

    config  = read_config()
    backend = _build_backend(config)
    gateway = LLMGateway(
        backend             = backend,
        min_call_interval_s = 0.0,   # no throttle during timing
        max_backoff_retries = 2,
        backoff_wait_s      = 5.0,
    )

    # ── 1. LLM call latency ───────────────────────────────────────────────────
    call_records = measure_llm_latency(gateway, N_LLM_CALLS)
    with open(OUT_DIR / "raw_call_latencies.json", "w") as f:
        json.dump(call_records, f, indent=2)

    successful_lats = [r["latency_ms"] for r in call_records if r["success"]]
    if successful_lats:
        print(f"\nLatency summary:")
        print(f"  n_successful  : {len(successful_lats)}/{N_LLM_CALLS}")
        print(f"  mean          : {np.mean(successful_lats):.1f} ms")
        print(f"  median        : {np.median(successful_lats):.1f} ms")
        print(f"  std           : {np.std(successful_lats):.1f} ms")
        print(f"  p95           : {np.percentile(successful_lats, 95):.1f} ms")
        print(f"  max           : {max(successful_lats):.1f} ms")

    # ── 2. Step overhead ──────────────────────────────────────────────────────
    step_records = measure_step_overhead(gateway, N_MOCK_STEPS)
    with open(OUT_DIR / "step_timing_log.json", "w") as f:
        json.dump(step_records, f, indent=2)

    # ── 3. Per-junction breakdown ─────────────────────────────────────────────
    per_jct = compute_per_junction_latency(call_records)
    if per_jct:
        csv_path = OUT_DIR / "per_junction_latency.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_jct[0].keys())
            writer.writeheader()
            writer.writerows(per_jct)
        print(f"\nPer-junction latency saved -> {csv_path}")

    # ── 4. Deployment viability ───────────────────────────────────────────────
    viability = compute_deployment_viability()

    # Summary CSV
    summary_rows = [
        {
            "metric": "n_nodes",            "value": NUM_NODES},
        {"metric": "max_nodes_per_step",    "value": MAX_NODES_PER_STEP},
        {"metric": "n_calls_tested",        "value": N_LLM_CALLS},
        {"metric": "n_successful",          "value": len(successful_lats)},
        {"metric": "mean_latency_ms",       "value": round(np.mean(successful_lats), 1) if successful_lats else "N/A"},
        {"metric": "median_latency_ms",     "value": round(np.median(successful_lats), 1) if successful_lats else "N/A"},
        {"metric": "p95_latency_ms",        "value": round(np.percentile(successful_lats, 95), 1) if successful_lats else "N/A"},
        {"metric": "mean_step_overhead_ms", "value": round(np.mean([r["total_ms"] for r in step_records]), 2)},
    ]
    with open(OUT_DIR / "latency_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)

    # ── 5. Plots ──────────────────────────────────────────────────────────────
    print("\n[5] Generating plots ...")
    plot_latency_distribution(call_records, str(OUT_DIR / "latency_distribution.png"))
    plot_deployment_viability(viability,    str(OUT_DIR / "deployment_viability.png"))

    print(f"\nLatency analysis complete. All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
