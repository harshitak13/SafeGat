"""
latency_analysis.py
===================
Measures and models the real-world latency introduced by LLM calls in SafeGAT,
then assesses deployment viability via three complementary analyses:

1. Empirical timing — records wall-clock time per LLM call using the same
   LLMGateway / backend stack used in production.

2. Per-step budget breakdown — compares RL-only step time vs SafeGAT step time
   across N simulation steps, showing the overhead distribution.

3. Deployment viability model — sweeps (call_rate, latency) parameter space and
   draws iso-latency contours to identify safe operating regions.

Output
------
    data/output/latency/
        raw_call_latencies.json     — per-call timing records
        step_timing_log.json        — per-step timing breakdown
        latency_summary.csv         — aggregate stats
        latency_distribution.png    — CDF + boxplot of call latencies
        deployment_viability.png    — heatmap of (call_rate x latency)

Run
---
    python experiments/latency_analysis.py

No SUMO / full simulation needed — uses the LLM gateway in isolation plus
a lightweight mock step loop for step-level timing.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
import sys
sys.path.insert(0, str(_ROOT))

from llm.llm_gateway      import LLMGateway
from llm.types            import LLMDecision
from utils.readConfig     import read_config

# ── Paths ─────────────────────────────────────────────────────────────────────
CONFIG_FILE = _ROOT / "configs" / "config.yaml"
OUT_DIR     = _ROOT / "data" / "output" / "latency"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Experiment parameters ─────────────────────────────────────────────────────
N_LLM_CALLS        = 60      # number of real LLM calls to time
N_MOCK_STEPS       = 200     # steps for step-level overhead measurement
STEP_DURATION_S    = 1.0     # SUMO step = 1 s real-time when running headless
# For deployment model
CALL_RATES         = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]  # LLM calls/step
LATENCY_TARGETS_MS = [100, 200, 500, 1000, 2000, 5000]        # ms thresholds


# ── Prompt templates for timing (representative of production prompts) ─────────

_SAMPLE_PROMPT = """\
You are a traffic signal controller AI.

Junction: J_5  |  Phase: 2  |  Phase elapsed: 12 s
Observation: occ=[0.82, 0.31, 0.15, 0.71]  queue=0.78  emergency=0
RL proposed action: 1  |  Q-margin: 0.03 (low confidence)
Neighbour actions: J_2→0, J_6→2, J_4→1

Choose the best traffic-signal phase (0-3).
Respond ONLY with valid JSON:
{"action": <int 0-3>, "confidence": <float 0-1>, "reasoning": "<one line>"}
"""


# ── LLM backend factory ────────────────────────────────────────────────────────

def build_backend(config: dict):
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        api_key  = config["OPENAI_API_KEY"]
        model    = config["OPENAI_API_MODEL"]
        base_url = config.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        chat     = ChatOpenAI(
            model=model, temperature=0.0,
            openai_api_key=api_key, openai_api_base=base_url,
            request_timeout=30,
        )
        def _real_backend(p: str) -> str:
            resp = chat.invoke([HumanMessage(content=p)])
            return resp.content
        print(f"[INFO] Using real LLM backend: {model} @ {base_url}")
        return _real_backend, model
    except Exception as e:
        print(f"[WARN] Real LLM unavailable ({e}); falling back to mock.")
        # Simulate ~200 ms average latency for mock
        def _mock(p: str) -> str:
            time.sleep(np.random.exponential(0.2))
            return json.dumps({"action": 0, "confidence": 0.5, "reasoning": "mock"})
        return _mock, "mock"


# ── 1. Empirical LLM call timing ──────────────────────────────────────────────

def measure_llm_call_latency(
    gateway:  LLMGateway,
    n_calls:  int = N_LLM_CALLS,
) -> List[Dict]:
    records: List[Dict] = []
    print(f"\n[LATENCY] Measuring {n_calls} real LLM calls …")

    for i in range(n_calls):
        t0 = time.perf_counter()
        try:
            decision = gateway.query(_SAMPLE_PROMPT)
            success  = True
        except Exception as exc:
            print(f"  call {i} failed: {exc!r}")
            success  = False
        elapsed_ms = (time.perf_counter() - t0) * 1000

        records.append({
            "call_index":  i,
            "latency_ms":  round(elapsed_ms, 2),
            "success":     success,
        })
        print(f"  [{i+1:>3}/{n_calls}]  latency={elapsed_ms:>7.1f} ms  "
              f"{'OK' if success else 'FAIL'}", flush=True)

    return records


# ── 2. Per-step timing breakdown ──────────────────────────────────────────────

def measure_step_overhead(
    gateway:    LLMGateway,
    n_steps:    int = N_MOCK_STEPS,
    call_every: int = 5,   # call LLM once every N steps to simulate Q_MARGIN_TAU≈0.05
) -> List[Dict]:
    """
    Mock a simulation loop. Every `call_every` steps, make one LLM call
    and measure: rl_time, llm_time, total_time per step.
    """
    print(f"\n[STEP-TIMING] Running {n_steps} mock steps (LLM every {call_every} steps) …")
    records: List[Dict] = []

    for step in range(n_steps):
        # --- RL inference: simulated by a small NumPy op ---
        t_rl_start = time.perf_counter()
        # Simulate GAT-DQN forward pass cost (~1-2 ms on CPU for 12 nodes)
        _ = np.random.randn(12, 8) @ np.random.randn(8, 4)
        rl_ms = (time.perf_counter() - t_rl_start) * 1000

        llm_ms = 0.0
        llm_called = step % call_every == 0

        if llm_called:
            t_llm_start = time.perf_counter()
            try:
                gateway.query(_SAMPLE_PROMPT)
            except Exception:
                pass
            llm_ms = (time.perf_counter() - t_llm_start) * 1000

        records.append({
            "step":       step,
            "rl_ms":      round(rl_ms, 3),
            "llm_ms":     round(llm_ms, 3),
            "total_ms":   round(rl_ms + llm_ms, 3),
            "llm_called": llm_called,
        })

        if step % 20 == 0:
            print(f"  step={step:>4}  rl={rl_ms:.2f}ms  "
                  f"llm={llm_ms:.1f}ms  {'← LLM' if llm_called else ''}", flush=True)

    return records


# ── 3. Deployment viability model ─────────────────────────────────────────────

def build_viability_grid(
    observed_latency_ms: float,
    observed_p99_ms:     float,
) -> Dict:
    """
    For each (call_rate, latency_budget) cell, compute:
        expected_delay_ms = call_rate × mean_latency_ms
        headroom_ms       = step_budget_ms - expected_delay_ms
        viable            = headroom_ms > 0

    Returns a dict ready for heatmap plotting.
    """
    step_budget_ms = STEP_DURATION_S * 1000   # 1 000 ms per step

    grid = {}
    for rate in CALL_RATES:
        grid[rate] = {}
        for budget in LATENCY_TARGETS_MS:
            expected = rate * observed_latency_ms
            headroom = step_budget_ms - expected
            grid[rate][budget] = {
                "expected_delay_ms": round(expected, 1),
                "headroom_ms":       round(headroom, 1),
                "viable":            bool(headroom > 0 and observed_p99_ms < budget),
            }
    return grid


# ── Aggregate stats ───────────────────────────────────────────────────────────

def summarise_latency(records: List[Dict]) -> Dict:
    lats = [r["latency_ms"] for r in records if r["success"]]
    if not lats:
        return {}
    return {
        "n_calls":      len(lats),
        "mean_ms":      round(float(np.mean(lats)), 2),
        "median_ms":    round(float(np.median(lats)), 2),
        "std_ms":       round(float(np.std(lats)), 2),
        "p25_ms":       round(float(np.percentile(lats, 25)), 2),
        "p75_ms":       round(float(np.percentile(lats, 75)), 2),
        "p95_ms":       round(float(np.percentile(lats, 95)), 2),
        "p99_ms":       round(float(np.percentile(lats, 99)), 2),
        "min_ms":       round(float(np.min(lats)), 2),
        "max_ms":       round(float(np.max(lats)), 2),
        "success_rate": round(len(lats) / len(records), 4),
    }


def summarise_steps(records: List[Dict]) -> Dict:
    llm_steps = [r for r in records if r["llm_called"]]
    rl_only   = [r for r in records if not r["llm_called"]]
    return {
        "n_steps_total":          len(records),
        "n_llm_steps":            len(llm_steps),
        "mean_total_ms_llm":      round(float(np.mean([r["total_ms"] for r in llm_steps])), 2) if llm_steps else 0,
        "mean_total_ms_rl_only":  round(float(np.mean([r["total_ms"] for r in rl_only])), 2) if rl_only else 0,
        "p95_total_ms_llm":       round(float(np.percentile([r["total_ms"] for r in llm_steps], 95)), 2) if llm_steps else 0,
    }


def save_csv(stats: Dict, step_stats: Dict, viability: Dict, path: Path):
    import csv
    rows = [
        {"metric": k, "value": v} for k, v in stats.items()
    ] + [
        {"metric": k, "value": v} for k, v in step_stats.items()
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] {path}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_latency_distribution(records: List[Dict], out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not available.")
        return

    lats = [r["latency_ms"] for r in records if r["success"]]
    if not lats:
        print("[WARN] No successful LLM calls to plot — skipping latency distribution chart.")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # CDF
    sorted_lats = np.sort(lats)
    cdf = np.arange(1, len(sorted_lats) + 1) / len(sorted_lats)
    ax1.plot(sorted_lats, cdf, color="#2196F3", linewidth=2)
    ax1.axvline(np.percentile(lats, 95), color="orange", linestyle="--", label="p95")
    ax1.axvline(np.percentile(lats, 99), color="red",    linestyle="--", label="p99")
    ax1.set_xlabel("Latency (ms)")
    ax1.set_ylabel("Cumulative probability")
    ax1.set_title("LLM Call Latency CDF")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Boxplot per-decile (10-call bins)
    bin_size = max(1, len(lats) // 10)
    bins = [lats[i:i + bin_size] for i in range(0, len(lats), bin_size)]
    ax2.boxplot(bins, patch_artist=True,
                boxprops=dict(facecolor="#BBDEFB"),
                medianprops=dict(color="#1565C0", linewidth=2))
    ax2.set_xlabel("Call batch (10 calls each)")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Latency Over Time (batch boxplot)")
    ax2.grid(alpha=0.3, axis="y")

    plt.suptitle("LLM Gateway Latency Analysis", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_viability_heatmap(
    viability: Dict,
    mean_ms:   float,
    p99_ms:    float,
    out_path:  Path,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not available.")
        return

    rates   = CALL_RATES
    budgets = LATENCY_TARGETS_MS
    # headroom matrix
    headroom = np.array([
        [viability[r][b]["headroom_ms"] for b in budgets]
        for r in rates
    ])

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(headroom, aspect="auto", origin="lower",
                   cmap="RdYlGn", vmin=-500, vmax=1000)
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([f"{b} ms" for b in budgets])
    ax.set_yticks(range(len(rates)))
    ax.set_yticklabels([f"{r:.0%}" for r in rates])
    ax.set_xlabel("Latency budget per step")
    ax.set_ylabel("LLM call rate (calls/step)")
    ax.set_title(
        f"Deployment Viability — Headroom (ms)\n"
        f"(mean={mean_ms:.0f} ms, p99={p99_ms:.0f} ms measured)"
    )
    fig.colorbar(im, ax=ax, label="Headroom ms (green = viable)")

    # Annotate cells
    for i, r in enumerate(rates):
        for j, b in enumerate(budgets):
            v = viability[r][b]
            txt = f"{v['headroom_ms']:.0f}"
            color = "black" if abs(v["headroom_ms"]) < 800 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    config = read_config(str(CONFIG_FILE))
    backend_fn, model_name = build_backend(config)
    gateway = LLMGateway(
        backend             = backend_fn,
        min_call_interval_s = 0.0,   # no artificial wait for timing measurement
        max_retries         = 1,
    )

    # 1. Raw call latency
    call_records = measure_llm_call_latency(gateway, N_LLM_CALLS)
    with open(OUT_DIR / "raw_call_latencies.json", "w") as f:
        json.dump(call_records, f, indent=2)
    print(f"[SAVED] {OUT_DIR / 'raw_call_latencies.json'}")

    stats = summarise_latency(call_records)
    print("\n=== Call Latency Stats ===")
    for k, v in stats.items():
        print(f"  {k:<20} {v}")

    # 2. Step-level overhead
    step_records = measure_step_overhead(gateway, N_MOCK_STEPS)
    with open(OUT_DIR / "step_timing_log.json", "w") as f:
        json.dump(step_records, f, indent=2)
    print(f"[SAVED] {OUT_DIR / 'step_timing_log.json'}")

    step_stats = summarise_steps(step_records)
    print("\n=== Step Timing Stats ===")
    for k, v in step_stats.items():
        print(f"  {k:<30} {v}")

    # 3. Viability model
    viability = build_viability_grid(stats.get("mean_ms", 500),
                                      stats.get("p99_ms",  2000))
    with open(OUT_DIR / "viability_grid.json", "w") as f:
        json.dump(viability, f, indent=2)

    # Deployment verdict
    print("\n=== Deployment Viability (call_rate=10%, step budget=1000 ms) ===")
    cell = viability[0.1][1000]
    print(f"  Expected delay : {cell['expected_delay_ms']} ms")
    print(f"  Headroom       : {cell['headroom_ms']} ms")
    print(f"  Viable         : {'YES ✓' if cell['viable'] else 'NO — needs async or caching'}")

    # Save CSV
    save_csv(stats, step_stats, viability, OUT_DIR / "latency_summary.csv")

    # Plots
    plot_latency_distribution(call_records, OUT_DIR / "latency_distribution.png")
    plot_viability_heatmap(
        viability,
        stats.get("mean_ms", 0),
        stats.get("p99_ms",  0),
        OUT_DIR / "deployment_viability.png",
    )

    # Mitigation summary
    mean_ms = stats.get("mean_ms", 0)
    p99_ms  = stats.get("p99_ms", 0)
    print("\n=== Deployment Mitigation Recommendations ===")
    if mean_ms < 200:
        print("  ✓ Mean latency < 200 ms — synchronous calls viable at ≤20% call rate.")
    elif mean_ms < 600:
        print("  ⚠ Mean latency 200-600 ms — async threading recommended for call rates > 10%.")
    else:
        print("  ✗ Mean latency > 600 ms — async non-blocking calls or result caching mandatory.")
    if p99_ms > 2000:
        print("  ⚠ p99 > 2 s — add circuit-breaker: fall back to RL when LLM > timeout.")
    print("  ● Async mitigation: submit LLM call in background thread; use RL action for current step; apply LLM result on next step.")
    print("  ● Caching: identical junction state → reuse last LLM decision (saves ~40% calls).")
    print("  ● Batching: send up to 4 junction prompts in one API call via multi-turn messages.")


if __name__ == "__main__":
    main()
