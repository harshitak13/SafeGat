"""
run_all.py
==========
Runs all 7 SafeGAT plotting scripts in sequence and reports results.
All figures are saved to the figures/ directory as PDF + PNG.

Usage:
    python run_all.py
"""

import subprocess
import sys
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

SCRIPTS = [
    ("fig2_training_convergence.py", "Fig 2 — Training Convergence & Inference Stability"),
    ("fig3_benchmark_comparison.py", "Fig 3 — Traffic Efficiency & Benchmark Comparison"),
    ("fig4_robustness.py",           "Fig 4 — Robustness under Dynamic Disturbances"),
    ("fig5_ablation.py",             "Fig 5 — Ablation Study of SafeGAT Components"),
    ("fig6_per_junction.py",         "Fig 6 — Spatial Analysis of Intervention Behaviour"),
    ("fig7_latency.py",              "Fig 7 — Latency & Deployment Feasibility"),
    ("fig8_dashboard.py",            "Fig 8 — Comprehensive System Evaluation Dashboard"),
]

print("=" * 65)
print("SafeGAT — Generating All Figures")
print("=" * 65)
t0 = time.time()

ok = []; fail = []
for script, desc in SCRIPTS:
    print(f"\n>  {desc}")
    t = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True, cwd=str(HERE)
    )
    elapsed = time.time() - t
    if result.returncode == 0:
        print(result.stdout.rstrip())
        print(f"   [OK]  Done in {elapsed:.1f}s")
        ok.append(script)
    else:
        print(result.stdout.rstrip())
        print(result.stderr.rstrip())
        print(f"   [FAILED]  FAILED in {elapsed:.1f}s")
        fail.append(script)

print("\n" + "=" * 65)
print(f"Summary: {len(ok)}/{len(SCRIPTS)} scripts succeeded "
      f"({time.time()-t0:.1f}s total)")
if fail:
    print("Failed:", ", ".join(fail))

print("\nFigures saved to:  figures/")
figs = sorted(Path(HERE / "figures").glob("*.png"))
for f in figs:
    size_kb = f.stat().st_size // 1024
    print(f"  {f.name:50s}  {size_kb:>5} KB")
print("=" * 65)
