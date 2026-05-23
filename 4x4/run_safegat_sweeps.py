"""Run SafeGAT sensitivity, threshold, failure, and K/N sweeps for 4x4."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = {
    "uncertainty": 2.0,
    "anomaly": 1.5,
    "queue": 1.0,
    "waiting": 0.5,
    "safety": 1.0,
}


def parse_att(path: Path) -> float | None:
    if not path.exists():
        return None
    rows = [
        float(item.attrib.get("duration", 0.0))
        for item in ET.parse(path).getroot().findall("tripinfo")
    ]
    return sum(rows) / len(rows) if rows else None


def collect_metrics(output_dir: Path) -> dict:
    step_path = output_dir / "step_log.json"
    summary_path = output_dir / "intervention_summary.json"
    steps = json.loads(step_path.read_text()) if step_path.exists() else []
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return {
        "att": parse_att(output_dir / "safegat.tripinfo.xml"),
        "mean_queue": (
            sum(float(row.get("mean_queue", row.get("mean_occ", 0.0))) for row in steps)
            / len(steps)
            if steps else None
        ),
        "llm_calls": summary.get("llm_calls", steps[-1].get("llm_calls") if steps else None),
        "safety_adjustments": summary.get("safety_adjustments"),
        "override_precision_%": summary.get("override_precision_%"),
    }


def write_pareto_plots(rows: list[dict], output_root: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    for kind, filename in (
        ("threshold_sweep", "threshold_pareto.png"),
        ("intervention_rate", "intervention_rate_pareto.png"),
    ):
        points = [
            row for row in rows
            if row.get("kind") == kind
            and row.get("att") is not None
            and row.get("llm_calls") is not None
        ]
        if not points:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter([row["llm_calls"] for row in points], [row["att"] for row in points])
        for row in points:
            ax.annotate(row["name"], (row["llm_calls"], row["att"]), fontsize=8)
        ax.set_xlabel("LLM call count")
        ax.set_ylabel("Average travel time")
        ax.set_title(kind.replace("_", " ").title())
        fig.tight_layout()
        fig.savefig(output_root / filename, dpi=200)
        plt.close(fig)


def variants() -> list[dict]:
    runs = [{"kind": "baseline", "name": "baseline", "env": {}}]
    for key, value in DEFAULT_WEIGHTS.items():
        for factor in (0.5, 1.5):
            weights = dict(DEFAULT_WEIGHTS)
            weights[key] = value * factor
            runs.append({
                "kind": "risk_weight_sensitivity",
                "name": f"weight_{key}_{factor:g}x",
                "env": {"SAFEGAT_RISK_WEIGHTS_JSON": json.dumps(weights)},
            })
    for tau in (0.01, 0.02, 0.05, 0.10, 0.20):
        runs.append({
            "kind": "threshold_sweep",
            "name": f"tau_{tau:.2f}",
            "env": {"SAFEGAT_CONFIDENCE_THRESHOLD": str(tau)},
        })
    for rate in (0.10, 0.30, 0.50, 1.00):
        runs.append({
            "kind": "failure_injection",
            "name": f"failure_{rate:.2f}",
            "env": {"SAFEGAT_LLM_FAILURE_RATE": str(rate)},
        })
    for rate in (0.01, 0.02, 0.05, 0.10, 0.20):
        runs.append({
            "kind": "intervention_rate",
            "name": f"k_rate_{rate:.2f}",
            "env": {"SAFEGAT_INTERVENTION_RATE": str(rate)},
        })
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(ROOT / "sweep_results"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in variants():
        out_dir = output_root / variant["name"]
        env = os.environ.copy()
        env.update(variant["env"])
        env["SAFEGAT_RESULT_PATH"] = str(out_dir)
        print(f"[SWEEP] {variant['kind']} / {variant['name']} -> {out_dir}")
        if not args.dry_run:
            subprocess.run([args.python, "run_safegat.py"], cwd=ROOT, env=env, check=True)
        row = {
            "kind": variant["kind"],
            "name": variant["name"],
            **variant["env"],
            **collect_metrics(out_dir),
        }
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    with (output_root / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_pareto_plots(rows, output_root)


if __name__ == "__main__":
    main()
