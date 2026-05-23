"""Sensitivity, threshold, failure, and K/N sweeps for CityFlow datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cityflow_safegat.offline_runner import main as run_offline_main


DEFAULT_WEIGHTS = {
    "uncertainty": 2.0,
    "anomaly": 1.5,
    "queue": 1.0,
    "waiting": 0.5,
    "safety": 1.0,
}


def variants() -> list[dict]:
    runs = [{"kind": "baseline", "name": "baseline", "args": []}]
    for key, value in DEFAULT_WEIGHTS.items():
        for factor in (0.5, 1.5):
            weights = dict(DEFAULT_WEIGHTS)
            weights[key] = value * factor
            runs.append({
                "kind": "risk_weight_sensitivity",
                "name": f"weight_{key}_{factor:g}x",
                "args": [
                    "--weight-uncertainty", str(weights["uncertainty"]),
                    "--weight-anomaly", str(weights["anomaly"]),
                    "--weight-queue", str(weights["queue"]),
                    "--weight-waiting", str(weights["waiting"]),
                    "--weight-safety", str(weights["safety"]),
                ],
            })
    for tau in (0.01, 0.02, 0.05, 0.10, 0.20):
        runs.append({
            "kind": "threshold_sweep",
            "name": f"tau_{tau:.2f}",
            "args": ["--confidence-threshold", str(tau)],
        })
    for rate in (0.10, 0.30, 0.50, 1.00):
        runs.append({
            "kind": "failure_injection",
            "name": f"failure_{rate:.2f}",
            "args": ["--failure-injection-rate", str(rate)],
        })
    for rate in (0.01, 0.02, 0.05, 0.10, 0.20):
        runs.append({
            "kind": "intervention_rate",
            "name": f"k_rate_{rate:.2f}",
            "args": ["--intervention-rate", str(rate)],
        })
    return runs


def read_summary(path: Path) -> dict:
    summary_path = path / "intervention_summary.json"
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "att_proxy": summary.get("att_proxy"),
        "mean_queue": summary.get("mean_queue"),
        "llm_calls": summary.get("llm_calls"),
        "llm_failures": summary.get("llm_failures", 0),
        "safety_adjustments": summary.get("safety_adjustments"),
        "intervention_rate": summary.get("intervention_rate"),
        "confidence_threshold": summary.get("confidence_threshold"),
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
            and row.get("att_proxy") is not None
            and row.get("llm_calls") is not None
        ]
        if not points:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter([row["llm_calls"] for row in points], [row["att_proxy"] for row in points])
        for row in points:
            ax.annotate(row["name"], (row["llm_calls"], row["att_proxy"]), fontsize=8)
        ax.set_xlabel("LLM call count")
        ax.set_ylabel("ATT proxy")
        ax.set_title(kind.replace("_", " ").title())
        fig.tight_layout()
        fig.savefig(output_root / filename, dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--roadnet", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--impl-dir", required=True)
    parser.add_argument("--output-root", default="safegat_cityflow_sweeps")
    parser.add_argument("--steps", default="120")
    parser.add_argument("--bucket-seconds", default="30")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_root = dataset_dir / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in variants():
        out_name = f"{args.output_root}/{variant['name']}"
        argv = [
            "--dataset-dir", str(dataset_dir),
            "--roadnet", args.roadnet,
            "--flow", args.flow,
            "--impl-dir", args.impl_dir,
            "--output-dir", out_name,
            "--steps", args.steps,
            "--bucket-seconds", args.bucket_seconds,
            *variant["args"],
        ]
        print(f"[CITYFLOW SWEEP] {variant['kind']} / {variant['name']}")
        if not args.dry_run:
            run_offline_main(argv)
        rows.append({
            "kind": variant["kind"],
            "name": variant["name"],
            **read_summary(dataset_dir / out_name),
        })

    fieldnames = sorted({key for row in rows for key in row})
    with (output_root / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_pareto_plots(rows, output_root)


if __name__ == "__main__":
    main()
