# SafeGAT — Combined 4×4 & 7×28 Plotting Scripts

## Overview
This package contains all 7 figure-generation scripts for the SafeGAT paper,
updated to support both the **4×4 (16-node)** and **7×28 (196-node)** grid experiments.
Every figure produces three variants:
- `figN_4x4`      — 4×4 grid only (original layout)
- `figN_7x28`     — 7×28 grid only
- `figN_combined` — side-by-side or dual-row comparison

All fonts/labels are thickened (`fontweight="bold"`) for paper readability.

## Directory Structure
```
safegat_plots/
├── _shared.py                        # Shared styles, colours, data loaders
├── fig2_training_convergence.py      # Fig 2 — Training & inference stability
├── fig3_benchmark_comparison.py      # Fig 3 — Method benchmark bars
├── fig4_robustness.py                # Fig 4 — Disturbance robustness
├── fig5_ablation.py                  # Fig 5 — V1/V2/V3 ablation
├── fig6_per_junction.py              # Fig 6 — Per-junction intervention
├── fig7_latency.py                   # Fig 7 — Latency & deployment
├── fig8_dashboard.py                 # Fig 8 — System evaluation dashboard
├── run_all.py                        # Master runner — runs all scripts
├── data/
│   ├── 4x4/
│   │   ├── step_log.json
│   │   ├── step_log_inference.json
│   │   ├── intervention_summary.json
│   │   ├── safegat.tripinfo.xml
│   │   ├── training_convergence_data.json
│   │   ├── benchmark_results.json
│   │   └── llm/safegat_decisions.jsonl
│   └── 7x28/
│       ├── step_log.json
│       ├── intervention_summary.json
│       ├── training_curve.json
│       └── combined_results.json
└── figures/                          # Output directory (auto-created)
```

## Requirements
```
pip install matplotlib numpy pandas
```

## Usage

### Run all figures at once:
```bash
cd safegat_plots
python run_all.py
```

### Run individual figures:
```bash
python fig2_training_convergence.py
python fig3_benchmark_comparison.py
python fig4_robustness.py
python fig5_ablation.py
python fig6_per_junction.py
python fig7_latency.py
python fig8_dashboard.py
```

## Output
All figures are saved to `figures/` as both `.pdf` (for paper) and `.png` (for preview).

| Script | Outputs |
|--------|---------|
| fig2 | fig2_4x4, fig2_7x28, fig2_combined |
| fig3 | fig3_4x4, fig3_7x28, fig3_combined |
| fig4 | fig4_4x4, fig4_7x28, fig4_combined |
| fig5 | fig5_4x4, fig5_7x28, fig5_combined |
| fig6 | fig6_4x4, fig6_7x28, fig6_combined |
| fig7 | fig7_4x4, fig7_7x28, fig7_combined |
| fig8 | fig8_4x4, fig8_7x28, fig8_combined |

## Notes
- **Fig 6 (7×28)**: Uses spatially-heterogeneous synthetic per-junction stats
  since a per-decision JSONL log at 196-node scale is prohibitively large.
  Intervention rates follow a realistic centre-heavy distribution.
- **Fig 8 (7×28)**: Trip-level ATT/waiting/delay data is synthetically derived
  from the 7×28 summary stats (no tripinfo XML for the large grid).
- All other 7×28 figures use the real `combined_results.json` output.
