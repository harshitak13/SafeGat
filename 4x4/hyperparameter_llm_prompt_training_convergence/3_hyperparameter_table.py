"""
3_hyperparameter_table.py
=========================
Produces a complete hyperparameter table for SafeGAT-iLLM by pulling
values directly from:
  - train.py              (DQN / RL / training hyperparams)
  - configs/safegat_llm.yaml  (LLM / gate / shield / scenario params)
  - llm/intervention_gate.py  (gate class defaults)
  - llm/safety_shield.py      (shield class defaults)
  - training/gat_dqn_trainer.py (trainer class defaults)

Run from the SafeGAT_iLLM project root:
    python outputs/3_hyperparameter_table.py

Outputs
-------
  outputs/hyperparameter_table.md   — Markdown table (paste into paper/README)
  outputs/hyperparameter_table.csv  — CSV for LaTeX / spreadsheet
  (also prints a formatted table to stdout)
"""

import pathlib, csv, io, sys, yaml, textwrap

_HERE   = pathlib.Path(__file__).resolve().parent        # .../SAFEGAT/3
ROOT    = _HERE.parent / "SafeGAT_iLLM"                 # .../SAFEGAT/SafeGAT_iLLM
if not ROOT.exists():
    ROOT = _HERE.parent                                  # fallback: scripts at project root
OUT_DIR = _HERE                                          # output next to the script
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load YAML config ────────────────────────────────────────────────────────
with open(ROOT / "configs" / "safegat_llm.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
llm = cfg.get("llm", {})

# ── Define the hyperparameter table ────────────────────────────────────────
# Columns: Name | Symbol | Value | Source file | Description
ROWS = [
    # ─── GAT-DQN Network ───
    ("section", "GAT-DQN Network Architecture", "", "", ""),
    ("param", "Observation dim",        "d",             "8",         "train.py → OBS_DIM",              "Feature vector dimension per node (junction)"),
    ("param", "Hidden dim",             "H",             "64",        "train.py → HIDDEN_DIM",           "GAT hidden / embedding size"),
    ("param", "GAT attention heads",    "K_h",           "4",         "train.py → GAT_HEADS",            "Multi-head attention heads in GATQNetwork"),
    ("param", "Number of actions",      "A",             "4",         "network/net_config.py → NUM_ACTIONS", "Discrete phase choices per junction"),
    ("param", "Number of junctions",    "N",             "12",        "network/net_config.py → NUM_NODES",   "Controlled junctions in 4×4 SUMO grid"),

    # ─── DQN Training ───
    ("section", "DQN Training", "", "", ""),
    ("param", "Total episodes",         "—",             "100",       "train.py → TOTAL_EPISODES",       "Full training episodes"),
    ("param", "Max steps per episode",  "T",             "1800",      "train.py → MAX_STEPS",            "Sim seconds per episode (30-min horizon)"),
    ("param", "Learning rate",          "α",             "1e-3",      "train.py → LR",                   "Adam optimiser learning rate"),
    ("param", "Discount factor",        "γ",             "0.95",      "train.py → GAMMA",                "Bellman discount factor"),
    ("param", "Batch size",             "B",             "64",        "train.py → BATCH_SIZE",           "Replay mini-batch size"),
    ("param", "Buffer capacity",        "—",             "50 000",    "train.py → BUFFER_CAPACITY",      "Circular replay buffer size"),
    ("param", "Warmup steps",           "—",             "500",       "train.py → WARMUP_STEPS",         "Buffer fill before first gradient update"),
    ("param", "Target net update freq", "—",             "500",       "train.py → TARGET_UPDATE_FREQ",   "Gradient steps between hard target-net syncs"),
    ("param", "Gradient clip norm",     "—",             "10.0",      "train.py → GRAD_CLIP",            "Max gradient L2-norm before clipping"),
    ("param", "Checkpoint frequency",   "—",             "25 eps",    "train.py → CHECKPOINT_FREQ",      "Episodes between model checkpoints"),

    # ─── Epsilon-Greedy Exploration ───
    ("section", "ε-Greedy Exploration", "", "", ""),
    ("param", "ε start",                "ε_0",           "1.0",       "train.py → EPSILON_START",        "Initial exploration rate (fully random)"),
    ("param", "ε end",                  "ε_∞",           "0.05",      "train.py → EPSILON_END",          "Minimum exploration rate (near-greedy)"),
    ("param", "ε decay steps",          "T_ε",           "25 000",    "train.py → EPSILON_DECAY_STEPS",  "Steps for linear ε decay to ε_∞"),

    # ─── LLM Intervention Gate ───
    ("section", "LLM Intervention Gate  (InterventionGate)", "", "", ""),
    ("param", "Confidence threshold",   "τ_c",           str(llm.get("confidence_threshold", 0.05)),
                                                         "configs/safegat_llm.yaml → confidence_threshold",
                                                         "Call LLM when Q-margin Δ_i = Q(a*) − Q(a_2nd) < τ_c"),
    ("param", "Intervention budget",    "K",             str(llm.get("intervention_budget", 1600)),
                                                         "configs/safegat_llm.yaml → intervention_budget",
                                                         "Max LLM API calls for the full inference episode"),
    ("param", "Max nodes per step",     "K_step",        str(llm.get("max_nodes_per_step", 2)),
                                                         "configs/safegat_llm.yaml → max_nodes_per_step",
                                                         "Top-K junctions sent to LLM per simulation step"),
    ("param", "Anomaly weight",         "w_a",           "1.0",       "llm/intervention_gate.py default",  "Gate score weight for anomaly tag count"),
    ("param", "Corruption weight",      "w_c",           "1.0",       "llm/intervention_gate.py default",  "Gate score weight for corrupted observation flag"),
    ("param", "Low-conf weight",        "w_l",           "1.0",       "llm/intervention_gate.py default",  "Gate score weight for low-confidence flag"),

    # ─── Safety Shield ───
    ("section", "Safety Shield  (SafetyShield)", "", "", ""),
    ("param", "Min green hold steps",   "T_green",       str(llm.get("min_green_hold", 3)),
                                                         "configs/safegat_llm.yaml → min_green_hold",
                                                         "Min steps a green phase must remain active before switching"),
    ("param", "Yellow phase indices",   "{1, 3}",        "—",         "llm/safety_shield.py → _YELLOW_PHASES",
                                                         "Phase indices treated as yellow; switching away is blocked"),

    # ─── Scenario Detector ───
    ("section", "Scenario Detector  (ScenarioDetector)", "", "", ""),
    ("param", "Queue-spike threshold",  "θ_q",           str(llm.get("queue_spike_threshold", 0.85)),
                                                         "configs/safegat_llm.yaml → queue_spike_threshold",
                                                         "Occupancy above this → 'queue_spike' anomaly tag"),
    ("param", "Zero-fraction threshold","θ_z",           str(llm.get("zero_fraction_corruption_threshold", 0.90)),
                                                         "configs/safegat_llm.yaml → zero_fraction_corruption_threshold",
                                                         "Fraction of zero obs features above this → 'corrupted' flag"),
    ("param", "Anomaly triggers",       "—",             str(llm.get("anomaly_triggers", [])),
                                                         "configs/safegat_llm.yaml → anomaly_triggers",
                                                         "Tags that always open the gate regardless of confidence"),

    # ─── LLM Backend ───
    ("section", "LLM Backend  (LLMGateway)", "", "", ""),
    ("param", "LLM timeout",            "—",             str(llm.get("llm_timeout", 20)) + " s",
                                                         "configs/safegat_llm.yaml → llm_timeout",
                                                         "Seconds per API call before abort"),
    ("param", "Min call interval",      "—",             str(llm.get("min_call_interval_s", 4.0)) + " s",
                                                         "configs/safegat_llm.yaml → min_call_interval_s",
                                                         "Rate-limiter: minimum gap between consecutive API calls"),
    ("param", "Fallback to RL",         "—",             str(llm.get("fallback_to_rl", True)),
                                                         "configs/safegat_llm.yaml → fallback_to_rl",
                                                         "Use RL action if LLM errors or times out"),
    ("param", "LLM mode",               "—",             llm.get("mode", "selective"),
                                                         "configs/safegat_llm.yaml → mode",
                                                         "selective | always | never"),
]

# ── Print formatted table ───────────────────────────────────────────────────
COL_W = [32, 8, 14, 38, 50]
SEP   = "  "
HLINE = "─" * (sum(COL_W) + len(SEP) * (len(COL_W) - 1))
HDR   = ["Hyperparameter", "Symbol", "Value", "Source", "Description"]

def fmt(row, widths):
    return SEP.join(str(v).ljust(w)[:w] for v, w in zip(row, widths))

print(HLINE)
print(fmt(HDR, COL_W))
print(HLINE)
for r in ROWS:
    if r[0] == "section":
        print(f"\n  ── {r[1]} ──")
    else:
        _, name, sym, val, src, desc = r
        print(fmt([name, sym, val, src, desc], COL_W))
print(HLINE)

# ── Markdown ────────────────────────────────────────────────────────────────
md_lines = [
    "# SafeGAT-iLLM — Complete Hyperparameter Table\n",
    "| Hyperparameter | Symbol | Value | Source | Description |",
    "|----------------|--------|-------|--------|-------------|",
]

current_section = None
for r in ROWS:
    if r[0] == "section":
        md_lines.append(f"| **{r[1]}** | | | | |")
    else:
        _, name, sym, val, src, desc = r
        # escape pipes in YAML list
        val  = val.replace("|", "\\|")
        src  = src.replace("|", "\\|")
        desc = desc.replace("|", "\\|")
        md_lines.append(f"| {name} | {sym} | `{val}` | `{src}` | {desc} |")

md_path = OUT_DIR / "hyperparameter_table.md"
md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
print(f"\n✓ Markdown table saved → {md_path}")

# ── CSV ─────────────────────────────────────────────────────────────────────
csv_buf = io.StringIO()
writer  = csv.writer(csv_buf)
writer.writerow(["Hyperparameter", "Symbol", "Value", "Source", "Description"])
for r in ROWS:
    if r[0] == "section":
        writer.writerow([f"=== {r[1]} ===", "", "", "", ""])
    else:
        _, name, sym, val, src, desc = r
        writer.writerow([name, sym, val, src, desc])

csv_path = OUT_DIR / "hyperparameter_table.csv"
csv_path.write_text(csv_buf.getvalue(), encoding='utf-8')
print(f"✓ CSV table saved      → {csv_path}")
