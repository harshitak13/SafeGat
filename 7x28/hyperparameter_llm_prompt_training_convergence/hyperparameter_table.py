"""
hyperparameter_table.py — Hyperparameter Table for 7x28 SafeGAT-iLLM
======================================================================
Produces a Markdown + CSV hyperparameter table reflecting all parameter
changes made when scaling from the 4x4 (12-node) to 7x28 (196-node) grid.

Run from the project root::

    python hyperparameter_llm_prompt_training_convergence/hyperparameter_table.py

Outputs
-------
    hyperparameter_llm_prompt_training_convergence/
        hyperparameter_table.md
        hyperparameter_table.csv
"""

import csv
import os
import pathlib

_HERE   = pathlib.Path(__file__).resolve().parent
_ROOT   = _HERE.parent
OUT_DIR = _HERE
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns: Category | Name | Symbol | Value (7x28) | Old Value (4x4) | Source | Description
ROWS = [
    # ─── Network Architecture ───
    ("section", "GAT-DQN Network Architecture", "", "", "", "", ""),
    ("param", "Observation dim",        "d",      "8",         "8",       "train.py → OBS_DIM",              "Feature vector per junction (unchanged)"),
    ("param", "Hidden dim",             "H",      "128",       "64",      "train.py → HIDDEN_DIM",           "Larger to handle 196-node graph"),
    ("param", "GAT attention heads",    "K_h",    "4",         "4",       "train.py → GAT_HEADS",            "Multi-head attention (unchanged)"),
    ("param", "Number of actions",      "A",      "4",         "4",       "net_config.py → NUM_ACTIONS",     "Discrete phase choices per junction"),
    ("param", "Number of junctions",    "N",      "196",       "12",      "net_config.py → NUM_NODES",       "Controlled junctions (7x28 vs 4x4)"),
    ("param", "Graph rows",             "R",      "10",        "3",       "net_config.py → GRID_ROWS",       "Irregular rows in 7x28 network"),
    ("param", "Max graph cols",         "C",      "28",        "4",       "net_config.py → GRID_COLS",       "Maximum columns in any row"),

    # ─── DQN Training ───
    ("section", "DQN Training", "", "", "", "", ""),
    ("param", "Total episodes",         "—",      "200",       "100",     "train.py → TOTAL_EPISODES",       "More episodes for larger network"),
    ("param", "Max steps/episode",      "T",      "3600",      "1800",    "train.py → MAX_STEPS",            "Full 1-hour simulation per episode"),
    ("param", "Checkpoint frequency",   "—",      "25 eps",    "25 eps",  "train.py → CHECKPOINT_FREQ",      "Unchanged"),
    ("param", "Learning rate",          "α",      "1e-3",      "1e-3",    "train.py → LR",                   "Unchanged"),
    ("param", "Discount factor",        "γ",      "0.95",      "0.95",    "train.py → GAMMA",                "Unchanged"),
    ("param", "Batch size",             "B",      "64",        "64",      "train.py → BATCH_SIZE",           "Unchanged"),
    ("param", "Buffer capacity",        "—",      "200 000",   "50 000",  "train.py → BUFFER_CAPACITY",      "Scaled: 196 nodes generate 16x more data"),
    ("param", "Warmup steps",           "—",      "2 000",     "500",     "train.py → WARMUP_STEPS",         "Larger warmup for bigger buffer"),
    ("param", "Target net update freq", "—",      "1 000",     "500",     "train.py → TARGET_UPDATE_FREQ",   "Less frequent for bigger network"),
    ("param", "Gradient clip norm",     "—",      "10.0",      "10.0",    "train.py → GRAD_CLIP",            "Unchanged"),

    # ─── Epsilon-Greedy Exploration ───
    ("section", "Epsilon-Greedy Exploration", "", "", "", "", ""),
    ("param", "Initial epsilon",        "ε_0",    "1.0",       "1.0",     "train.py → EPSILON_START",        "Unchanged"),
    ("param", "Final epsilon",          "ε_∞",    "0.05",      "0.05",    "train.py → EPSILON_END",          "Unchanged"),
    ("param", "Decay steps",            "—",      "100 000",   "25 000",  "train.py → EPSILON_DECAY_STEPS",  "Longer decay for 196-node grid"),

    # ─── SafeGAT LLM Gate ───
    ("section", "SafeGAT Selective Intervention", "", "", "", "", ""),
    ("param", "Q-margin threshold",     "τ",      "0.05",      "0.05",    "run_safegat.py → Q_MARGIN_TAU",   "Unchanged: same uncertainty criterion"),
    ("param", "LLM budget/episode",     "B",      "6 400",     "1 600",   "run_safegat.py → LLM_BUDGET",     "Scaled ~4x for 196 nodes"),
    ("param", "Max nodes/step",         "—",      "8",         "2",       "run_safegat.py → MAX_NODES_PER_STEP", "Scaled: review more nodes per step"),
    ("param", "Min green hold",         "t_min",  "3 steps",   "3 steps", "run_safegat.py → MIN_GREEN_STEPS","Unchanged"),
    ("param", "LLM call interval",      "—",      "4.0 s",     "4.0 s",   "safegat_llm.yaml",                "Rate limiter (unchanged)"),
    ("param", "Sim length",             "—",      "3 600 s",   "1 600 s", "run_safegat.py → SIM_SECONDS",    "Full 1-hour inference episode"),

    # ─── Scenario Detector ───
    ("section", "Scenario Detector", "", "", "", "", ""),
    ("param", "Queue spike threshold",  "—",      "0.85",      "0.85",    "safegat_llm.yaml",                "Unchanged"),
    ("param", "Zero-fraction threshold","—",      "0.90",      "0.90",    "safegat_llm.yaml",                "Unchanged"),

    # ─── Network Files ───
    ("section", "SUMO Network", "", "", "", "", ""),
    ("param", "Network file",           "—",      "7x28.net.xml","4x4.net.xml","network/",                  "7x28 grid network"),
    ("param", "Route file",             "—",      "7x28.rou.xml","4x4.rou.xml","network/",                  "7x28 traffic flows"),
    ("param", "Config file",            "—",      "7x28.sumocfg","4x4.sumocfg","network/",                  "7x28 SUMO config"),
]


def write_markdown(rows: list, path: str) -> None:
    lines = [
        "# SafeGAT-iLLM Hyperparameter Table — 7x28 Network",
        "",
        "All changes relative to the 4x4 (12-junction) baseline are highlighted.",
        "",
        "| Category | Parameter | Symbol | Value (7x28) | Old (4x4) | Source | Description |",
        "|----------|-----------|--------|-------------|-----------|--------|-------------|",
    ]
    current_section = ""
    for row in rows:
        if row[0] == "section":
            current_section = row[1]
            lines.append(f"| **{row[1]}** | | | | | | |")
        else:
            _, name, sym, val_new, val_old, src, desc = row
            changed = " ⟵" if val_new != val_old else ""
            lines.append(
                f"| {current_section} | {name} | {sym} | **{val_new}**{changed} | {val_old} | `{src}` | {desc} |"
            )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Markdown written -> {path}")


def write_csv(rows: list, path: str) -> None:
    fieldnames = ["Category", "Parameter", "Symbol", "Value_7x28", "Value_4x4", "Source", "Description"]
    current_section = ""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if row[0] == "section":
                current_section = row[1]
                writer.writerow({
                    "Category": row[1], "Parameter": "", "Symbol": "",
                    "Value_7x28": "", "Value_4x4": "", "Source": "", "Description": "",
                })
            else:
                _, name, sym, val_new, val_old, src, desc = row
                writer.writerow({
                    "Category": current_section, "Parameter": name, "Symbol": sym,
                    "Value_7x28": val_new, "Value_4x4": val_old, "Source": src, "Description": desc,
                })
    print(f"CSV written -> {path}")


if __name__ == "__main__":
    md_path  = str(OUT_DIR / "hyperparameter_table.md")
    csv_path = str(OUT_DIR / "hyperparameter_table.csv")
    write_markdown(ROWS, md_path)
    write_csv(ROWS, csv_path)
    print("\nHyperparameter table generation complete.")
    print(f"  Markdown : {md_path}")
    print(f"  CSV      : {csv_path}")
