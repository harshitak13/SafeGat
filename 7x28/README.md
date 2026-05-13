# SafeGAT-iLLM — 7×28 Network

Adaptation of SafeGAT-iLLM from the original 4×4 (12-junction) SUMO grid
to the **7×28 (196-junction)** network.

## What changed

| File | Change |
|------|--------|
| `network/net_config.py` | 196 junctions, 10 rows, 28 max cols; full incoming-edge and neighbour maps |
| `network/graph_builder.py` | Edge index for 196-node graph (640 directed edges) |
| `envs/grid_env_wrapper.py` | Supports arbitrary N junctions; 196-node GridEnv |
| `utils/make_tsc_env.py` | Points to `7x28.net.xml` instead of `4x4.net.xml` |
| `train.py` | `HIDDEN_DIM=128`, `BUFFER=200k`, `WARMUP=2000`, `EPISODES=200`, `MAX_STEPS=3600` |
| `run_safegat.py` | `LLM_BUDGET=6400`, `MAX_NODES_PER_STEP=8`, `SIM_SECONDS=3600`, per-junction CSV |
| `configs/safegat_llm.yaml` | Updated budgets and paths to match 7×28 |
| `ablation_v1/v2/v3` | All three ablation variants updated for 196 nodes |
| `hyperparameter_table.py` | Full before/after comparison table (4×4 → 7×28) |
| `latency_analysis.py` | 196-node deployment viability sweep |
| `per_intersection_analysis.py` | Spatial heatmaps and per-junction CSV for all 196 nodes |
| `benchmark_results.py` | Benchmark comparison across all variants |

**The GAT-DQN architecture, LLM pipeline, selective intervention logic, and
safety shield are unchanged.** Only the topology constants, hyperparameters,
and output files are adapted.

## Key hyperparameter changes

| Parameter | 4×4 | 7×28 | Reason |
|-----------|-----|------|--------|
| NUM_NODES | 12 | 196 | Network size |
| HIDDEN_DIM | 64 | 128 | Bigger graph needs more capacity |
| BUFFER_CAPACITY | 50,000 | 200,000 | 196 nodes generate 16× more transitions |
| WARMUP_STEPS | 500 | 2,000 | Proportional to buffer |
| EPSILON_DECAY_STEPS | 25,000 | 100,000 | Longer exploration |
| TARGET_UPDATE_FREQ | 500 | 1,000 | Stabilises larger network |
| TOTAL_EPISODES | 100 | 200 | More training for bigger grid |
| MAX_STEPS | 1,800 | 3,600 | Full 1-hour episode |
| LLM_BUDGET | 1,600 | 6,400 | Scaled for 196 nodes |
| MAX_NODES_PER_STEP | 2 | 8 | Review more nodes per step |
| SIM_SECONDS | 1,600 | 3,600 | Full 1-hour inference |

## Quick start

```bash
# 1. Install dependencies
pip install torch torch_geometric loguru numpy pyyaml langchain langchain-openai

# Install TransSimHub (SUMO wrapper)
git clone https://github.com/Traffic-Alpha/TransSimHub.git
cd TransSimHub && pip install -e . && cd ..

# 2. Set your API key
cp configs/config.yaml.example configs/config.yaml
# Edit configs/config.yaml with your OpenAI API key

# 3. Train the GAT-DQN
python train.py
# Produces: models/gat_dqn_final.pt

# 4. Run SafeGAT inference
python run_safegat.py
# Produces: output/step_log.json, output/per_junction_results.csv, etc.

# 5. Ablation studies
python ablation_v1_gat_dqn_only.py
python ablation_v2_uniform_llm.py
python ablation_v3_full_safegat.py

# 6. Analysis scripts (no SUMO needed)
python hyperparameter_llm_prompt_training_convergence/hyperparameter_table.py
python latency_per_intersection_robustness/per_intersection_analysis.py
python latency_per_intersection_robustness/latency_analysis.py
python benchmark_results/benchmark_results.py
```

## Network topology

The 7×28 network has **196 priority junctions** arranged in 10 rows
(up to 28 junctions per row). All junctions are connected by bidirectional
edges to their N/S/E/W neighbours, yielding **640 directed edges** in the
graph attention network.

```
Row 0 (28 nodes): J226 J208 J145 J199 J136 J19 ... J244
Row 1 (27 nodes): J227 J209 J146 J200 J20  J124 ... J245
...
Row 9 (27 nodes): J232 J214 J151 J205 J142 J25  ... J250
```

## Output files

| File | Description |
|------|-------------|
| `output/step_log.json` | Per-step reward, occupancy, LLM usage |
| `output/intervention_summary.json` | Episode-level LLM stats |
| `output/per_junction_results.csv` | Per-junction reward + LLM calls (all 196) |
| `output/llm/safegat_decisions.jsonl` | Full JSONL audit trail |
| `output/safegat.tripinfo.xml` | SUMO trip-info for post-analysis |
| `output/per_junction_analysis/reward_heatmap.png` | Spatial reward heatmap |
| `output/per_junction_analysis/llm_calls_heatmap.png` | LLM call distribution |
