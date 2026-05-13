# SafeGAT: Uncertainty-Gated and Safety-Constrained LLM Supervision for Graph Attention Reinforcement Learning in Traffic Signal Control

> **Paper:** SafeGAT — submitted to *Expert Systems with Applications* (Elsevier), Manuscript No. ESWA-D-26-17727  
> **Authors:** Saidulu Thadikamalla, Harshita Karnam, Piyush Joshi  
> **Affiliation:** Indian Institute of Information Technology Sri City, Andhra Pradesh, India

---

## Overview

Deep reinforcement learning (DRL) has advanced adaptive traffic signal control (TSC), but value-based controllers remain vulnerable when Q-value estimates are ambiguous, observations are corrupted, or traffic demand shifts from the training distribution. Existing LLM-assisted TSC methods apply language-model reasoning uniformly, creating unnecessary overrides and latency without guaranteeing feasibility.

**SafeGAT** addresses this by treating the LLM as a *conditional advisor*, not a controller:

```
GAT-DQN Policy  →  Uncertainty/Anomaly Gate  →  Top-K LLM Advisor  →  Safety Projection  →  Execute
   (propose)            (should we intervene?)      (refine if needed)     (enforce feasibility)
```

The result is a three-tier authority hierarchy where the RL policy proposes, the LLM advises only under measurable risk, and a deterministic safety layer determines the final executable action — unconditionally.

---

## Key Contributions

**1. SafeGAT Framework** — A risk-bounded supervisory architecture integrating GAT-DQN, selective LLM refinement, and a safety projection layer.

**2. Uncertainty-Gated Intervention** — LLM calls are triggered only when the normalized Q-value confidence margin (gap between top-2 Q-values) falls below threshold τ, or when structured anomaly indicators fire:
- Queue-spike anomaly (EMA z-score > 3.0)
- Sensor-corruption flag (out-of-range occupancy or negative queue)
- Spillback indicator (queue exceeds lane capacity)
- Emergency vehicle event flag

**3. Budgeted Top-K Allocation** — Restricts LLM refinement to the K highest-risk intersections per control step, bounding inference overhead to O(K · C_LLM) regardless of network size.

**4. Safety Projection Layer** — A deterministic operator that enforces legal phase membership, valid phase transitions, and minimum green-time constraints on every executed action, with a formal feasibility guarantee (Proposition 1).

**5. Fully Specified LLM Pipeline** — Structured prompt template, JSON output schema, two-stage parser (strict decode + regex fallback), and a direct phase-index mapping from LLM output to signal-control action.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SafeGAT Control Loop                         │
│                                                                     │
│  Traffic Env  →  GAT-DQN Agent  →  SafeGAT Control Layer           │
│  (SUMO)           ├─ Node Feature Encoder                           │
│                   ├─ GAT Layers (spatial dependencies)              │
│                   └─ Online Q-Network + Target Q-Network            │
│                              ↓                                      │
│                    Scenario Detector                                 │
│                    (queue spike, spillback, corruption, emergency)   │
│                              ↓                                      │
│                    Intervention Gate                                 │
│                    (confidence margin + anomaly score → risk ρ)     │
│                              ↓                                      │
│                    Top-K Selection  (|I_t| ≤ K)                     │
│                              ↓                                      │
│                    LLM Gateway  (structured prompt → JSON)          │
│                              ↓                                      │
│                    Safety Shield  (legal phase + min-green)         │
│                              ↓                                      │
│                    Final Action  →  Simulator                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Results

Evaluated on SUMO 4×4 (16 intersections) and 7×28 (196 intersections) traffic networks:

| Method | ATT (s) | Queue Length | Delay (s) | Throughput |
|---|---|---|---|---|
| Webster (Fixed) | 227 | 179 | 206 | 344 |
| Plain DQN | 179 | 134 | 157 | 402 |
| CoLight | 151 | 110 | 132 | 432 |
| LLMLight | 155 | 114 | 136 | 428 |
| GAT-DQN (no LLM) | 151 | 110 | 132 | 432 |
| **SafeGAT** | **145** | **101** | **124** | **441** |

*4×4 grid results. SafeGAT achieves best performance on all four metrics at both scales.*

**Ablation highlights:**
- Uniform LLM (no gate): 2570 LLM calls, hundreds of safety violations
- SafeGAT (gated): 640 LLM calls (~4× reduction), **zero safety violations**

---

## Installation

### Requirements
- Python 3.8+
- [SUMO](https://sumo.dlr.de/docs/Installing/index.html) traffic simulator
- PyTorch
- torch-geometric (for GAT layers)

```bash
git clone https://github.com/<your-username>/SafeGAT.git
cd SafeGAT
pip install -r requirements.txt
```

### SUMO Setup
```bash
# Ubuntu/Debian
sudo apt-get install sumo sumo-tools sumo-doc

# macOS
brew install sumo

# Verify
sumo --version
```

---

## Usage

### Training

```bash
# Train on 4x4 grid
python train.py --network 4x4 --episodes 100 --safegat --llm-backend llama-3.1-8b

# Train on 7x28 grid
python train.py --network 7x28 --episodes 100 --safegat --llm-backend llama-3.1-8b

# Train without LLM (GAT-DQN only baseline)
python train.py --network 4x4 --episodes 100 --no-safegat
```

### Evaluation

```bash
python evaluate.py --network 4x4 --checkpoint checkpoints/safegat_4x4.pt
```

### LLM Backend Configuration

SafeGAT supports multiple LLM backends via a local API endpoint:

```python
# config.py
LLM_BACKEND = "llama-3.1-8b"   # default, best composite score
LLM_ENDPOINT = "http://localhost:11434/api"
LLM_TIMEOUT_MS = 500
INTERVENTION_BUDGET_K = 3
CONFIDENCE_THRESHOLD = 0.05
```

Tested backends (ranked by composite score — Traffic Efficiency 35% + Intervention Control 40% + 7×28 Reward 25%):

| Rank | Backend | Composite Score |
|---|---|---|
| 1st | Llama-3.1-8B | 100.0 |
| 2nd | Llama-4-Scout-17B | 75.3 |
| 3rd | Qwen3-32B | 47.0 |
| 4th | GPT-OSS-120B | 26.7 |
| 5th | Llama-3.3-70B | — |

> Note: larger model size does not imply better control quality. Calibrated accept/override behavior under structured risk signals matters more than parameter count.

---

## Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| Confidence threshold τ_c | 0.05 | Q-margin below which LLM is triggered |
| Queue-spike threshold τ_q | 3.0 | EMA z-score for anomaly detection |
| Intervention budget K | network-dependent | Max LLM calls per control step |
| LLM timeout | 500 ms | Fallback to RL action if exceeded |
| Risk weights λ_u / λ_a / λ_q / λ_w / λ_s | 2.0 / 1.5 / 1.0 / 0.5 / 1.0 | Risk score coefficients |
| Discount factor γ | 0.99 | RL discount |
| EMA smoothing α | 0.05 | For queue-spike running stats |

---

## Project Structure

```
SafeGAT/
├── agents/
│   ├── gat_dqn.py          # Graph Attention DQN backbone
│   └── replay_buffer.py    # Experience replay
├── safegat/
│   ├── scenario_detector.py  # Anomaly detection (queue spike, corruption, spillback, emergency)
│   ├── intervention_gate.py  # Risk scoring and Top-K selection
│   ├── llm_gateway.py        # Prompt construction, LLM query, JSON parsing
│   └── safety_shield.py      # Deterministic safety projection layer
├── envs/
│   ├── sumo_env.py           # SUMO environment wrapper
│   └── networks/             # 4x4 and 7x28 SUMO network files
├── train.py
├── evaluate.py
├── config.py
└── requirements.txt
```

---

## LLM Prompt Format

The LLM receives a structured 6-block prompt and must return a strict JSON response:

```json
{
  "decision": "accept" | "override",
  "final_phase": <int from legal_phases>,
  "reason": "<one concise sentence>"
}
```

The output parser attempts strict JSON decoding first, then falls back to regex extraction. If either the schema is malformed, the phase index is illegal, or the call times out, SafeGAT reverts to the RL-proposed action without modification.

---

## Citation

If you use SafeGAT in your research, please cite:

```bibtex
@article{thadikamalla2025safegat,
  title     = {SafeGAT: Uncertainty-Gated and Safety-Constrained LLM Supervision
               for Graph Attention Reinforcement Learning in Traffic Signal Control},
  author    = {Thadikamalla, Saidulu and Karnam, Harshita and Joshi, Piyush},
  journal   = {Expert Systems with Applications},
  year      = {2025},
  note      = {Under review, Manuscript No. ESWA-D-26-17727}
}
```

---

## License

This project is released for academic and research use. See `LICENSE` for details.
