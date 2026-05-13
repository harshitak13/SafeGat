# SafeGAT-iLLM — Complete Hyperparameter Table

| Hyperparameter | Symbol | Value | Source | Description |
|----------------|--------|-------|--------|-------------|
| **GAT-DQN Network Architecture** | | | | |
| Observation dim | d | `8` | `train.py → OBS_DIM` | Feature vector dimension per node (junction) |
| Hidden dim | H | `64` | `train.py → HIDDEN_DIM` | GAT hidden / embedding size |
| GAT attention heads | K_h | `4` | `train.py → GAT_HEADS` | Multi-head attention heads in GATQNetwork |
| Number of actions | A | `4` | `network/net_config.py → NUM_ACTIONS` | Discrete phase choices per junction |
| Number of junctions | N | `12` | `network/net_config.py → NUM_NODES` | Controlled junctions in 4×4 SUMO grid |
| **DQN Training** | | | | |
| Total episodes | — | `100` | `train.py → TOTAL_EPISODES` | Full training episodes |
| Max steps per episode | T | `1800` | `train.py → MAX_STEPS` | Sim seconds per episode (30-min horizon) |
| Learning rate | α | `1e-3` | `train.py → LR` | Adam optimiser learning rate |
| Discount factor | γ | `0.95` | `train.py → GAMMA` | Bellman discount factor |
| Batch size | B | `64` | `train.py → BATCH_SIZE` | Replay mini-batch size |
| Buffer capacity | — | `50 000` | `train.py → BUFFER_CAPACITY` | Circular replay buffer size |
| Warmup steps | — | `500` | `train.py → WARMUP_STEPS` | Buffer fill before first gradient update |
| Target net update freq | — | `500` | `train.py → TARGET_UPDATE_FREQ` | Gradient steps between hard target-net syncs |
| Gradient clip norm | — | `10.0` | `train.py → GRAD_CLIP` | Max gradient L2-norm before clipping |
| Checkpoint frequency | — | `25 eps` | `train.py → CHECKPOINT_FREQ` | Episodes between model checkpoints |
| **ε-Greedy Exploration** | | | | |
| ε start | ε_0 | `1.0` | `train.py → EPSILON_START` | Initial exploration rate (fully random) |
| ε end | ε_∞ | `0.05` | `train.py → EPSILON_END` | Minimum exploration rate (near-greedy) |
| ε decay steps | T_ε | `25 000` | `train.py → EPSILON_DECAY_STEPS` | Steps for linear ε decay to ε_∞ |
| **LLM Intervention Gate  (InterventionGate)** | | | | |
| Confidence threshold | τ_c | `0.05` | `configs/safegat_llm.yaml → confidence_threshold` | Call LLM when Q-margin Δ_i = Q(a*) − Q(a_2nd) < τ_c |
| Intervention budget | K | `1600` | `configs/safegat_llm.yaml → intervention_budget` | Max LLM API calls for the full inference episode |
| Max nodes per step | K_step | `2` | `configs/safegat_llm.yaml → max_nodes_per_step` | Top-K junctions sent to LLM per simulation step |
| Anomaly weight | w_a | `1.0` | `llm/intervention_gate.py default` | Gate score weight for anomaly tag count |
| Corruption weight | w_c | `1.0` | `llm/intervention_gate.py default` | Gate score weight for corrupted observation flag |
| Low-conf weight | w_l | `1.0` | `llm/intervention_gate.py default` | Gate score weight for low-confidence flag |
| **Safety Shield  (SafetyShield)** | | | | |
| Min green hold steps | T_green | `3` | `configs/safegat_llm.yaml → min_green_hold` | Min steps a green phase must remain active before switching |
| Yellow phase indices | {1, 3} | `—` | `llm/safety_shield.py → _YELLOW_PHASES` | Phase indices treated as yellow; switching away is blocked |
| **Scenario Detector  (ScenarioDetector)** | | | | |
| Queue-spike threshold | θ_q | `0.85` | `configs/safegat_llm.yaml → queue_spike_threshold` | Occupancy above this → 'queue_spike' anomaly tag |
| Zero-fraction threshold | θ_z | `0.9` | `configs/safegat_llm.yaml → zero_fraction_corruption_threshold` | Fraction of zero obs features above this → 'corrupted' flag |
| Anomaly triggers | — | `['emergency_vehicle', 'accident_flag', 'possible_packet_loss', 'nan_observation', 'queue_spike']` | `configs/safegat_llm.yaml → anomaly_triggers` | Tags that always open the gate regardless of confidence |
| **LLM Backend  (LLMGateway)** | | | | |
| LLM timeout | — | `20 s` | `configs/safegat_llm.yaml → llm_timeout` | Seconds per API call before abort |
| Min call interval | — | `4.0 s` | `configs/safegat_llm.yaml → min_call_interval_s` | Rate-limiter: minimum gap between consecutive API calls |
| Fallback to RL | — | `True` | `configs/safegat_llm.yaml → fallback_to_rl` | Use RL action if LLM errors or times out |
| LLM mode | — | `selective` | `configs/safegat_llm.yaml → mode` | selective \| always \| never |
