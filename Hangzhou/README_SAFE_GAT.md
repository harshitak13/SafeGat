# Hangzhou SafeGAT CityFlow Controller

This folder now runs SafeGAT on the Hangzhou CityFlow dataset with the same
workflow as `4x4` and `7x28`:

1. train a GAT-DQN checkpoint on live CityFlow simulation;
2. load `models/gat_dqn_final.pt`;
3. run live CityFlow control with SafeGAT uncertainty gating, real LLM
   prompting, anomaly forecasting, corridor context, and safety shielding.

Train:

```bash
python Hangzhou/train_cityflow.py
```

Run live SafeGAT-LLM control:

```bash
python Hangzhou/run_safegat_cityflow.py
```

To use another Hangzhou flow:

```bash
python Hangzhou/train_cityflow.py --flow anon_4_4_hangzhou_real_5734.json
python Hangzhou/run_safegat_cityflow.py --flow anon_4_4_hangzhou_real_5734.json
```

Training writes `Hangzhou/4_4/models/gat_dqn_final.pt`.
Inference writes logs under `Hangzhou/4_4/safegat_cityflow_live_output/`.

The live runner requires the CityFlow Python package and the same LLM config
used by the `4x4` SafeGAT implementation.
