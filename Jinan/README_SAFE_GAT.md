# Jinan SafeGAT CityFlow Controller

This folder now runs SafeGAT on the Jinan CityFlow dataset with the same
workflow as `4x4` and `7x28`:

1. train a GAT-DQN checkpoint on live CityFlow simulation;
2. load `models/gat_dqn_final.pt`;
3. run live CityFlow control with SafeGAT uncertainty gating, real LLM
   prompting, anomaly forecasting, corridor context, and safety shielding.

Train:

```bash
python Jinan/train_cityflow.py
```

Run live SafeGAT-LLM control:

```bash
python Jinan/run_safegat_cityflow.py
```

To use another Jinan flow:

```bash
python Jinan/train_cityflow.py --flow anon_3_4_jinan_real_2000.json
python Jinan/run_safegat_cityflow.py --flow anon_3_4_jinan_real_2000.json
```

Training writes `Jinan/3_4/models/gat_dqn_final.pt`.
Inference writes logs under `Jinan/3_4/safegat_cityflow_live_output/`.

The live runner requires the CityFlow Python package and the same LLM config
used by the `4x4` SafeGAT implementation.
