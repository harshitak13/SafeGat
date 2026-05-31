# NewYork SafeGAT CityFlow Controller

This folder now runs SafeGAT on the NewYork CityFlow dataset with the same
workflow as `4x4` and `7x28`:

1. train a GAT-DQN checkpoint on live CityFlow simulation;
2. load `models/gat_dqn_final.pt`;
3. run live CityFlow control with SafeGAT uncertainty gating, real LLM
   prompting, anomaly forecasting, corridor context, and safety shielding.

Default 28x7 training:

```bash
python NewYork/train_cityflow.py
```

Default 28x7 live SafeGAT-LLM control:

```bash
python NewYork/run_safegat_cityflow.py
```

To run the 16x3 dataset:

```bash
python NewYork/train_cityflow.py ^
  --dataset-dir NewYork/16_3 ^
  --roadnet roadnet_16_3.json ^
  --flow anon_16_3_newyork_real.json ^
  --impl-dir 4x4 ^
  --gat-heads 4

python NewYork/run_safegat_cityflow.py ^
  --dataset-dir NewYork/16_3 ^
  --roadnet roadnet_16_3.json ^
  --flow anon_16_3_newyork_real.json ^
  --impl-dir 4x4 ^
  --gat-heads 4
```

Training writes `NewYork/28_7/models/gat_dqn_final.pt` by default.
Inference writes logs under `NewYork/28_7/safegat_cityflow_live_output/`.

The live runner requires the CityFlow Python package and the same LLM config
used by the `7x28` SafeGAT implementation.
