# NewYork SafeGAT CityFlow Adapter

This folder now has a runnable offline adapter for the modified SafeGAT stack.
It loads a NewYork roadnet/flow pair, builds the junction graph, applies the
updated risk gate, GRU anomaly forecast, corridor cache, and low-confidence
`T_i` safety fallback.

Default run uses the 28x7 dataset:

```bash
python NewYork/run_safegat_cityflow.py
```

To run the 16x3 dataset:

```bash
python NewYork/run_safegat_cityflow.py ^
  --dataset-dir NewYork/16_3 ^
  --roadnet roadnet_16_3.json ^
  --flow anon_16_3_newyork_real.json ^
  --impl-dir 4x4 ^
  --max-nodes-per-step 4
```

Outputs are written under the selected dataset folder in
`safegat_cityflow_output/`.

This is an offline audit runner, not a live CityFlow simulator controller. It is
meant to bring the modified SafeGAT logic into the real dataset folder and
exercise the new conservative CityFlow safety fallback.
