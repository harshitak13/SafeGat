# Jinan SafeGAT CityFlow Adapter

This folder now has a runnable offline adapter for the modified SafeGAT stack.
It loads `3_4/roadnet_3_4.json` and a CityFlow flow file, builds the junction
graph, applies the updated risk gate, GRU anomaly forecast, corridor cache, and
low-confidence `T_i` safety fallback.

Run:

```bash
python Jinan/run_safegat_cityflow.py
```

To use a different Jinan flow:

```bash
python Jinan/run_safegat_cityflow.py --flow anon_3_4_jinan_real_2000.json
python Jinan/run_safegat_cityflow.py --flow anon_3_4_jinan_real_2500.json
```

Outputs are written under `Jinan/3_4/safegat_cityflow_output/`.

This is an offline audit runner, not a live CityFlow simulator controller. It is
meant to bring the modified SafeGAT logic into the real dataset folder and
exercise the new conservative CityFlow safety fallback.
