# Hangzhou SafeGAT CityFlow Adapter

This folder now has a runnable offline adapter for the modified SafeGAT stack.
It loads `4_4/roadnet_4_4.json` and a CityFlow flow file, builds the junction
graph, applies the updated risk gate, GRU anomaly forecast, corridor cache, and
low-confidence `T_i` safety fallback.

Run:

```bash
python Hangzhou/run_safegat_cityflow.py
```

To use a different Hangzhou flow:

```bash
python Hangzhou/run_safegat_cityflow.py --flow anon_4_4_hangzhou_real_5734.json
python Hangzhou/run_safegat_cityflow.py --flow anon_4_4_hangzhou_real_5816.json
```

Outputs are written under `Hangzhou/4_4/safegat_cityflow_output/`.

This is an offline audit runner, not a live CityFlow simulator controller. It is
meant to bring the modified SafeGAT logic into the real dataset folder and
exercise the new conservative CityFlow safety fallback.
