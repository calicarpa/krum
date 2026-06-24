# MoNNA — decentralised

Demonstrates the **MoNNA** decentralised simulation
(`krum.simulations.decentralised.MonnaSimulation`): each honest worker runs a
local momentum-SGD step and then replaces its model with a nearest-neighbor
average over the closest received models, while a Byzantine worker applies a
sign-flip attack.

## Usage

```bash
uv run python -m experiments.decentralised.monna_icml_2023.experiment_1
```

Configuration lives as constants at the top of `experiment_1.py` — edit them in
place (dataset, worker counts, rounds, learning rate, attack, partitioning).

## Files

- [run.py](run.py): `monna_experiment()` — an orchestrator experiment that
  builds one `MonnaSimulation` (datasets included, from config) and **pushes**
  per-round `train_loss`, `test_loss`, and `test_accuracy` to `Metric` channels.
- [experiment_1.py](experiment_1.py): configuration constants + entry point;
  drives `monna_experiment` with an `Orchestrator` and prints the collected
  metrics via `orchestrator.get(...).to_pandas()`.

Dataset and model code is shared one level up, under
`experiments/decentralised/`, so future decentralised experiments can reuse it:

- [../datasets.py](../datasets.py): MNIST / FakeData loading and the IID /
  Dirichlet per-worker batch streams.
- [../models.py](../models.py): the small MNIST classifier.
