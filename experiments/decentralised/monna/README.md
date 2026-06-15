# MoNNA — decentralised

Demonstrates the **MoNNA** decentralised simulation
(`krum.simulations.decentralised.MonnaSimulation`): each honest worker runs a
local momentum-SGD step and then replaces its model with a nearest-neighbor
average over the closest received models, while a Byzantine worker applies a
sign-flip attack.

## Usage

```bash
uv run python -m experiments.decentralised.monna.experiment_1
```

Configuration lives as constants at the top of `experiment_1.py` — edit them in
place (dataset, worker counts, rounds, learning rate, attack, partitioning).

## Files

- [run.py](run.py): `run_monna_simulation()` — builds one `MonnaSimulation`,
  trains it, and **returns** per-round `(train_loss, test_loss, test_accuracy)`
  records for the caller to print, plot, or assert on.
- [experiment_1.py](experiment_1.py): configuration constants + entry point;
  prints the metrics returned by `run.py`.

Dataset and model code is shared one level up, under
`experiments/decentralised/`, so future decentralised experiments can reuse it:

- [../datasets.py](../datasets.py): MNIST / FakeData loading and the IID /
  Dirichlet per-worker batch streams.
- [../models.py](../models.py): the small MNIST classifier.
