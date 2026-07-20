# MoNNA — decentralised

Demonstrates the **MoNNA** decentralised simulation
(`krum.simulations.decentralised.MonnaSimulation`): each honest worker runs a
local momentum-SGD step and then replaces its model with a nearest-neighbor
average (NNA) over the closest received models, while `f` Byzantine workers
apply an attack. The experiment here sweeps the aggregator (NNA vs a
plain, non-robust Mean) to illustrate the phenomenon behind issue #42: NNA
keeps training under attack, while Mean collapses to chance-level accuracy.

## Usage

```bash
uv run python -m experiments.decentralised.monna_icml_2023.experiment
```

Configuration lives as constants at the top of the experiment file — edit
them in place (dataset, worker/Byzantine counts, rounds, learning rate,
aggregators, partitioning).

## Files

- [run.py](run.py): `monna_experiment()` — an orchestrator experiment that
  builds one `MonnaSimulation` (datasets included, from config; aggregator
  defaults to MoNNA's own nearest-neighbor average when not overridden) and
  **pushes** per-round `train_loss`, `test_loss`, and `test_accuracy` to
  `Metric` channels.
- [experiment.py](experiment.py): a fast proof-of-concept demo — a small MLP
  on MNIST under a sign-flip attack, illustrating the NNA-vs-Mean phenomenon
  in seconds. Not a paper reproduction.

Dataset and model code is shared one level up, under
`experiments/decentralised/`, so future decentralised experiments can reuse it:

- [../datasets.py](../datasets.py): MNIST / CIFAR-10 / FakeData loading and
  the IID / Dirichlet per-worker batch streams.
- Standard models (`Monna2023SmallMnist`, `Monna2023CNNMnist`,
  `Monna2023CNNCifar10`, ...) now live under `krum.primitives.models`, shared
  across all experiments rather than duplicated per-experiment.
