# MoNNA — decentralised

Demonstrates the **MoNNA** decentralised simulation
(`krum.simulations.decentralised.MonnaSimulation`): each honest worker runs a
local momentum-SGD step and then replaces its model with a nearest-neighbor
average (NNA) over the closest received models, while `f` Byzantine workers
apply an attack. All three experiments here sweep the aggregator (NNA vs a
plain, non-robust Mean) to illustrate the phenomenon behind issue #42: NNA
keeps training under attack, while Mean collapses to chance-level accuracy.

## Usage

```bash
uv run python -m experiments.decentralised.monna_icml_2023.small_experiment
uv run python -m experiments.decentralised.monna_icml_2023.experiment_mnist
uv run python -m experiments.decentralised.monna_icml_2023.experiment_cifar
```

Configuration lives as constants at the top of each experiment file — edit
them in place (dataset, worker/Byzantine counts, rounds, learning rate,
aggregators, partitioning).

## Files

- [run.py](run.py): `monna_experiment()` — an orchestrator experiment that
  builds one `MonnaSimulation` (datasets included, from config; aggregator
  defaults to MoNNA's own nearest-neighbor average when not overridden) and
  **pushes** per-round `train_loss`, `test_loss`, and `test_accuracy` to
  `Metric` channels.
- [small_experiment.py](small_experiment.py): a fast proof-of-concept demo —
  a small MLP on MNIST under a sign-flip attack, illustrating the NNA-vs-Mean
  phenomenon in seconds. Not a paper reproduction.
- [experiment_mnist.py](experiment_mnist.py): reproduces the MNIST column of
  Table 2 (Appendix D.2) of the ICML 2023 paper — the paper's own CNN
  (`Monna2023CNNMnist`), learning rate, batch size, momentum, weight decay,
  worker/Byzantine counts (`n=26, f=5`), and round count (`T=600`), sweeping
  the paper's tested data-heterogeneity values (`alpha in {0.5, 1, 5}`)
  against the ALIE and sign-flip attacks (exact matches for the paper's
  "little" and "signflipping"; "empire" and "labelflipping" are not covered —
  see the module docstring for why).
- [experiment_cifar.py](experiment_cifar.py): same as `experiment_mnist.py`
  but for the CIFAR-10 column of Table 2 (`Monna2023CNNCifar10`, `n=16, f=3`,
  `T=2000`, `alpha=5`).

Dataset and model code is shared one level up, under
`experiments/decentralised/`, so future decentralised experiments can reuse it:

- [../datasets.py](../datasets.py): MNIST / CIFAR-10 / FakeData loading and
  the IID / Dirichlet per-worker batch streams.
- Standard models (`Monna2023SmallMnist`, `Monna2023CNNMnist`,
  `Monna2023CNNCifar10`, ...) now live under `krum.primitives.models`, shared
  across all experiments rather than duplicated per-experiment.
