# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.1] — 2026-10-23

Fixes to the orchestrator's change detection: edits it missed in 0.9.0, which
silently kept a stale result, now re-run the jobs they affect.

### Fixed

- A helper module of your own that is neither the experiment's package nor
  `krum`, such as a `utils.py` next to the sweep script, is now part of a
  job's identity. Before, it was treated as a third-party dependency, and
  editing it did not re-run anything.
- Modules imported inside the experiment body, while the job runs, are
  covered too. So are the defaults and the module constants their functions
  use, and the constants or lambdas the job reads straight off them without
  calling anything.
- Native code is folded in by file content: your own extension modules, and
  libraries loaded through `ctypes`, which used to raise `HashError`.
- A module value set after `Orchestrator.run`, such as a config filled in
  before the drain, counts: keys are recomputed when a plan or drain starts.
  The key `run` returns is provisional until then.
- The installed packages are a new environment witness, so upgrading a
  dependency re-runs jobs even in a project without a `uv.lock`.

### Changed

- `owned_for` returns a `Modules` ownership test instead of a set of module
  prefixes, and `Orchestrator.runner` accepts one.
- Results recorded by 0.9.0 re-run once, since their manifests lack the new
  installed-packages witness.

## [0.9.0] — 2026-10-04

First public release. Pre-1.0: the API is usable and documented, but may still
change in response to how it gets used.

### Aggregation rules

Eleven Gradient Aggregation Rules, as stateless classmethods over a flat
gradient tensor: `Average`, `Median`, `TrimmedMean`, `Krum`, `MultiKrum`,
`Bulyan`, `Brute`, `GeometricMedian`, `Medoid`, `NearestNeighborAverage` and
`Aksel`. Each documents the paper it comes from, and the bound on `f` under
which its guarantee holds.

### Byzantine attacks

Five attacks for evaluating a rule's robustness: `ALIEAttack`,
`SignFlipAttack`, `GaussianAttack`, `FullGradientNegationAttack` and
`SmallPerturbationAttack`.

### Data partitioning

Four strategies for distributing a dataset across workers, IID or not:
`IidPartitioner`, `PerLabelsPartitioner`, `DirichletPartitioner` and
`MixingPartitioner`, which composes two others at a chosen ratio.

### Models

A `Model` wrapper exposing a `torch.nn.Module`'s parameters and gradients as
zero-copy flat tensors, which is the layout the aggregators consume, plus the
MLP and CNN architectures the reproduced papers use.

### Simulations

Training simulations reproducing the setting of each paper: `KrumSimulation`
(Blanchard et al., NIPS 2017) and `HiddenVulnerabilitySimulation` (El Mhamdi
et al., ICML 2018) in the centralised setting, and `MonnaSimulation`
(Farhadkhani et al., ICML 2023) in the decentralised one.

### Orchestration

`Orchestrator` runs an experiment over a parameter sweep and keeps each run's
output in a folder of its own, named by a key derived from the parameters *and*
from the code the run executes. A run already recorded is skipped, so changing
one aggregator re-runs the configurations that use it and leaves the rest
alone.

- Metrics are recorded through `Metric` and read back as a `MetricTable` of
  plain Python lists, with `to_pandas()`, `to_csv()` and `to_dict()` for
  whichever library you prefer. Reading metrics needs no dataframe dependency.
- Every job records a manifest: its parameters, the git commit it ran at and
  whether the tree was dirty, the resolved environment, and its timings.
- A job is re-run when its environment has changed, or when a function it
  called at runtime has since been edited. `Orchestrator.plan()` reports what
  a sweep would do, and why, without running any of it.
- `isolate=True` runs each job in a fresh interpreter, so nothing one job
  leaves behind reaches the next.

### Reproducible experiments

Three runnable studies under `experiments/`, each with a README recording the
phenomenon it demonstrates and the results observed: MultiKrum versus Mean
under sign-flip attack, Bulyan versus MultiKrum under small-perturbation
attack, and MoNNA versus Mean in the decentralised setting.

### Requirements

Python 3.12 to 3.14. `torch` and `torchvision` at runtime; `pandas`,
`matplotlib`, `numpy` and `seaborn` only for the `experiments` extra.
