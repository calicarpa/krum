# RPEL-BF2D — Claude's Analysis

> Complementary to [byzantine_momentum_CLAUDE_analysis.md](byzantine_momentum_CLAUDE_analysis.md)
> (ByzantineMomentum, the baseline). This file focuses on how RPEL-BF2D's
> randomized peer-to-peer design differs from BM's parameter-server world.

## 1. What this repo is

- **Paper:** "Byzantine Epidemic Learning: Breaking the Communication Barrier in Robust Collaborative Learning."
- **Authors visible in headers:** John Stephan (john.stephan@epfl.ch, EPFL, 2023) for the new training loops and gossip code; Sébastien Rouault (EPFL, 2018-2021) for the inherited `tools/pytorch.py`.
- **Setting:** **Decentralized peer-to-peer with randomized dynamic topology.** Each honest worker at each step picks `nb_neighbors` other workers at random and aggregates their updates. No server, no fixed graph.
- **License:** EPFL DCL (same family as BM).

## 2. Architecture

```
RPEL-BF2D/
├── run.py                    (133 lines — CIFAR-10 launcher, uses tools.Jobs)
├── mnist_run.py              (MNIST launcher)
├── train_p2p.py              (~388 lines — random-topology P2P trainer, John Stephan 2023)
├── fx_train_p2p.py           (~388 lines — fixed-graph variant)
├── fx_{mnist,cifar10}_run.py (fixed-graph baselines, large reproducers)
├── simulations.py            (hypergeometric b̂ analysis, paper Fig. 3)
├── b_hat.py, comp_plot.py    (analysis helpers)
├── src/
│   ├── worker_p2p.py         (main P2P worker logic)
│   ├── fx_worker_p2p.py      (fixed-graph worker)
│   ├── byzWorker.py          (Byzantine worker class + DecByzantineWorker variant)
│   ├── byz_attacks.py        (8 attacks, dict-based)
│   ├── robust_aggregators.py (16 GARs + RobustAggregator class, dict-based)
│   ├── robust_summations.py  (CS+, GTS, CS-He — gossip-specific clipping)
│   ├── topo.py               (CommunicationNetwork, Metropolis weights)
│   ├── models.py, dataset.py, misc.py, study.py
├── tools/                    (pytorch.py with EPFL header, misc.py, jobs.py, cluster.py, access.py)
└── utils/                    (gossip.py with LaplacianGossipMatrix, torch_tools.py, conversion.py)
```

**Entry points.** [train_p2p.py](../RPEL-BF2D/train_p2p.py) (random topology, 388 lines) and [fx_train_p2p.py](../RPEL-BF2D/fx_train_p2p.py) (fixed graph, 388 lines). Both are comparable in size to BM's `attack.py` but split by topology model.

## 3. Algorithms implemented

**Aggregators** (16, stored as a plain dict in [src/robust_aggregators.py](../RPEL-BF2D/src/robust_aggregators.py) — *not* the `register()` factory):
- `average`, `trmean`, `median`, `geometric_median`, `krum`, `multi_krum`, `nearest_neighbor_mixing` (NNM), `bucketing`, `pseudo_multi_krum`, `centered_clipping` (CC), `minimum_diameter_averaging` (MDA), `minimum_variance_averaging` (MVA), `monna`, `meamed`, `server_clip`.
- Legacy variants: `krum_old`, `nnm_old`.

Access pattern: `robust_aggregators[name](aggregator_obj, vectors)` via the `RobustAggregator` class.

**Attacks** (8, same dict pattern in [src/byz_attacks.py](../RPEL-BF2D/src/byz_attacks.py)):
- `SF` (sign flip), `LF` (label flip), `FOE` (fall of empires), `ALIE` (A Little Is Enough), `mimic`, `auto_ALIE`, `auto_FOE`, `inf`.

**Gossip-specific — [src/robust_summations.py](../RPEL-BF2D/src/robust_summations.py):**
- `CS+` (consensus + adaptive clipping)
- `GTS` (gradient-trim-then-sum → NNA)
- `CS-He` (variant from He et al. 2022)
Used in fixed-graph mode via `methods_dict` in [src/fx_worker_p2p.py](../RPEL-BF2D/src/fx_worker_p2p.py).

**Topology** ([src/topo.py](../RPEL-BF2D/src/topo.py)):
- `CommunicationNetwork` class, NetworkX-backed.
- Topologies: `fully_connected`, `Erdos_Renyi`, `lattice`, `two_worlds`, `random_geometric`.
- Metropolis (max-degree based) and unitary mixing weights.

## 4. Capabilities and limitations

**Can simulate:**
- **Randomized dynamic topology** (the paper's headline): each step, every worker picks `nb_neighbors` other workers uniformly at random.
- **`b̂` adaptive clipping** — `--b-hat` is the number of Byzantines expected in any single worker's neighborhood; it drives the local clipping threshold.
- Fixed-graph gossip (alternative mode) on the five topologies above.
- Worker-side momentum (default `μ = 0.99`).
- Local SGD steps between gossip rounds (`nb_local_steps`).
- Heterogeneous data via `--dirichlet-alpha`.
- Per-worker worst-case accuracy tracking.

**Cannot do:**
- Asynchronous gossip.
- Differential privacy.
- True distributed execution.
- Aggregator/attack registration via factory — adding a new GAR requires editing `robust_aggregators.py` directly.

## 5. Library-readiness

- No `setup.py` / `pyproject.toml`.
- No `requirements.txt` — README states "we do not yet provide a requirements.txt" and suggests the user install whatever PyTorch was current "as of September 25th, 2025."
- No tests, no CI.
- README minimal; docstrings sparse. Inline comments marked `JS:` (John Stephan) and `SY:` help with archaeology.

## 6. Comparison with ByzantineMomentum

### 6.1 Lineage evidence — *partial*

- **`tools/pytorch.py` keeps the Rouault 2018-2021 header and the `flatten`/`relink` trick untouched.**
- **No `aggregators/` or `attacks/` directory** at the top level — those concepts moved into `src/` as plain modules.
- **`register()` factory is gone.** GARs and attacks are stored in plain Python dicts (`robust_aggregators`, `byzantine_attacks`). Adding a new one = editing the dict file.
- **No `upper_bound()` or `influence()` methods** on aggregators — the introspection hooks BM exposed are absent.
- `experiments/` wrapper directory is **not present**. Models, datasets, loss, and optimizer are simpler free-standing modules in `src/`.

### 6.2 What this repo adds

- **Randomized per-step topology.** `--nb-neighbors` controls how many peers each worker samples. Key difference vs. a fixed-graph gossip model.
- **`b̂` abstraction.** The declared bound on Byzantines *inside a single worker's neighborhood*, distinct from the global f. Drives clipping thresholds in CS+, GTS, CS-He.
- **CG+ training mode.** An alternative to the generic `RobustAggregator` path, selected with `--rag=False`. Clips *difference vectors* rather than raw gradients.
- **DecByzantineWorker** ([src/byzWorker.py](../RPEL-BF2D/src/byzWorker.py)) — Byzantines in the decentralized setting compute attacks as negatives of weighted differences, so they push the consensus *away* rather than a global average.
- **Communication LR schedule.** `communication_lr = 1/(current_step//250 + 1)` decays the gossip mixing every 250 steps — not present in BM.
- **Per-worker worst-case evaluation.** Tracks each worker's accuracy and reports the worst trajectory. BM only reports global model accuracy.

### 6.3 What it drops

- The `register()` factory.
- `upper_bound()` / `influence()` introspection.
- The `experiments/` wrapper layer (Model, Dataset, Loss, Criterion, Optimizer, Checkpoint, Storage classes).
- The 17 study metrics. Only loss and accuracy are logged here.

## 7. Takeaways for the library plan

1. **Dropping the `register()` factory was a deliberate choice, not an oversight.** A single author (John Stephan) writing a single paper's code found the factory heavier than useful — dicts were enough. That's feedback about the factory's real cost.
2. **`b̂` is a nice first-class abstraction.** In centralized BM, the declared Byzantine count `f_decl` is global. In gossip it's local to a neighborhood. If the library wants to cover both settings, `f_decl` should be accompanied by a `f_local` (or `b_hat`) that gossip GARs can use.
3. **Randomized vs. fixed topology is a real axis.** [byzantine_robust_gossip_analysis.md](byzantine_robust_gossip_analysis.md) picked fixed NetworkX graphs; RPEL-BF2D picked random per-step neighborhoods. Both are valid. The library should accommodate both — a topology sampler callable (`Callable[[step], Neighborhood]`) rather than a static adjacency matrix.
4. **The `RobustAggregator` wrapper class is a nice middle ground.** It carries per-worker state (previous momentum, aggregation count) while delegating the math to a plain function. That pattern is byzfl-flavored (class wraps state) without forcing every GAR to be a class.
5. **Attribution is clean.** RPEL-BF2D keeps EPFL headers, cites 2023 in new files, credits Stephan — this is what decent downstream practice looks like. Worth citing as the positive example in a CONTRIBUTING guide.
