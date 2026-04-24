# Byzantine-Robust-Gossip — Claude's Analysis

> Complementary to [byzantine_momentum_CLAUDE_analysis.md](byzantine_momentum_CLAUDE_analysis.md)
> (ByzantineMomentum, the baseline). This file focuses on what this repo does
> and how it relates to ByzantineMomentum's plumbing.

## 1. What this repo is

- **Paper:** "Robust Byzantine Gossip: tight breakdown point and topology aware attacks."
- **Venue:** ICML 2025 (per README).
- **Authors visible in file headers:** Sébastien Rouault (EPFL, 2018-2021, inherited files) and John Stephan (john.stephan@epfl.ch, EPFL, 2018-2023, new/updated files — e.g., [aggregators/krum.py](../Byzantine-Robust-Gossip/aggregators/krum.py)).
- **Setting:** **Decentralized gossip** — peer-to-peer, no central server. Every honest worker locally aggregates updates from a neighborhood defined by an explicit communication graph.
- **License:** EPFL DCL (copied from ByzantineMomentum).

## 2. Architecture

Top-level layout:

```
Byzantine-Robust-Gossip/
├── aggregators/                  (25 files, register() factory, EPFL 2018-2023 headers)
├── attacks/                      (12 files, same factory pattern)
├── experiments/                  (Model/Dataset/Loss/... wrappers — same as BM)
├── tools/                        (pytorch.py flatten/relink, misc.py, jobs.py, cluster.py, access.py)
├── topology.py                   (86 lines — NetworkX-based graph wrapper, NEW)
├── PeerToPeerSparseAVG.py        (760 lines — averaging-only simulator)
├── peerToPeerSparse.py           (1079 lines — full gossip SGD simulator)
├── reproduce_averaging{0,1}.py   (Two Worlds / Erdős–Rényi experiments)
├── reproduce_{mnist,cifar}*.py   (federated training reproducers)
├── study.py                      (795 lines — analysis, imports matplotlib + optional GTK)
├── analysis_{acp,mnist}.ipynb
└── config.txt                    (dependency list, not pip-usable as-is)
```

**Main entry points.** The monolithic `attack.py` of ByzantineMomentum is replaced by two script entry points — [peerToPeerSparse.py](../Byzantine-Robust-Gossip/peerToPeerSparse.py) (full gossip SGD, 1079 lines) and [PeerToPeerSparseAVG.py](../Byzantine-Robust-Gossip/PeerToPeerSparseAVG.py) (averaging only, 760 lines). Both keep the same "argparse + loop + writing CSVs" style as `attack.py`; they just loop over many honest workers' local aggregations instead of one server-side call.

## 3. Algorithms implemented

**Aggregators** (~23 registered, all via the same `register()` factory in
[aggregators/__init__.py](../Byzantine-Robust-Gossip/aggregators/__init__.py)):

- Inherited from ByzantineMomentum: `average`, `krum`, `multikrum`, `median`, `trmean`, `bulyan`, `aksel`, `cge`.
- Added in robust-collaborative-learning (the intermediate fork): `cva`, `iter_cva`, `mva`, `rfa`, `mom`, `bucketing`, `centeredclip`, `filterL2`, `mea`, `cenna`, `krum_pseudo`, `multiKrum_pseudo`.
- **NEW in this repo:** `CSplus_RG`, `GTS_RG`, `CShe_RG` ([aggregators/robust_gossip.py](../Byzantine-Robust-Gossip/aggregators/robust_gossip.py)) — topology-aware gossip GARs that adapt their clipping threshold to the per-edge Metropolis weights and declared Byzantine neighbor count. They also accept an `honest_index` argument — the receiver node — which the centralized GARs do not.
- **NEW:** `bucketing_stress` (stress variant), `ios` (iterative outlier scrubbing).

**Attacks** (12): `identical`, `identical_sparse`, `dissension`, `empire`, `nan`, `signflipping`, `scaledSF`, `labelflipping`, `mimic`, `mimic_heuristic`, `anticge`. Mostly inherited from robust-collaborative-learning.

**Gossip-specific primitive — [topology.py](../Byzantine-Robust-Gossip/topology.py).** A `CommunicationNetwork` class wraps a NetworkX graph and exposes:
- Built-in topologies: `fully_connected`, `Erdos_Renyi`, `lattice`, `two_worlds`, `random_geometric`.
- Metropolis edge weights (denominator = `max_degree + 1 + byz`) with Byzantine-tolerant adjustment.
- Laplacian spectrum + algebraic connectivity (for theoretical analysis in the paper).

## 4. Capabilities and limitations

**Can simulate:**
- Fully decentralized gossip with arbitrary NetworkX topology.
- Byzantine neighbors embedded into honest workers' neighborhoods.
- 23+ GARs including topology-aware ones.
- Mixed GAR selection via `--gars "name1,freq1;name2,freq2;…"` (same pattern as BM).
- `--nb-local-steps` local SGD between gossip rounds.
- MNIST, CIFAR-10, synthetic averaging experiments.
- Multi-GPU orchestration via `--supercharge` (tools/jobs.py style).

**Cannot do:**
- Asynchronous gossip (synchronous rounds only).
- Dynamic topology mid-run.
- Gradient compression / quantization.
- Differential privacy.
- True distributed execution — all "workers" are Python objects in one process.

**Linux-only caveat.** README states the dataset manager only runs on Linux. Averaging-only experiments (`PeerToPeerSparseAVG.py`) work on any OS.

## 5. Library-readiness

- No `setup.py` / `pyproject.toml`. Not pip-installable.
- No tests, no CI.
- Dependencies in `config.txt` (plain list, not pip-parseable).
- Docs: README is 42 lines. Inline docstrings sparse.
- Research code — reproducibility artifact, not a library.

## 6. Comparison with ByzantineMomentum

### 6.1 Lineage evidence (strong)

- **`tools/pytorch.py` is byte-compatible with BM's.** Same `flatten`/`relink` at lines 30-64, same Rouault header, same 2018-2021 copyright.
- **[aggregators/__init__.py](../Byzantine-Robust-Gossip/aggregators/__init__.py)** uses the identical `register(name, unchecked, check, upper_bound=None, influence=None)` factory (lines 42-86).
- **[attacks/__init__.py](../Byzantine-Robust-Gossip/attacks/__init__.py)** keeps BM's 2019-2021 header and same `register()` signature.
- **`experiments/`** wrappers (`Model`, `Dataset`, `Loss`, `Criterion`, `Optimizer`, `Checkpoint`, `Storage`, `Configuration`) unchanged.
- **[aggregators/krum.py](../Byzantine-Robust-Gossip/aggregators/krum.py)** header extended to 2018-2023 and lists John Stephan — one of the few files with the year bump.

### 6.2 What this repo adds

- **Gossip topology.** [topology.py](../Byzantine-Robust-Gossip/topology.py) introduces `CommunicationNetwork` (NetworkX graph + Metropolis weights + Laplacian). Nothing equivalent in BM.
- **New GAR signature.** Gossip GARs (`CSplus_RG`, `GTS_RG`, `CShe_RG`) take an additional `honest_index` argument — the index of the *receiver* performing the aggregation. In centralized mode BM's aggregators have no such concept. This is the cleanest example of how the `register()` API stretches to accommodate a new setting without being redesigned.
- **Byzantine-aware Metropolis weights.** Edge weights include a `byz` term in their denominator so that declared Byzantine neighbors do not dominate the averaging — a topology-level defense layered on top of the GAR.
- **Dual entry points** (averaging vs. full SGD) instead of BM's single `attack.py`.

### 6.3 What it drops or loses

- **No `--momentum-at worker` mode in the gossip setting** — the headline contribution of the BM paper (variance-reducing worker-side momentum) isn't the focus here.
- **Same monolithic-script problem as BM**, only bigger (1079 lines instead of 885). Library usability still absent.
- **Byzantine gradients assumed at the tail of the input list** ([aggregators/robust_gossip.py](../Byzantine-Robust-Gossip/aggregators/robust_gossip.py)) — a subtle contract narrowing that would bite anyone calling the GAR from outside the simulator.

## 7. Takeaways for the library plan

1. **The `register()` factory survives.** When this team needed to add the gossip setting they did NOT change the factory — they just added an optional `honest_index` kwarg to the aggregate function. That's evidence the contract is already flexible enough for both paradigms. Keep it.
2. **`topology.py` is the missing abstraction in BM.** If the library aims to cover both settings, lifting this module almost verbatim is the right move. NetworkX is a reasonable choice.
3. **Gossip GAR signature.** The cleanest generalization is `aggregate(gradients, f, *, honest_index=None, **kwargs)` where centralized calls leave `honest_index=None` and gossip calls set it. No rework needed for BM's existing GARs.
4. **CSplus_RG / GTS_RG / CShe_RG are must-include** if the library targets the gossip community.
5. **Watch the "Byzantine tail" convention.** Any public API should require explicit labeling of which gradients are honest vs. Byzantine rather than relying on list position.
