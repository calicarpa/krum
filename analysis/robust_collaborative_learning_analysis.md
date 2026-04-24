# robust-collaborative-learning — Claude's Analysis

> Complementary to [byzantine_momentum_CLAUDE_analysis.md](byzantine_momentum_CLAUDE_analysis.md)
> (ByzantineMomentum, the baseline). **This is the cleanest example of how
> ByzantineMomentum got extended the "right" way** — nothing ripped out,
> everything additive. Worth studying as a template.

## 1. What this repo is

- **Paper:** ICML 2023 (per README).
- **Authors visible in headers:** Sébastien Rouault (EPFL, 2018-2021, inherited files) and **John Stephan** (john.stephan@epfl.ch, EPFL, 2021-2023, new code — this is the first sibling repo where Stephan appears prominently).
- **License:** **MIT** (2023, EPFL DCL) — upgrade from BM's proprietary license.
- **Setting:** Both **centralized (parameter server)** and **decentralized (gossip)** in the same codebase. The gossip path is introduced by `peerToPeer.py`.

## 2. Architecture

Directly mirrors ByzantineMomentum's layout, with additions:

```
robust-collaborative-learning/
├── aggregators/                 (~22 GARs — BM's 9 + 13 new)
├── attacks/                     (~9 attacks — BM's 4 + 5 new)
├── experiments/                 (Model/Dataset/Loss/... — unchanged from BM)
├── tools/                       (pytorch.py flatten/relink — unchanged)
├── peerToPeer.py                (~1000+ lines — NEW gossip training loop, John Stephan 2021-2022)
├── reproduce.py                 (MNIST centralized reproducer)
├── reproduce_cifar.py           (CIFAR-10 reproducer)
└── study.py                     (same analysis framework as BM)
```

**Note on the missing `attack.py`.** BM's centralized training loop is not at the top level here; users enter via [reproduce.py](../robust-collaborative-learning/reproduce.py) (which internally invokes training configurations) or [peerToPeer.py](../robust-collaborative-learning/peerToPeer.py) for gossip. The centralized training code was either refactored into `reproduce.py` helpers or moved — worth double-checking if you port.

## 3. Aggregators — what's added vs. ByzantineMomentum

**Inherited (same files as BM):** `average`, `krum`, `bulyan`, `median`, `trmean` (with `phocas` / `meamed` variants), `aksel`, `brute`, `cge`.

**NEW (13 files), all by John Stephan 2021-2023:**

- **`cva.py`** — **C**losest **V**ector **A**ggregation. Selects the `n-f` gradients closest to a pivot (another worker's gradient in P2P; the output of a helper GAR in PS), then averages. Dual-mode by design.
- **`iter_cva.py`** — Iterative CVA.
- **`mva.py`** — **M**inimum **V**ariance **A**veraging. Exhaustively enumerates `C(n, n-f)` subsets, picks the one with minimum variance. Correct but expensive; practical only for small `n`.
- **`rfa.py`** — **R**obust **F**ederated **A**veraging (geometric median via smoothed Weiszfeld). Configurable `T` (iterations) and `nu` (smoothing).
- **`mom.py`** — **M**edian-**o**f-**M**eans. Partitions gradients into buckets, averages each, applies RFA on the bucket averages.
- **`multikrum.py`, `multiKrum_pseudo.py`, `krum_pseudo.py`** — Variants/pseudo-versions of Krum and MultiKrum for experimental comparison.
- **`bucketing.py`, `bucketing_stress.py`** — Bucketing (group-averaging) pre-aggregation.
- **`filterL2.py`** — Eigenvalue filtering. Iteratively removes the gradient with the largest projection onto the top eigenvector of the covariance matrix. Pulls in scipy.
- **`centeredclip.py`** — Centered clipping (Karimireddy et al., ICML 2021). Iteratively clips differences from the current aggregate.
- **`cenna.py`, `mea.py`** — Additional experimental variants.

All new GARs use the same `register(name, unchecked, check, upper_bound=None, influence=None)` factory — zero API drift.

## 4. Attacks — what's added

**Inherited:** `nan`, `empire`, `empire-strict`, `anticge`, `bulyan` / `little` (both in `identical.py`).

**NEW:**
- **`labelflipping.py`** — Byzantines average honest workers' gradients computed on **label-flipped** batches. Requires honest cooperation (`--flip` flag tells honest workers to compute flipped gradients that Byzantines can then aggregate).
- **`mimic.py`** — Naïve mimicry: replicate the first honest worker's gradient `f_real` times.
- **`mimic_heuristic.py`** — Adaptive mimicry with a **learning phase** (default 100 steps). During the phase, `z` and `mu` vectors are updated based on cumulative distance-weighted gradient similarity; after the phase, the attack locks onto the "best" worker to impersonate. Spiritual predecessor of SignGuard's adaptive-γ attacks.
- **`signflipping.py`** — Negation: average honest gradients and flip sign.
- **`scaledSF.py`** — Scaled sign-flipping (hyperparameter on the scale factor).

## 5. Gossip extension — [peerToPeer.py](../robust-collaborative-learning/peerToPeer.py)

Header:
```python
# @file   peerToPeer.py
# @author John Stephan <john.stephan@epfl.ch>
# Copyright © 2021-2022 École Polytechnique Fédérale de Lausanne (EPFL).
```

**Design of the gossip loop** (approx. lines 840-1040):

1. Each honest worker `i` maintains its own parameter vector `honest_thetas[i]` (no shared server model).
2. Per step, each worker samples a local batch (possibly from its own heterogeneous dataset under `--hetero` + `--dirichlet-alpha`) and computes a gradient.
3. **Momentum is applied at the worker** (the paper's contribution, inherited verbatim from BM): `grad_momentum_workers[i] ← μ·grad_momentum_workers[i] + (1-d)·g_i`.
4. For each worker `j`, a **random subset of `n-2f-1` honest peers** is sampled, and **all `f` Byzantines** are appended. The defense GAR aggregates this bucket, the result becomes `honest_thetas[j]`.
5. If the attack is `mimic_heuristic`, `z` and `mu` update afterward.

**Key properties of this gossip model:**
- **Topology is implicit and random** — re-sampled every round. No NetworkX graph, no fixed edges. Byzantine-Robust-Gossip (a later fork) moved to an explicit NetworkX topology; this repo stays with uniform random sampling.
- **All `f` Byzantines are included in every neighborhood.** Worst-case threat model: the adversary saturates every local aggregation.
- **Heterogeneous data is first class.** `dataset_worker[i]` is passed to `model_sample_peer()`. BM's honest workers all sample from the same pool.

## 6. Capabilities and limitations

**Beyond BM:**
- Decentralized gossip with random per-round topology.
- Heterogeneous data (Dirichlet non-IID).
- Label flipping with honest cooperation.
- Learning-phase adaptive mimic.
- Coordinate descent / sparse updates (`--coordinates`, `--nb-params`).
- **Momentum Variance Reduction** (MVR, `--mvr`) — tracks both current and previous gradients plus their difference, so only active if `momentum > 0`.
- **Jungle algorithm** (`--jungle`) — forces batch size 1, zeros momentum/dampening, requires heterogeneous data. Narrow mode from a cited paper.
- Gaussian noise privacy (`--privacy`) — ad-hoc, no formal DP accounting.

**Limitations:**
- Single-process simulator (same as BM).
- Random-only gossip topology — no structured graph support.
- No `setup.py`, no tests.
- README is ~67 lines; most documentation lives in argparse help strings.

## 7. Library-readiness

- No `setup.py` / `pyproject.toml`. Not pip-installable.
- No tests.
- README is minimal (~67 lines, Python 3.7.3, torch 1.6.0).
- Dependencies auto-detected via `tools/misc.py::get_loaded_dependencies()` — same as BM.
- scipy added as a new dependency for `filterL2.py` but not listed in the README.

## 8. Comparison with ByzantineMomentum

### 8.1 Lineage evidence — *ironclad*

- **`aggregators/__init__.py`** header: "Copyright © 2018-2021 École Polytechnique Fédérale de Lausanne (EPFL), @author Sébastien Rouault." Identical to BM.
- **`tools/pytorch.py`** lines 30-64 are **character-for-character identical** to BM's `flatten`/`relink`.
- **`experiments/model.py`, `experiments/checkpoint.py`** etc. keep the Rouault 2019-2021 copyright — wrapper layer essentially unchanged.
- **`register()` factory is unchanged** — same signature, same `checked` / `unchecked` / `upper_bound` / `influence` members.
- **New files bump to 2022 or 2023** and credit John Stephan. `filterL2.py` has 2018-2023, showing it was updated multiple times.
- **License changes from EPFL DCL to MIT** in 2023 — the only file where the license block differs substantively.

### 8.2 What this repo adds vs. BM

Everything in §3, §4, §5 above, plus:
- First sibling to introduce the gossip setting.
- First to support non-IID data distribution.
- First to introduce adaptive mimicry with a learning phase.
- First with coordinate-descent / sparse updates.
- First with MVR and Jungle.

### 8.3 What it drops

**Nothing.** That's the notable part. Every BM GAR, attack, tool, and wrapper is still present; the extensions are strictly additive. This is the only sibling in the folder with this property.

## 9. Takeaways for the library plan

1. **This is the proof that the BM contract was future-proof.** Adding 13 GARs, 5 attacks, non-IID data, heterogeneous workers, MVR, Jungle, and a whole gossip training loop did not require a single change to the `register()` factory, the `flatten`/`relink` helpers, or the `experiments/` wrappers. That is strong evidence against redesigning those abstractions for the library.
2. **`peerToPeer.py` is the template gossip entry point.** Library gossip support can be delivered as an alternative entry point that reuses the same `aggregators/` and `attacks/` registries. No refactoring needed.
3. **Random vs. fixed-topology gossip is a real design axis.** This repo picked random per-round, [byzantine_robust_gossip_analysis.md](byzantine_robust_gossip_analysis.md) picked fixed NetworkX. Both are valid. A library should accept either — a topology sampler callable (see the takeaways in the Byzantine-Robust-Gossip analysis).
4. **The 13 new GARs are natural "batteries included" candidates.** `cva`, `mva`, `rfa`, `mom`, `multikrum`, `bucketing`, `filterL2`, `centeredclip` — include these and most downstream research doesn't have to reimplement anything.
5. **Adaptive mimicry deserves a first-class slot.** The `mimic_heuristic` learning-phase pattern generalizes to "attacks that adapt over time" and prefigures SignGuard's `adaptive_attack.py`. Worth providing as a composable attack primitive.
6. **License precedent.** The MIT switch happened here (2023). If the library adopts MIT from day one, the provenance story is: ByzantineMomentum (proprietary) → robust-collaborative-learning (MIT) → this library (MIT). Clean.
7. **Attribution is the positive model.** New files credit Stephan with new years, inherited files keep Rouault headers untouched. No stripping, no re-dating. A CONTRIBUTING.md for the library can cite this repo as the reference for how contributors update headers.
