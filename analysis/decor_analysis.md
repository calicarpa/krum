# DECOR — Claude's Analysis

> Complementary to [byzantine_momentum_CLAUDE_analysis.md](byzantine_momentum_CLAUDE_analysis.md)
> (ByzantineMomentum, the baseline). This file focuses on what DECOR actually
> does — which is not Byzantine robustness — and why its relationship to
> ByzantineMomentum matters anyway.

## 1. What this repo is

- **Paper:** "The Privacy Power of Correlated Noise in Decentralized Learning."
- **Topic:** *Differential privacy* in decentralized SGD via **pairwise-canceling Gaussian noise** exchanged over a communication graph. Introduces **SecLDP** (Secure Local Differential Privacy), a relaxation of local DP that assumes pairwise shared secrets between connected nodes.
- **Authors:** Not listed in the README, file headers, or a LICENSE file. The code contains no attribution — a concrete compliance issue given that parts of it are copied from BM (see §6.1).
- **Setting:** **Decentralized + private**, not Byzantine. No attack/defense code anywhere in the repo.

## 2. Architecture

```
DECOR/
├── train.py                   (~400 lines — main training loop)
├── worker.py                  (~170 lines — Worker class, per-node state)
├── evaluator.py               (~100 lines — test-time evaluator)
├── dataset.py                 (MNIST/LibSVM/synthetic loaders, Dirichlet non-IID)
├── models.py                  (simple MNIST CNN, logistic regression, linear)
├── misc.py                    (flatten/unflatten helpers, clipping, antisymmetric noise)
├── quadratics.py              (synthetic linear-regression experiments)
├── reproduce_{libsvm,mnist}.py + tuning_{libsvm,mnist}.py
├── study.py                   (GTK-based plotting, optional)
├── tools/                     (pytorch.py, __init__.py, access.py, cluster.py, jobs.py, misc.py)
└── utils/
    ├── topology.py            (ring, grid, centralized mixing matrices)
    ├── dp_account.py          (~180 lines — RDP accounting for SecLDP)
    ├── avg_consensus_algorithms.py  (standard, finite-time, push-sum consensus)
    ├── optimizers.py          (decentralized SGD variants with correlated noise)
    ├── plotting.py, param_search.py
```

**Entry points.** [train.py](../DECOR/train.py) is the main runner (analogous to BM's `attack.py`, but without Byzantines). [reproduce_libsvm.py](../DECOR/reproduce_libsvm.py) / [reproduce_mnist.py](../DECOR/reproduce_mnist.py) are per-dataset sweep launchers. [quadratics.py](../DECOR/quadratics.py) is the synthetic-data companion.

## 3. Algorithms implemented

There are **no aggregators or attacks** here — this is not a Byzantine codebase. Instead:

**Privacy mechanisms** in [train.py](../DECOR/train.py):
- **CDP** (centralized DP): single Gaussian noise scalar `--sigma`.
- **LDP** (local DP): independent per-node Gaussian.
- **Corr** (DECOR's contribution): per-edge **antisymmetric** noise matrix `V[i,j] = -V[j,i]`, multiplied by the adjacency mask `W[i,j]` so only neighbors carry shared randomness. The summed per-node noise has reduced variance because cross-edge contributions partially cancel.

**Privacy accounting** ([utils/dp_account.py](../DECOR/utils/dp_account.py)):
- `rdp_account()` — per-iteration RDP ε, via a Laplacian + covariance linear system.
- `user_level_rdp()` — composition over iterations.

**Consensus algorithms** ([utils/avg_consensus_algorithms.py](../DECOR/utils/avg_consensus_algorithms.py)):
- Standard gossip, finite-time consensus, push-sum.

**Topology** ([utils/topology.py](../DECOR/utils/topology.py)):
- `FixedMixingMatrix` class. Topologies: ring (weight 1/3 per neighbor), 2-D grid, centralized star.

## 4. Capabilities and limitations

**Can simulate:**
- Decentralized SGD on fixed small graphs (ring, grid, star).
- Three privacy regimes (CDP, LDP, Corr) with automatic ε/δ tuning.
- User-level or example-level DP via `--privacy`.
- Dirichlet non-IID data partition.
- MNIST, LibSVM a9a (hardcoded 123 features), synthetic quadratics.
- Gradient clipping, LR decay, L2 weight decay.

**Cannot do:**
- Anything Byzantine. No attack injection, no GAR.
- Dynamic topologies (mixing matrix is fixed at init).
- True distributed execution (single process, all nodes in memory).
- Privacy *attacks* (membership inference, reconstruction) — only the defense is implemented.

## 5. Library-readiness

- No `setup.py` / `pyproject.toml`.
- No tests.
- `requirements.txt` present (torch 2.0.1, torchvision 0.15.2, numpy, scipy, pandas, matplotlib, networkx, scikit-learn, tqdm).
- **No LICENSE file** — confirmed via glob. This is a compliance issue because parts of the code are copied from an EPFL project (BM) that has a non-MIT license.
- README is brief; no API docs, no tutorials.

## 6. Comparison with ByzantineMomentum

### 6.1 Lineage evidence — partial and *unattributed*

- **[tools/pytorch.py](../DECOR/tools/pytorch.py)** contains the same `flatten`/`relink` implementation as BM, **but the author header and copyright block are stripped**. Side-by-side diff shows identical logic and docstrings.
- **[tools/__init__.py](../DECOR/tools/__init__.py)** implements the same `Context`, `UserException`, colored-print infrastructure as BM, again without the Rouault attribution or EPFL copyright.
- **No `aggregators/`, no `attacks/`, no `experiments/`** — DECOR does not use BM's domain-specific machinery. Only the lowest-level tools/ travelled across.

The pattern: **infrastructure inherited, domain code original, attribution missing.** Legally this is a problem. For the library plan it is also data: it shows that if someone repackages your tools/ module, they will strip the headers unless the license or the module makes attribution structurally necessary.

### 6.2 What this repo adds (vs. BM)

- **Pairwise-canceling correlated noise.** [misc.py](../DECOR/misc.py) builds an antisymmetric tensor `V` with `V[j,i] = -V[i,j]` over upper triangular indices, then masks by the adjacency matrix. This is the paper's technical contribution.
- **SecLDP accounting via RDP on the Laplacian.** The privacy loss is computed from `(L, Σ)` where `L = D − A` is the graph Laplacian and `Σ` is the noise covariance. Handles sparse/dense/Cholesky representations.
- **Decentralized optimizer variants** in [utils/optimizers.py](../DECOR/utils/optimizers.py) with the Worker class in [worker.py](../DECOR/worker.py). BM has no client/worker abstraction — here each node is a full object with its own model, gradient, and noise state.

### 6.3 What it drops

- All Byzantine machinery.
- The `register()` factory (no registries here).
- The 17 study metrics. DECOR logs loss, test accuracy, and empirical error only.

## 7. Takeaways for the library plan

1. **Keep `tools/` modular and separately licensed.** DECOR shows that downstream teams will copy the tools/ module independently of the domain code. If the library ships `tools/` as its own sub-package with a clear license header per file, accidental stripping becomes harder and the attribution survives forks.
2. **Privacy and Byzantine defenses are orthogonal.** DECOR proves the infrastructure (flat gradients, worker abstraction, consensus) can host DP noise alongside or instead of a GAR. The library could expose pre-aggregation hooks for noise injection so DP mechanisms plug in without the GAR contract changing.
3. **Correlated noise is a candidate primitive.** If the library wants to cover both threat models (Byzantine + curious), the antisymmetric-noise construction from `misc.py` is small, self-contained, and worth including.
4. **LICENSE hygiene matters.** ByzantineMomentum uses a proprietary EPFL license; robust-collaborative-learning switched to MIT; DECOR has no license. If the library wants adoption, MIT or Apache-2.0 from day one is the right default.
5. **SecLDP is not a competitor to byzfl.** DECOR and byzfl solve different problems. Good news for positioning: a Byzantine library does not need to absorb DECOR's privacy mechanisms, but it should not *block* them either (no monomorphic types, no "gradients must be pure" assumptions).
