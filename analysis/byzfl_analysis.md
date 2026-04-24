# byzfl — Claude's Analysis

> Complementary to [byzantine_momentum_CLAUDE_analysis.md](byzantine_momentum_CLAUDE_analysis.md)
> (ByzantineMomentum, the baseline). **This is the most important sibling**
> for any library-ization effort: byzfl is the existing pip-installable
> EPFL+INRIA library. Any new library has to justify its existence against
> this one.

## 1. What this repo is

- **Project:** ByzFL (Byzantine Federated Learning), official library.
- **Paper:** "ByzFL: Research Framework for Robust Federated Learning" — González, Guerraoui, Pinot, Rizk, Stephan, Taïani. arXiv 2505.24802 (2025).
- **Affiliations:** DCL (EPFL) + WIDE (INRIA Rennes).
- **License:** MIT (Copyright 2024 EPFL). Clean, permissive — upgrade from BM's proprietary EPFL DCL license.
- **Framework:** Supports both **PyTorch tensors** and **NumPy arrays** with a unified interface.
- **Setting:** Centralized federated learning (parameter-server). Gossip is not a first-class setting here.

## 2. Architecture

```
byzfl/
├── byzfl/
│   ├── __init__.py
│   ├── aggregators/
│   │   ├── __init__.py
│   │   ├── aggregators.py         (~1441 lines — 13 robust aggregators, class-based)
│   │   └── preaggregators.py      (~481 lines — 4 pre-aggregators)
│   ├── attacks/
│   │   ├── __init__.py
│   │   └── attacks.py             (~1160 lines — 9 Byzantine attacks, class-based)
│   ├── fed_framework/
│   │   ├── server.py              (Server class)
│   │   ├── client.py              (honest Client — client-side momentum lives here)
│   │   ├── byzantine_client.py    (ByzantineClient)
│   │   ├── data_distributor.py    (IID, Dirichlet non-IID, gamma similarity)
│   │   ├── model_base_interface.py (shared base for Server/Client)
│   │   ├── models.py              (9 built-in models: MNIST/CIFAR CNNs, ResNet18-152)
│   │   └── robust_aggregator.py   (composes pre-aggregators + main aggregator)
│   ├── utils/
│   │   ├── torch_tools.py, conversion.py, misc.py
│   └── benchmark/
│       ├── benchmark.py           (combo generation + multiprocessing)
│       ├── train.py               (single-scenario training loop)
│       ├── managers.py
│       └── evaluate_results.py    (hyperparameter selection)
├── tests/                         (5 unit test files)
├── docs/                          (Sphinx)
├── README.md                      (quick-start for PyTorch and NumPy)
└── LICENSE.txt                    (MIT, 2024 EPFL)
```

**Quick-start pattern** (from README):

```python
from byzfl import Client, Server, ByzantineClient, DataDistributor

server = Server({
    "aggregator_info": {"name": "TrMean", "parameters": {"f": 1}},
    ...
})
honest_clients = [Client({...}) for _ in range(3)]
byz_client = ByzantineClient({
    "name": "InnerProductManipulation",
    "f": 1,
    "parameters": {"tau": 3.0},
})

for step in range(nb_steps):
    honest_gradients = [c.get_flat_gradients_with_momentum() for c in honest_clients]
    byz_vectors = byz_client.apply_attack(honest_gradients)
    server.update_model(honest_gradients + byz_vectors)
```

## 3. Aggregator API — class-based

Aggregators are classes with `__call__`:

```python
class TrMean:
    def __init__(self, f=0):
        self.f = f
    def __call__(self, vectors):
        ...   # vectors: (n, d) tensor, (n, d) ndarray, or list of 1-D
```

- **State lives on the instance** (e.g., `GeometricMedian` maintains iteration count).
- **`f` is a constructor argument**, not a call argument.
- **Input types are flexible.** A utility `check_vectors_type()` ([byzfl/utils/](../byzfl/byzfl/utils/)) normalizes PyTorch tensors, NumPy arrays, and Python lists to a common representation.
- **No `register()` factory, no `upper_bound`, no `influence`** — each aggregator is just a class.

**Implemented (13):** `Average`, `Median`, `TrMean`, `GeometricMedian`, `Krum`, `MultiKrum`, `CenteredClipping`, `MDA`, `MoNNA`, `Meamed`, `CAF`, `SMEA`, and one more.

**Pre-aggregators (4)** ([preaggregators.py](../byzfl/byzfl/aggregators/preaggregators.py)): `NNM` (nearest neighbor mixing), `Bucketing`, `Clipping`, `ARC` (adaptive robust clipping). Composed before the main aggregator via `RobustAggregator`.

## 4. Attack API — class-based, same pattern

```python
class InnerProductManipulation:
    def __init__(self, tau=2.0):
        self.tau = tau
    def __call__(self, honest_vectors):
        ...   # returns a single 1-D attack vector
```

**Implemented (9):** `SignFlipping`, `InnerProductManipulation`, `Optimal_InnerProductManipulation`, `ALittleIsEnough`, `Optimal_ALittleIsEnough`, `Mimic`, `Inf`, `Gaussian`, `LabelFlipping` (client-side variant, in `fed_framework`).

## 5. Federated-learning framework

This is what ByzantineMomentum **entirely lacks**:

- **`Server`** ([server.py](../byzfl/byzfl/fed_framework/server.py)) — global model, robust aggregation, SGD step with `MultiStepLR` scheduler.
- **`Client`** ([client.py](../byzfl/byzfl/fed_framework/client.py)) — local model, local optimizer, **client-side momentum** (the only momentum position byzfl exposes):
  ```python
  self.momentum_gradient.mul_(self.momentum)
  self.momentum_gradient.add_(gradient, alpha=1 - self.momentum)
  ```
- **`ByzantineClient`** ([byzantine_client.py](../byzfl/byzfl/fed_framework/byzantine_client.py)) — wraps an attack name + params dict, instantiates the attack class via `inspect.signature()`, returns `f` faulty vectors.
- **`DataDistributor`** ([data_distributor.py](../byzfl/byzfl/fed_framework/data_distributor.py)) — splits a dataset across clients in three modes: `iid`, `dirichlet_niid`, `gamma_similarity_niid`.
- **`RobustAggregator`** ([robust_aggregator.py](../byzfl/byzfl/fed_framework/robust_aggregator.py)) — dynamically chains pre-aggregators and the main aggregator from JSON-ish dicts.
- **`ModelBaseInterface`** — base class for Server/Client. Ships 9 PyTorch models out of the box.

## 6. Benchmark / evaluation framework

[byzfl/benchmark/](../byzfl/byzfl/benchmark/):
- `Benchmark` generates the Cartesian product of aggregators × attacks × `f` × learning rates × momentums × data distributions, runs them via `multiprocessing.Pool`, logs to a results directory.
- `evaluate_results.find_best_hyperparameters()` picks the hyperparameter tuple maximizing worst-case accuracy across attacks.
- Default config in [benchmark.py](../byzfl/byzfl/benchmark/benchmark.py) includes 800 training steps, 10 honest clients, `f ∈ [1,2,3,4]`, two aggregators, three attacks, two pre-aggregators, grids over LR / momentum / weight decay.

No equivalent in BM — `reproduce.py` there is a hand-written sweep.

## 7. Library-readiness

- **Pip-installable as `byzfl`** (PyPI). The agent could not locate `setup.py` / `pyproject.toml` in the local tree but the package is published on PyPI, so one of them exists.
- **5 unit test files** in [tests/](../byzfl/tests/): aggregators, attacks, benchmark, preaggregators, `run_all_tests.py`.
- **Sphinx docs** under [docs/](../byzfl/docs/).
- **README with quick-start** for both PyTorch and NumPy.
- **Versioned and published** (2025 arXiv paper references versioned releases).

This is the only sibling in this folder that is actually a library.

## 8. Comparison with ByzantineMomentum

### 8.1 Code lineage — *zero direct reuse*

byzfl is not a fork of ByzantineMomentum. No `tools/pytorch.py` with `flatten`/`relink`, no `register()` factory, no EPFL/Rouault headers copied in. Several of the ByzFL authors (Stephan, Pinot, Guerraoui) are from the same lab that wrote BM, so there is clear *intellectual* lineage, but the code is a clean-slate rewrite with new conventions.

### 8.2 Architectural differences

| Axis | ByzantineMomentum | byzfl |
|---|---|---|
| Aggregator API | Function + `register()` factory | Class with `__call__` |
| Attack API | Function + `register()` factory | Class with `__call__` |
| Tensor layout | One flat `(D,)` buffer per gradient (flatten/relink) | Flexible: `(n, d)` matrix, `(n, d)` ndarray, or `list[1-D]` |
| Framework support | PyTorch only | PyTorch + NumPy |
| Client abstraction | None | `Client`, `ByzantineClient` |
| Server abstraction | Implicit in `attack.py` | Explicit `Server` class |
| Data distribution | Shared trainset, every worker samples the same | `DataDistributor` (IID + Dirichlet + gamma similarity) |
| Pre-aggregators | Ad-hoc inside the GAR | First-class, composable via `RobustAggregator` |
| Momentum positions | `update` \| `server` \| `worker` | **Client-side only** |
| Introspection | `upper_bound(n,f,d)`, `influence(honests, attacks)` | None |
| Study metrics | 17 per step (variances, norms, cosines) | Loss + accuracy |
| Benchmark | Hand-written `reproduce.py` | Programmatic `Benchmark` with multiprocess sweep |
| Built-in models | None (user brings a `torch.nn.Module`) | 9 (MNIST/CIFAR CNNs, full ResNet family) |
| Tests / docs / license | None / minimal / EPFL DCL | Unit tests / Sphinx / MIT |

### 8.3 What ByzantineMomentum has that byzfl appears to lack

- **Worker-side momentum** (`--momentum-at worker`) — the paper's *actual contribution*. byzfl has client-side momentum, which is closer to "update" or "worker" depending on how you squint, but does NOT give the GAR momentum-averaged inputs the way BM's `worker` mode does. To replicate BM's variance-reduction result in byzfl you would need to patch `Client.get_flat_gradients_with_momentum()`.
- **Server-side momentum** (`--momentum-at server`) — not exposed.
- **`upper_bound(n, f, d)` and `influence(honests, attacks)`** — theoretical per-aggregator hooks. byzfl aggregators are opaque callables.
- **The 17 study metrics** — variances, norms, cosines among sampled/honest/attack/defense gradients, composite curvature. byzfl logs loss and accuracy only.
- **Flatten/relink performance** — byzfl allocates on every call (`check_vectors_type` normalization). At BM's scale (tens of millions of params × dozens of workers × thousands of steps), this may matter.
- **Mixed-GAR runs.** BM's `--gars "name1,freq1;name2,freq2;…"` picks a GAR randomly per step. byzfl has no equivalent.

### 8.4 What byzfl has that ByzantineMomentum lacks

- Actual library packaging (pip, tests, docs, MIT).
- Federated learning framework (Server/Client/ByzantineClient/DataDistributor).
- Non-IID data distribution.
- Benchmark automation with hyperparameter sweep.
- NumPy support.
- Built-in models.
- Pre-aggregator composition as a first-class pipeline.
- Broader set of attacks (Gaussian, Mimic, Inf, Optimal variants).

## 9. Takeaways for the library plan

1. **byzfl occupies the "pip-installable Byzantine FL library" space.** Any new library has to answer: *what does this give users that byzfl does not?*
2. **Clearest differentiators (what byzfl lacks):**
   - Worker-side momentum + variance/curvature metrics (the BM paper's contribution).
   - `upper_bound` / `influence` introspection.
   - Flatten/relink performance path.
   - Mixed-GAR scheduling.
   - Gossip as a first-class setting (byzfl is centralized-only; BM's gossip descendants are not).
3. **Do not compete on polish.** byzfl's tests, Sphinx docs, benchmark module, and non-IID data distribution are high-quality and mature. Trying to out-polish them is a losing fight.
4. **Three viable strategies:**
   - **Adopt byzfl and contribute upstream.** Add `BaseAggregator.upper_bound()`/`influence()`, add a `MomentumPosition` enum, add gossip support. Lower effort, larger community.
   - **Thin add-on.** Ship a separate package (`byzfl-momentum`? `byzfl-study`?) that registers BM-style metrics and worker-momentum clients on top of byzfl. Keeps differentiation narrow and honest.
   - **Fork-and-diverge.** Build a new library focused on research introspection (metrics + bounds) and gossip. Higher effort, clearer identity. Justified only if the research community explicitly wants the tradeoffs.
5. **Either way, translate the naming.** `TrMean` ≈ `trmean`, `GeometricMedian` ≈ `rfa` (in the EPFL descendant repos), `InnerProductManipulation` ≈ `empire`. A compatibility shim makes cross-citation between papers possible without renaming history.
6. **The class-based API wins for public consumption; the function + factory wins for research speed.** The two research repos that stayed close to BM (robust-collaborative-learning, Byzantine-Robust-Gossip) kept the function factory. The one that aimed at being a library (byzfl) went class-based. Evidence that the two use cases pull in different directions — a library that tries to serve both needs a thin class wrapper over a functional core.
