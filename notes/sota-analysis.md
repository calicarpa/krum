# SOTA: Byzantine-Resilient ML Libraries

> Analysis date: 2026-07-20

> Scope: Comparison of Krum, ByzFL, and other open-source libraries for
> Byzantine-resilient distributed machine learning research.

---

## 1. Libraries Surveyed

Three categories emerge from the landscape:

**Byzantine-resilient ML libraries** (built-in robust aggregation + attacks):
- **Krum** — targeting JMLR MLOSS. Byzantine-resilient distributed ML.
- **ByzFL** — arXiv:2505.24802 (May 2025). Robust federated learning.

**General FL simulation frameworks** (no built-in Byzantine resilience):
- **FedLab** — JMLR 24(100), 2023. General FL simulation. Dormant since 2022.

**Research prototypes** (specialised scope, lower maturity):
- **Blades** — IoTDI 2024. Byzantine attack/defense benchmark.
- **FL-Byzantine-Library** — Aggregator/attack collection.
- **ByzPy** — Byzantine-robust learning runtime. Low activity.

### Libraries not in scope

- **Flower**, **FedML**, **PySyft**, **OpenFL**, **TensorFlow Federated**:
  general-purpose FL frameworks with optional (but not built-in) Byzantine
  resilience. None publish a comprehensive aggregator+attack collection.

---

## 2. Krum — Deep Analysis

### 2.1 Architecture

Three-layer package structure:

```
krum/
├── primitives/       Stateless building blocks
│   ├── aggregators/  10 gradient aggregation rules
│   ├── attacks/      5 Byzantine attack strategies
│   └── models/       Zero-copy flat tensor Model wrapper
├── simulations/      Faithful paper protocol reproductions
│   ├── centralised/  Parameter-server simulations
│   └── decentralised/  Peer-to-peer simulations
└── orchestration/    Experiment campaign management
    ├── Orchestrator   Parameter binding + run lifecycle
    ├── Metric         Typed metric channel
    └── MetricDataFrame  Multi-run result storage
```

### 2.2 Key Design Decisions

**Stateless primitives.** Aggregators and attacks are `@classmethod` —
no instantiation, no hidden state. `Krum.aggregate(grads, n=7, f=2)`
is the calling convention. This contrasts with ByzFL's instance-based
`aggregator = Krum(f=2, ...); result = aggregator(vectors)`.

**Zero-copy model wrapper.** `Model` wraps `nn.Module` and provides
flat 1-D tensor views of parameters and gradients that share memory
with the underlying module. `model.gradients = flat_value` unpacks
into each parameter's `.grad` without copying. `relink_*()` methods
restore the shared-storage invariant after PyTorch ops break it
(e.g. `zero_grad(set_to_none=True)`).

**Faithful simulations.** Each simulation reproduces the exact
protocol from its reference paper:

| Simulation | Paper | LR Schedule | Init | Weight Decay | Attack Stop |
|---|---|---|---|---|---|
| `KrumSimulation` | NIPS 2017 | Constant | Default | 0.0 | — |
| `HiddenVulnerabilitySimulation` | ICML 2018 | Robbins-Monro | Xavier | 1e-4 | ✓ |
| `MonnaSimulation` | ICML 2023 | Momentum-SGD | Default | 0.0 | — |

**Lightweight orchestration.** `Orchestrator.run()` resolves
parameters against function signatures, freezes hashable state, and
records `(step, value)` pairs per `Metric`. All data is in-memory
(no persistence in v0.1). Runs are synchronous and single-threaded.

### 2.3 Aggregators (10)

| Rule | Class | Complexity | Resilience | Notes |
|---|---|---|---|---|
| Average | `Average` | O(nd) | None | Baseline |
| Median | `Median` | O(nd log n) | n > 2f | Coordinate-wise quantile |
| TrimmedMean | `TrimmedMean` | O(nd log n) | n > 2f | Drop f extremes per coord |
| Krum | `Krum` | O(n²d) | n ≥ 2f+3 | Lowest-score selection |
| MultiKrum | `MultiKrum` | O(n²d) | n ≥ 2f+3 | Top-m averaging |
| Bulyan | `Bulyan` | O(θ·n²d) | n ≥ 4f+3 | MultiKrum → TrimmedMean |
| Brute | `Brute` | O(C(n,k)·k²d) | n ≥ 2f+1 | Combinatorial min-diameter |
| GeoMed | `GeoMed` | O(n²d) | — | Medoid (selects one vector) |
| Aksel | `Aksel` | O(nd) | n > 2f | Median pivot + n-f nearest |
| NearestNeighborAverage | `NearestNeighborAverage` | O(md) | Caller policy | Pivot + num_closest |

Bulyan uses a documented approximation: scores computed once, removed
gradients masked with `inf` rather than recomputed per iteration.

### 2.4 Attacks (5)

| Attack | Class | Strategy | Parameters |
|---|---|---|---|
| SignFlip | `SignFlipAttack` | `-scale · mean(honest)` | `scale` |
| ALIE | `ALIEAttack` | `mean ± z · std` per coord | `z="max"` or numeric |
| Gaussian | `GaussianAttack` | Isotropic `N(μ, σ²)` | `mu`, `std` |
| FullGradientNegation | `FullGradientNegationAttack` | `-κ · g_full` | `full_gradient`, `kappa` |
| SmallPerturbation | `SmallPerturbationAttack` | Max γ evading aggregator | `aggregator`, `n`, `p` |

`ALIEAttack.max_z()` computes Φ⁻¹((n-s)/n) via Normal.icdf (Algorithm 3
from Baruch et al., NeurIPS 2019). Raises `ValueError` for degenerate
configurations.

`SmallPerturbationAttack` implements exponential + binary search for
the maximum perturbation γ that the target aggregator still "selects."
Has two selection-test paths: score-based (check Krum/MultiKrum scores)
and output-change heuristic (for other aggregators). Documents
non-monotonicity in γ and the `gamma_init` floor limitation.

### 2.5 Strengths

- **Zero-copy model wrapper** — unique feature, enables efficient
  per-round flat gradient operations without allocation overhead.
- **Decentralised simulation** — peer-to-peer support (MoNNA) with
  Byzantine reach models ("all" worst-case vs "sampled" random gossip).
- **Protocol-faithful simulations** — each simulation reproduces
  exactly the learning rate schedule, initialization, and stopping
  conditions of the original paper. This is not a generic framework
  but a curated reproduction suite.
- **Clean stateless primitives** — `@classmethod` API means no
  object management, easy functional composition.
- **Type safety** — full type annotations, `TypedDict`, `Generic`,
  `Literal`. `__slots__` on Model.
- **ADRs** — design decisions documented as dated markdown files.
- **Buffer reuse (`out=`)** — consistent zero-allocation pattern
  across all primitives.

### 2.6 Limitations

- **No persistence** — orchestrator is in-memory only. All data lost
  on process exit.
- **Synchronous** — single-threaded. No parallel run execution.
- **No seed management** — orchestrator doesn't distribute seeds.
  Experiments must self-seed.
- **No data heterogeneity** — IID only in centralised simulations.
- **Overlapping aggregator validation** — each rule validates `n`/`f`
  independently (~15 duplicated lines each).
- **Bulyan approximation unmeasured** — impact of single-pass scoring
  vs recomputation not tested.
- **No GPU tests** — CI is CPU-only.
- **Matplotlib in core deps** — used only in experiment scripts.
- **No config system** — everything is Python code.

---

## 3. ByzFL — Deep Analysis

### 3.1 Architecture

```
byzfl/
├── aggregators/        12 aggregators + 4 pre-aggregators
├── attacks/            9 attack classes
├── benchmark/          JSON config → Cartesian sweep → results
├── fed_framework/      Client, Server, ByzantineClient, DataDistributor
└── utils/              Type dispatch (NumPy/PyTorch backend)
```

### 3.2 Key Design Decisions

**Instance-based API.** `aggregator = Krum(f=2); result = aggregator(vectors)`.
Stateful objects with `__init__` parameters and `__call__` invocation.

**Dual-backend type dispatch.** `check_vectors_type()` returns a `tools`
object that is either `numpy` or a custom `torch_tools` shim implementing
a NumPy-compatible API over PyTorch ops. All aggregators and attacks
use `tools.xxx()` exclusively — same code works for both backends.

**Flat-vector protocol.** All gradients are flattened via
`flatten_dict()` into a single 1D tensor. Aggregation operates on
`(n, d)` matrices. Results are unflattened via `unflatten_dict()`.

**JSON-based experimentation.** A single `config.json` declares the
entire experiment. `generate_all_combinations()` recursively computes
the Cartesian product of list-valued fields. `run_benchmark()` uses
`multiprocessing.Pool` for parallel execution.

**FedAvg + DSGD.** Two training paths: direct gradient aggregation
(DSGD) and local model update with weight averaging (FedAvg).

### 3.3 Aggregators (12 + 4 pre-aggregators)

| Aggregator | Complexity | Notes |
|---|---|---|
| Average | O(nd) | — |
| Median | O(nd log n) | Uses `quantile(q=0.5)` |
| TrMean(f) | O(nd log n) | Sort → trim → mean |
| GeometricMedian(ν, T) | O(T·n·d) | Smoothed Weiszfeld, 3 iters |
| Krum(f) | O(n²d) | `cdist²` → score → argmin |
| MultiKrum(f) | O(n²d) | Top m by score → mean |
| CenteredClipping(m, L, τ) | O(L·n·d) | Stateful (momentum buffer) |
| MDA(f) | O(C(n,k)·n·d) | Combinatorial, exponential |
| MoNNA(f, idx) | O(n·d + k log n) | Nearest of chosen pivot |
| MeaMed(f) | O(nd log n) | Per-dim nearest-to-median |
| CAF(f) | O(iters·(n·d + d²·n)) | Power method on weighted cov |
| SMEA(f) | O(C(n,k)·(n·d + d²·iters)) | Combinatorial, exponential |

**Pre-aggregators:**
- `NNM(f)`: Replace each vector with mean of its n-f nearest neighbors
- `Bucketing(s)`: Random permutation → bucket means (reduces n)
- `Clipping(c)`: Scale each vector to L2 norm ≤ c
- `ARC(f)`: Adaptive clipping threshold

### 3.4 Attacks (9)

| Attack | Strategy | Notes |
|---|---|---|
| SignFlipping | `-mean(x)` | — |
| InnerProductManipulation | `-τ · mean(x)` | τ=2.0 default |
| Opt-IPM | `-τ_opt · mean(x)` | Line search for optimal τ |
| ALittleIsEnough | `μ + τ · σ` per dim | τ=1.5 default |
| Opt-ALIE | `μ + τ_opt · σ` | Line search for optimal τ |
| Mimic | Returns `x[ε]` | Collusive |
| Inf | `[+∞, ..., +∞]` | — |
| Gaussian | `N(μ, σ²)` | — |
| LabelFlipping | `mean(x)` | No-op in aggregation space |

Opt-IPM and Opt-ALIE use the same line-search: greedy expansion then
contraction, evaluating `||agg(pre_agg([honest ‖ byz(τ)])) - mean(honest)||₂`.
Each evaluation runs the full pipeline; up to 20 evaluations per attack.

### 3.5 Strengths

- **Wide coverage** — 12 aggregators, 4 pre-aggregators, 9 attacks.
- **JSON config** — non-programmers can define experiment sweeps.
- **Data heterogeneity** — Dirichlet and gamma-similarity non-IID.
- **Dual backend** — NumPy for testing, PyTorch for GPU.
- **Parallel execution** — `multiprocessing.Pool` for sweeps.
- **Worst-case metric** — `min(max_accuracy)` across attacks per cell.
- **Heatmap visualization** — built-in plotting.
- **Documentation** — Sphinx docs at byzfl.epfl.ch.

### 3.6 Limitations

- **No peer-to-peer** — parameter-server only. No decentralised topology.
- **Combinatorial aggregators unusable at scale** — MDA and SMEA enumerate
  all C(n, k) subsets without guard for large n.
- **No gradient masking** — LabelFlipping attack returns `mean(honest)`,
  a no-op in aggregation space. The actual flip is client-side.
- **In-place mutation** — `Clipping.__call__` modifies input list in-place.
- **Momentum quirk** — momentum buffer applied before Byzantine attack,
  so attacker sees momentum-augmented gradients.
- **Multi-process limitations** — Python `multiprocessing.Pool` has
  pickling overhead for large models.
- **Test coverage gaps** — no tests for CAF, SMEA, pipeline composition,
  or integration.
- **Pinned dependencies** — exact version pins (numpy==1.26.4, etc.).
- **arXiv only** — not peer-reviewed.
- **Packaging gap** — `setup.py` is gitignored, packaging external.

---

## 4. Feature Comparison

### 4.1 Six-way comparison table

| Feature | Krum | ByzFL | FedLab | Blades | FL-Byz-Lib | ByzPy |
|---|---|---|---|---|---|---|
| **Aggregators** | 10 | 12+4 pre | 2 | 9 | **36** | 12+4 pre |
| **Attacks** | 5 | 9 | 0 | **10** | **23** | 8 |
| **Pre-agg** | — | 4 | — | — | — | 4 |
| **Topology** | PS + **P2P** | PS only | PS only | PS only | PS only | PS + **P2P** |
| **Data split** | IID | IID+Dir+γ | **9 schemes** | Dir+Shard | IID+Dir+sort | IID+Dir |
| **FL algos** | 3 paper | 2 generic | **10 baselines** | FedAvg+DP | 1 generic | PS+P2P |
| **Compression** | — | — | **QSGD+TopK** | — | — | — |
| **Model wrapper** | **Zero-copy** | std flatten | none | std | std | std |
| **Orchestration** | Python API | JSON bench | ServerHandler | YAML+Ray | CLI dataclass | DAG scheduler |
| **Parallel** | sync | mproc.Pool | **3 deploy modes** | Ray Tune | — | **actors (5 bkends)** |
| **Publication** | Targeting JMLR | arXiv 2025 | **JMLR 2023** | **IoTDI 2024** | TIFS 2024 | none |
| **Stars** | — | 36 | 828 | 156 | 12 | 2 |
| **Persistence** | in-memory | filesystem | filesystem | filesystem | filesystem | filesystem |
| **Visualisation** | manual | **curves+heat** | manual | manual | manual | manual |
| **License** | MIT | MIT | Apache 2.0 | none | none | MIT |
| **CI / checks** | **Ruff+ty+pytest** | none | Codecov | flake8 | none | pytest |
| **Type annot.** | **full (ty)** | partial | partial | partial | partial | partial |
| **Tests** | **comprehensive** | basic | moderate | moderate | basic | basic |

Legend: PS = parameter-server, P2P = peer-to-peer, Dir = Dirichlet,
γ = gamma-similarity. **Bold** = best in category.

### 4.2 Positioning map

```
                    Byzantine resilience built-in
                              │
         FL-Byz-Lib (36 agg)  │  Krum (10 agg, P2P, zero-copy)
         ByzFL (12+4 agg)     │  Blades (benchmark, IoTDI 2024)
         ByzPy (12+4 agg,     │
               actor runtime, │
               P2P)           │
                              │
    ──────────────────────────┼──────────────────────────→ General FL
    General FL                │                         simulation
    simulation                │
    (no Byz)                  │
                    FedLab (JMLR 2023, 9 partition schemes,
                    10 FL algos, 3 deploy modes, 828★)
                              │
```

### 4.3 Categorisation by purpose

| Category | Libraries |
|---|---|
| **Byzantine-robust libraries** | Krum, ByzFL, FL-Byzantine-Library, ByzPy |
| **Benchmark suite** | Blades |
| **General FL framework** | FedLab |
| **Research prototype / low maturity** | FL-Byzantine-Library, ByzPy |



---

## 5. Detailed Delta

### 5.1 What Krum has that ByzFL does not

| Feature | Why it matters |
|---|---|
| **Bulyan** (n ≥ 4f+3) | Two-stage robust aggregator used in several papers |
| **Brute** (combinatorial min-diameter) | Theoretically optimal subset selection |
| **Aksel** (O(nd) median-pivot) | Linear-time alternative to Krum |
| **FullGradientNegationAttack** | Requires full-dataset knowledge; used in ICML 2018 |
| **SmallPerturbationAttack** | Curse-of-dimensionality attack; ICML 2018 |
| **Decentralised (P2P) simulation** | MoNNA protocol with sampled Byzantine reach |
| **Zero-copy Model wrapper** | Shared-storage flat gradients, no per-round copies |
| **Faithful paper reproductions** | Exact protocol fidelity (not generic framework) |
| **Orchestrator + Metric API** | Programmatic sweep definition, typed metrics |
| **Type-safe primitives** | Full `ty`-checked type annotations |
| **ADR documentation** | Architecture decisions documented |
| **Edge case test coverage** | f=0, n=f, minimal config, deterministic init |

### 5.2 What ByzFL has that Krum does not

| Feature | Why it matters |
|---|---|
| **CAF** (Covariance-bound Agnostic Filter, 2025) | State-of-the-art robust aggregator |
| **SMEA** (Smallest Maximum Eigenvalue, 2023) | Recent robust aggregator |
| **CenteredClipping** (ICML 2021) | Widely cited clipping-based defense |
| **MDA** (Minimum Diameter Averaging) | Early robust aggregator |
| **MeaMed** (Mean Around Median) | Hybrid coord/vector defense |
| **Pre-aggregators** (NNM, Bucketing, Clipping, ARC) | Composable preprocessing pipeline |
| **LabelFlipping attack** | Data-poisoning simulation |
| **IPM + Opt-IPM attacks** | Inner-product based adversarial strategy |
| **Mimic attack** | Collusive adversary simulation |
| **Data heterogeneity** (Dirichlet, gamma) | Non-IID realism |
| **JSON config sweeps** | Accessible to non-programmers |
| **Parallel execution** | Multiprocessing for large sweeps |
| **Filesystem persistence** | Results survive process crashes |
| **Built-in heatmap visualization** | Immediate results |
| **NumPy backend** | Lightweight testing without GPU |
| **Broader aggregator+attack count** | 12+4 vs 10, 9 vs 5 |

### 5.3 Overlap matrix

| Feature | Krum | ByzFL | FedLab | Blades | FL-Byz-Lib | ByzPy |
|---|---|---|---|---|---|---|
| Average | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| Median | ✓ | ✓ | — | ✓ | ✓ | ✓ |
| TrimmedMean | ✓ | ✓ | — | ✓ | ✓ | ✓ |
| Krum / MultiKrum | ✓ | ✓ | — | ✓ | ✓ | ✓ |
| GeoMed | ✓ | ✓ | — | ✓ | — | ✓ |
| MoNNA / NN-Avg | ✓ | ✓ | — | — | — | ✓ |
| CenteredClipping | — | ✓ | — | ✓ | ✓ | ✓ |
| Bulyan | ✓ | — | — | — | ✓ | — |
| SignFlip | ✓ | ✓ | — | ✓ | ✓ | ✓ |
| ALIE | ✓ | ✓ | — | ✓ | ✓ | ✓ |
| Gaussian | ✓ | ✓ | — | ✓ | — | ✓ |
| IPM | — | ✓ | — | ✓ | ✓ | ✓ |
| Mimic | — | ✓ | — | — | ✓ | ✓ |
| LabelFlip | — | ✓ | — | ✓ | ✓ | ✓ |
| PyTorch | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| NumPy | — | ✓ | — | — | — | optional |
| MIT license | ✓ | ✓ | — (Apache) | — | — | ✓ |
| JMLR paper | target | — | ✓ | — | — | — |

Key insight: **No single library covers all common aggregators and attacks.**
Krum has unique coverage of Bulyan, Brute, Aksel, FullGradientNegation,
SmallPerturbation. The research landscape is fragmented — this is the
problem Krum's paper should highlight.

### 5.4 Feature count summary

```
               Krum  ByzFL  FedLab  Blades  FL-Byz  ByzPy
Aggregators     10   12+4     2       9      36    12+4
Attacks          5     9      0      10      23      8
Pre-agg          0     4      0       0       0      4
Topologies       2     1      1       1       1      2
FL algos       3pp    2gen   10gen   1+DP    1gen   PS+P2P
Model wrapper   ZC   std     none    std     std    std
Orchestration  PyAPI  JSON   SH      YAML    CLI    DAG
Parallel       sync   Pool   3mode   Ray     —      actors
Publication   JMLR   arXiv  JMLR   IoTDI   TIFS    none
Stars           —     36     828    156      12      2
Type checks    ty    part   part    part    part   part
Tests          full  basic  mod     mod     basic  basic
```

3pp = paper-specific protocols, 2gen = 2 generic training loops,
10gen = 10 generic FL algorithms, SH = ServerHandler.

---

## 6. Other Libraries

### 6.1 FedLab — Deep Analysis

**Paper**: Dun Zeng, Siqi Liang, Xiangjing Hu, Hui Wang, Zenglin Xu.
*FedLab: A Flexible Federated Learning Framework*.
JMLR 24(100):1–7, 2023.

**GitHub**: https://github.com/SMILELab-FL/FedLab — 828 stars, 141 forks.
Last release v1.3.0 (Oct 2022). Dormant since then.

**License**: Apache 2.0.

#### Architecture

```
fedlab/
├── contrib/algorithm/    FL algorithm implementations
│   ├── FedAvg, FedProx, SCAFFOLD, FedDyn, q-FFL
│   ├── FedNova, IFCA, Ditto, pFedAvg, CFL
├── contrib/client_sampler/  Power-of-choice, VRB, MABS, DIVFL
├── contrib/compressor/      QSGD, Top-K sparsification
├── core/
│   ├── client/           ClientManager, Trainer
│   ├── server/           ServerHandler, ServerManager, Hierarchical
│   ├── communicator/     Package, Processor (torch.distributed)
│   ├── coordinator/      Orchestration
│   ├── network/          NetworkManager
│   └── standalone/       Single-process simulation
├── models/               CNN, MLP, RNN
└── utils/
    ├── dataset/          DataPartitioner + 9 partition schemes
    └── aggregator.py     fedavg_aggregate + fedasync_aggregate ONLY
```

#### Byzantine resilience: zero

The `fedlab/utils/aggregator.py` file contains exactly two aggregation
functions — `fedavg_aggregate` (weighted averaging) and
`fedasync_aggregate` (staleness-weighted averaging). Neither is robust
to any number of Byzantine workers. The paper, README, and all
documentation make no mention of attacks, defenses, or poisoning.

A 2024 survey comparing 15 open-source FL frameworks (International
Journal of ML and Cybernetics) confirms: *"FedLab and EasyFL provide
no security mechanisms and receive a score of zero in this criterion."*

#### What FedLab does well

1. **Data heterogeneity (9 partition schemes)** — the most comprehensive
   data partitioning of any library surveyed:
   - Balanced IID, unbalanced IID, heterogeneous Dirichlet, shards,
     balanced/unbalanced Dirichlet, quantity-based label skew,
     noise-based feature skew, FCUBE synthetic.
   - 14 supported datasets (CIFAR-10/100, MNIST, FashionMNIST, SVHN,
     CelebA, FEMNIST, Shakespeare, Sent14, Reddit, Adult, Covtype,
     RCV1, FCUBE, LEAF-Synthetic).

2. **FL algorithm baselines** — 10 algorithms implemented, including
   FedAvg, FedProx, SCAFFOLD, FedDyn, q-FFL, FedNova, IFCA, Ditto,
   pFedAvg, and CFL. Each follows the paper's official implementation
   where available.

3. **Communication compression** — QSGD (4/8/16-bit) and Top-K
   sparsification. Measured compression ratios against accuracy.

4. **Deployment modes** — three modes with increasing complexity:
   - **Standalone**: single-process simulation (simplest).
   - **Cross-process**: multi-GPU/multi-machine via `torch.distributed`.
   - **Hierarchical-hybrid**: multi-level topology for cross-silo FL.

5. **Publication venue** — JMLR MLOSS is the same venue Krum targets.
   FedLab is the only directly comparable paper in terms of publication
   format and expectations.

#### Limitations

1. **No Byzantine resilience whatsoever.** Cannot run a single robust
   aggregation experiment without implementing everything from scratch.
2. **Dormant.** No commits since October 2022. The PyPI package (v1.3.0)
   is pinned to old PyTorch versions.
3. **Algorithms are contrib, not core.** The FL algorithm implementations
   live in `fedlab/contrib/algorithm/`, suggesting they are
   community-contributed rather than core-maintained.
4. **No attack simulation.** Cannot test robustness to any threat model.
5. **No built-in orchestration sweeps.** Experiment configuration is
   manual Python scripting.
6. **Minimal type annotations.** `aggregator.py` uses bare `list` and
   `torch.Tensor` without generics.

#### Relevance to Krum

FedLab is **not a competitor** — a researcher choosing between Krum and
FedLab would be choosing between two entirely different problems
(Byzantine resilience vs general FL simulation). FedLab's relevance is
as the closest publication precedent:

- **Same venue** (JMLR MLOSS): FedLab's paper is the most directly
  comparable paper in terms of format, length (7 pages), and editorial
  expectations.
- **Orthogonal focus**: The FedLab paper covers communication efficiency,
  data partitioning, and FL algorithm reproduction — none of which Krum
  addresses. Krum covers Byzantine resilience, protocol-faithful
  simulations, and experiment orchestration — none of which FedLab
  addresses.
- **Complementary citation**: The Krum paper should cite FedLab as the
  closest JMLR MLOSS precedent for FL frameworks, and contrast its
  general-purpose FL scope with Krum's Byzantine-specific focus.

### 6.2 Blades — Deep Analysis

**Paper**: Shenghui Li et al. *Blades: A Unified Benchmark Suite for
Byzantine Attacks and Defenses in Federated Learning*.
IEEE/ACM IoTDI 2024, pp. 158–169. arXiv:2206.05359.

**GitHub**: https://github.com/lishenghui/blades — 156 stars, 25 forks.
Last commit Feb 2025. Lightly maintained.

**License**: None specified.

#### Scope and philosophy

Blades is a **benchmark suite**, not a general library. Its goal is to
provide a standardised evaluation framework for comparing Byzantine
attacks and defenses. The paper re-evaluates 6 aggregation rules against
6 attacks across 3 datasets (~1,500 trials) and finds that many defenses
break under non-IID data even *without* attackers.

#### Aggregators (9)

| Aggregator | Reference |
|---|---|
| MultiKrum | Blanchard et al., NIPS 2017 |
| GeoMed | Chen et al., POMACS 2018 |
| Median | Yin et al., ICML 2018 |
| TrimmedMean | Yin et al., ICML 2018 |
| CenteredClipping | Karimireddy et al., ICML 2021 |
| Clustering | Sattler et al., ICASSP 2020 |
| ClippedClustering (Blades' own) | Li et al., IEEE TBD 2023 |
| DnC (Divide-and-Clip) | Shejwalkar et al., NDSS 2021 |
| SignGuard | Xu et al., ICDCS 2022 |

Note: Core aggregator implementations live in a separate dependency
(`fedlib`), not in Blades itself. The `blades/aggregators/` directory
referenced in the README does not exist in the repo — a significant
documentation/code mismatch.

#### Attacks (10)

| Attack | Reference |
|---|---|
| Noise | Baseline |
| SignFlipping | Li et al., AAAI 2019 |
| LabelFlipping | Fang et al., USENIX Security 2020 |
| ALIE | Baruch et al., NeurIPS 2019 |
| IPM | Xie et al., UAI 2020 |
| DistanceMaximization (Min-Max) | Shejwalkar et al., NDSS 2021 |
| AdaptiveAdversary (targets Median/TM) | Blades original |
| SignGuardAdversary | Targets SignGuard |
| AttackclippedclusteringAdversary | Targets ClippedClustering |

#### Architecture

```
blades/
├── adversaries/       10 attack classes (Ray callbacks)
├── algorithms/        FedAvg + FedAvgDP trainers
├── clients/           Client + ClientProxy (monkey-patching)
├── train.py           CLI entrypoint (Typer + YAML)
└── tuned_examples/    14 YAML experiment configs
```

Key design decisions:
- **Adversary-as-callback**: Attacks hook into the training loop via
  Ray's `TrainerCallback` system — keeps the core FL loop clean.
- **ClientProxy**: Monkey-patching that wraps a benign client and
  conditionally swaps in adversarial behavior at runtime.
- **YAML-driven experiments**: `ray.tune` integration for hyperparameter
  sweeps. Example configs are versioned in the repo.

#### Relation to Krum

Blades is not a library for *building* experiments — it is a benchmark
for *comparing* results. A researcher would use Krum to implement a new
aggregator and then use Blades to benchmark it against existing methods.
Blades' key finding (defenses fail under non-IID data) is a cautionary
result that the Krum paper should cite in Related Work.

### 6.3 FL-Byzantine-Library (CRYPTO-KU) — Deep Analysis

**GitHub**: https://github.com/CRYPTO-KU/FL-Byzantine-Library — 12 stars,
4 forks. Last push March 2026. Actively maintained.

**License**: None (no LICENSE file).

**Papers**: Ships as the codebase for two papers:
- *Byzantines Can Also Learn From History: Fall of Centered Clipping in FL*
  — IEEE TIFS 2024
- *Aggressive or Imperceptible, or Both: Network Pruning Assisted Hybrid
  Byzantines in FL* — arXiv:2404.06230 (WACV 2025)

No standalone library paper.

#### Coverage: the most comprehensive by count

**Aggregators: 36** (22 base + 14 variants).
Includes very recent methods: FedSECA (CVPR 2025), FoundationFL
(NDSS 2025), LASA (WACV 2025), SkyMask (ECCV 2024).

| Core aggregators | Reference |
|---|---|
| avg, krum, bulyan, tm, cm | Classic |
| cc (Centered Clipping), scc (Sequential CC) | ICML 2021 / IEEE TIFS 2024 |
| sign, rfa, fl_trust, gas, foolsgold | Various |
| dnc, flame, fldetector, skymask | NDSS 2021 / USENIX 2022 / KDD 2022 / ECCV 2024 |
| fl_defender, fedredefense, foundation | GitHub / ICML 2024 / NDSS 2025 |
| signguard, lasa, fedseca | GitHub / WACV 2025 / CVPR 2025 |

**Attacks: 23** (16 base + 7 variants). Includes lab's own ROP
(IEEE TIFS 2024) and sparse/pruning-assisted attacks (WACV 2025).

| Core attacks | Reference |
|---|---|
| label_flip, bit_flip, alie, ipm | Classic |
| reloc (ROP), minmax, minsum, fang | IEEE TIFS 2024 / NDSS 2021 / USENIX 2020 |
| cw, krum_attack, trimmed_mean_attack | IEEE S&P 2017 / Various |
| mimic, sparse, sparse_opt, lasa, sign_flip | Various / WACV 2025 |

#### Unique features

1. **Network pruning module** — dedicated `pruners/` with synflow, ERK,
   force, magnitude pruning. Enables sparse attacks that only manipulate
   a fraction of parameters to evade detection. No other Byzantine FL
   library offers this.

2. **Typed dataclass configs** — 9 `@dataclass` classes
   (`FederationConfig`, `ModelConfig`, `AttackConfig`, etc.) composed
   into `FLConfig`. More structured than ByzFL's JSON dict.

3. **Cross-device simulation** — `--cl_part` parameter for partial client
   participation.

4. **CLI entry points** — `python main.py --aggr krum --attack alie` and
   `fl-byzantine` packaged command.

#### Architecture

```
main.py → config/parser.py → mapper.py → fl.py (FL coordinator)
              │                  │
              │                  ├── aggregators/aggr_mapper.py
              │                  ├── attacks/attack_mapper.py
              │                  ├── models/model_registry.py
              │                  ├── pruners/prune_mapper.py
              │                  └── client.py
              │
              └── config/ (9 dataclasses)
```

Aggregators follow `_BaseAggregator.__call__(inputs: List[Tensor]) → Tensor`.
Attacks follow `_BaseByzantine.omniscient_callback(benign_gradients)`
and see honest gradients to craft adversarial updates.

#### Relation to Krum

This library is the **closest in spirit** to Krum's primitives layer:
it is a flat collection of aggregators and attacks with no simulation
framework. Its advantages: wider coverage (36 vs 10 aggregators,
23 vs 5 attacks), recent methods (FedSECA 2025, FoundationFL 2025).
Its disadvantages: no zero-copy model wrapper, no decentralised support,
no orchestration system, no published library paper (the Krum paper
would be the first JMLR MLOSS paper in this space). Note the absence
of a license file — a concern for derivative work.

### 6.4 ByzPy — Deep Analysis

**GitHub**: https://github.com/Byzpy/byzpy — 2 stars, 4 forks.
26 commits on main. Active since Dec 2025. Pre-1.0.

**License**: MIT.

**Docs**: https://byzpy.github.io/byzpy/

**No paper** — practitioner-first engineering project.

#### Architecture: 3-tier actor runtime

ByzPy's defining feature is a **bespoke actor runtime** with three tiers:

```
Application Layer    Aggregators │ Attacks │ Pre-agg │ PS/P2P Helpers
                             ↓
Scheduling Layer     ComputationGraph │ NodeScheduler │ ActorPool │ Operators
                             ↓
Actor Layer          Thread │ Process │ GPU │ Remote (TCP/UCX) │ Channels
```

**Actor Layer** — Five backends behind a unified `ActorRef`:
`ThreadActorBackend`, `ProcessActorBackend`, `GPUActorBackend`,
`RemoteActorBackend` (TCP), `UCXRemoteActorBackend` (InfiniBand/RDMA).

**Scheduling Layer** — DAG-based `ComputationGraph` validated
topologically. `NodeScheduler` fans out subtasks across an `ActorPool`.
Operators declare `supports_subtasks = True` with `create_subtasks()` /
`reduce_subtasks()` for automatic parallelisation.

**Application Layer** — Aggregators, attacks, pre-aggregators, PS/P2P
helpers. All extend `Operator` for uniform schedulability.

#### Aggregators (12 + 4 pre-aggregators)

Coordinate-wise: `CoordinateWiseMedian`, `CoordinateWiseTrimmedMean`,
`MeanOfMedians`.

Geometric/selection: `Krum`, `MultiKrum`, `GeometricMedian`,
`MinimumDiameterAveraging`, `MoNNA`, `SMEA`.

Norm-wise: `CenteredClipping`, `ComparativeGradientElimination`, `CAF`.

Pre-aggregators: `Bucketing`, `Clipping`, `ARC`, `NNM`.

Notable: every aggregator supports optional `chunk_size` for subtask
parallelisation.

#### Attacks (8)

`Empire` (scale * mean), `SignFlip`, `LabelFlip`, `Little` (ALIE with
z_max computation), `Gaussian`, `Inf`, `Mimic`, `IPM`.

Each attack declares input requirements via class flags
(`uses_honest_grads`, `uses_base_grad`, `uses_model_batch`) — the
scheduler only supplies what's needed.

#### Topologies

**Parameter-server** — 5 backend variants (thread, process, remote TCP,
heterogeneous).

**Peer-to-peer** — 5 backend variants + topology-aware routing via
`NeighborSampler`. Message-driven execution: nodes wait for messages
from neighbours before proceeding, fully async. This is the **only**
library with true decentralised P2P Byzantine training.

#### Relation to Krum

ByzPy is the most architecturally ambitious library in the ecosystem.
It is the only one that:
- Treats **decentralised P2P** as a first-class topology alongside PS
- Provides **real distributed execution** (not simulation)
- Offers **heterogeneous compute backends** under a unified actor API

However, it is pre-1.0, has 2 stars, and its bespoke actor runtime
has not been battle-tested. It also has no accompanying paper — the
Krum paper would be peer-reviewed while ByzPy is not.

ByzPy also ships a `benchmarks/byzfl/` directory with 15+ direct
cross-comparison scripts against ByzFL, indicating the maintainers
care about correctness parity with the academic reference.

---

## 7. Positioning for the JMLR Paper

### 7.1 Competitive advantage

Krum's differentiating features for the paper:

1. **Protocol-faithful simulations.** Not a generic FL framework but
   a curated collection of paper-accurate reproductions. Each simulation
   matches the exact protocol of a specific paper (learning rate schedule,
   initialization, weight decay, evaluation metrics). This is the
   strongest differentiator from ByzFL's generic FedAvg/DSGD loops.

2. **Decentralised (P2P) support.** ByzFL is parameter-server only.
   Krum implements the MoNNA ICML 2023 protocol with peer-to-peer
   model mixing and two Byzantine reach models. This is the only
   library with built-in decentralised Byzantine resilience.

3. **Three-layer architecture.** Primitives → Simulations → Orchestration
   is a cleaner separation than ByzFL's monolithic fed_framework.
   The primitives are stateless `@classmethod` (simpler composition),
   the simulations are paper-specific (accuracy), and the orchestration
   is programmatic (flexibility).

4. **Zero-copy model wrapper.** A unique engineering contribution:
   flat tensor views that share memory with the underlying `nn.Module`.
   Eliminates per-round gradient copying overhead.

5. **Unique aggregators and attacks.** Bulyan, Brute, and Aksel
   aggregators; FullGradientNegation and SmallPerturbation attacks
   are not available in any other library.

6. **Code quality.** Full type annotations, `__slots__`, comprehensive
   test coverage including edge cases and regressions, pre-commit CI
   (Ruff + ty + pytest). ByzFL has no CI, no pre-commit, pinned deps.

### 7.2 Honest gaps to acknowledge

1. **Fewer aggregators (10 vs 12+4).** Krum lacks CAF, SMEA, MDA,
   MeaMed, and any pre-aggregator (NNM, Clipping, Bucketing, ARC).
   These are genuinely useful for researchers.

2. **Fewer attacks (5 vs 9).** Krum lacks IPM, Mimic, and LabelFlipping.
   ALIE has only the basic version (no optimal-Opt-ALIE line search).

3. **No data heterogeneity.** IID-only. ByzFL's Dirichlet and gamma
   distributions enable non-IID experiments.

4. **No parallel execution.** Synchronous orchestration is a bottleneck
   for large sweeps (ByzFL has `multiprocessing.Pool`).

5. **No persistence.** Orchestrator is in-memory only.

6. **No built-in visualization.** Requires manual matplotlib.

### 7.3 Paper narrative

The introduction and related work should make four points:

1. **The landscape is fragmented.** Six libraries exist (Krum, ByzFL,
   FedLab, Blades, FL-Byzantine-Library, ByzPy) but none is
   comprehensive. The most complete by count (FL-Byzantine-Library,
   36 aggregators, 23 attacks) has no license, no paper, no
   documentation site, and no simulation framework. The most
   feature-rich library with a paper (ByzFL, arXiv 2025) is
   parameter-server only, uses generic training loops, and has no
   peer review. The only JMLR-published library (FedLab, JMLR 2023)
   has **zero** Byzantine resilience. The only benchmark (Blades,
   IoTDI 2024) delegates all aggregator implementations to an
   external dependency. The most architecturally ambitious (ByzPy)
   is pre-1.0 with 2 stars and no paper.

2. **No library combines protocol-faithful simulations with a
   comprehensive primitive collection.** ByzFL and FL-Byzantine-Library
   offer many aggregators and attacks but use generic training loops.
   Krum is the first to provide paper-specific simulations
   (NIPS 2017, ICML 2018, ICML 2023) that match the exact learning
   rate schedule, initialisation, weight decay, and stopping
   conditions of the original papers.

3. **Krum fills three gaps no other library covers:**
   (a) protocol-faithful simulations (not generic loops),
   (b) decentralised (P2P) Byzantine resilience (only ByzPy also
   supports P2P, but it is pre-1.0 and unpublished),
   (c) a lightweight orchestration system that connects primitives
   to simulations with typed metrics and parameter sweeps.

4. **Krum's engineering quality** (type safety with `ty`, zero-copy
   model wrapper, ADRs, comprehensive edge-case tests, pre-commit CI
   with Ruff) sets a standard that no existing library in this space
   meets — including the JMLR-published FedLab.

### 7.4 How to frame the related work

**FedLab** (JMLR 2023): closest in venue, orthogonal in focus.
Acknowledge as the only JMLR MLOSS precedent for FL frameworks,
then contrast its general-FL scope with Krum's Byzantine-specific
focus. Cite the 2024 survey that gives FedLab "zero" on security.

**ByzFL** (arXiv 2025): closest in content. Acknowledge the wider
aggregator/attack coverage (12+4 vs 10, 9 vs 5) and data heterogeneity
support. Contrast ByzFL's generic FedAvg/DSGD loops with Krum's
protocol-faithful simulations. Note the lack of peer review, P2P
support, zero-copy design, and orchestration system.

**FL-Byzantine-Library** (CRYPTO-KU): most comprehensive by count.
Acknowledge the coverage (36 aggregators, 23 attacks) and unique
pruning module. Note the absence of a license, paper, documentation
site, simulation framework, and orchestration system.

**Blades** (IoTDI 2024): benchmark, not library. Cite their key
finding (defenses fail under non-IID data) as motivation for
protocol-faithful simulations. Note the fedlib dependency mismatch.

**ByzPy** (no paper): most ambitious architecture. Acknowledge the
P2P support and actor-runtime design. Note the pre-1.0 status,
absence of paper, and minimal community adoption.

### 7.5 The Krum pitch

```
Comprehensiveness:  moderate (10 agg, 5 att)
                    fewer than ByzFL or FL-Byz-Lib
                    but unique aggregators (Bulyan, Brute, Aksel)
                    unique attacks (FullGradNeg, SmallPerturb)

Fidelity:           BEST IN CLASS
                    only library with paper-specific simulations
                    exact LR schedules, init, wd, attack stop

Architecture:       BEST IN CLASS
                    only library with P2P + PS topologies
                    only library with zero-copy model wrapper
                    only library with stateless @classmethod API

Engineering:        BEST IN CLASS
                    only library with full type checking (ty)
                    only library with comprehensive edge-case tests
                    only library with ADRs documenting design decisions
                    only library with pre-commit CI (Ruff + ty + pytest)

Publication:        FIRST JMLR MLOSS in this niche
                    FedLab (JMLR) has zero Byzantine resilience
                    ByzFL (arXiv) has no peer review
```

### 7.4 How to frame FedLab in Related Work

```
FedLab (JMLR 2023)
├── Focus: general FL simulation
│   ├── Data heterogeneity (9 schemes)
│   ├── Communication compression (QSGD, Top-K)
│   └── FL algorithm baselines (FedAvg, SCAFFOLD, FedProx, ...)
├── Byzantine resilience: NONE
└── Relation to Krum: COMPLEMENTARY
    ├── FedLab targets "how to simulate FL"
    └── Krum targets "how to make FL robust"

Krum (this paper)
├── Focus: Byzantine-resilient distributed ML
│   ├── Aggregation rules + attacks + zero-copy model
│   ├── Protocol-faithful simulations
│   └── Experiment orchestration
├── General FL simulation: NONE (IID only)
└── Relation to FedLab: COMPLEMENTARY
    ├── Krum does not compete on data heterogeneity or FL algos
    └── FedLab does not compete on Byzantine resilience
```

The paper should present FedLab as the closest precedent in venue
(JMLR MLOSS) and acknowledge its complementary scope, then position
Krum as the first JMLR MLOSS library targeting Byzantine resilience
specifically.
