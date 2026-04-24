# SignGuard — Claude's Analysis

> Complementary to [byzantine_momentum_CLAUDE_analysis.md](byzantine_momentum_CLAUDE_analysis.md)
> (ByzantineMomentum, the baseline). SignGuard is the most *independent* of
> the sibling repos — it borrows almost nothing from BM architecturally.

## 1. What this repo is

- **Paper:** "SignGuard: Byzantine-robust Federated Learning through Collaborative Malicious Gradient Filtering."
- **Venue:** ICDCS 2022.
- **Authors:** Not listed in README or file headers (one EPFL header in a dormant file — see §6.1).
- **Setting:** **Centralized federated learning** — parameter server, synchronous rounds. Clients do local training, server aggregates; up to `f` clients may be Byzantine.

## 2. Architecture

```
SignGuard/
├── federated_main.py   (139 lines — main training loop)
├── running.py          (54 lines — benignWorker / byzantineWorker simulation)
├── options.py          (42 lines — argparse)
├── tools.py            (258 lines — flat utilities, NOT a package)
├── data_loader.py      (324 lines — MNIST/FashionMNIST/CIFAR loaders, IID + non-IID)
├── aggregators/
│   ├── __init__.py         (dict-based registry — buggy, see §8)
│   ├── Mean.py, Median.py, GeoMed.py, Krum.py, Bulyan.py, DnC.py, TrMean.py
│   └── signguard.py        (291 lines — 3 SignGuard variants)
├── attacks/
│   ├── __init__.py         (dict-based registry)
│   ├── naive.py            (random, sign-flip, noise, zero, NaN, no-op)
│   ├── empire.py           (EPFL header — inherited but NOT REGISTERED)
│   ├── lie.py              ("A Little Is Enough")
│   ├── byzMean.py          (two-group coordinated attack)
│   ├── max_sum.py          (MinMax, MinSum, gradient-descent-based)
│   └── adaptive_attack.py  (161 lines — γ-search against the defense)
└── models/
    ├── cnn.py              (CNN for MNIST / Fashion-MNIST / CIFAR-10)
    └── resnet.py           (ResNet18)
```

**Entry point.** [federated_main.py](../SignGuard/federated_main.py) runs one experiment end-to-end: sample `m = ceil(frac × num_users)` clients, designate first `num_byzs` as Byzantine, collect gradients, apply attack, aggregate via chosen GAR, step.

## 3. Algorithms implemented

**Aggregators** (8 classes, imported via dict in [aggregators/__init__.py](../SignGuard/aggregators/__init__.py)):
- Classical: `Mean`, `Median` (NaN-resilient coord-wise), `TrMean`, `GeoMed` (geometric median), `Krum` / `MultiKrum` (67 lines), `Bulyan` (74 lines).
- `DnC` — Divide-and-Conquer, PCA-based outlier detection.
- **SignGuard** ([aggregators/signguard.py](../SignGuard/aggregators/signguard.py)) — the paper's contribution, in three variants:
  - `signguard_multiclass` — **Two-phase filter:**
    1. **L2 norm filter** (lines 32-38): keep gradients whose L2 norm lies in `[0.1·median, 3.0·median]`.
    2. **Sign-pattern cluster** (lines 40-87): sample 10% of coordinates, build 3-D feature `[sign_pos%, sign_zero%, sign_neg%]`, run MeanShift clustering, keep the largest cluster. Intersect with Phase 1 survivors. Clip to median norm. Average.
  - `signguard_multiclass_plus1` — adds temporal similarity to gradient history.
  - `signguard_multiclass_plus2` — uses distance from prior aggregate.

**Attacks** (13 entries):
- [naive.py](../SignGuard/attacks/naive.py): NaN / zero / random / noise / sign-flip / no-attack baseline.
- [lie.py](../SignGuard/attacks/lie.py): "A Little Is Enough" with hardcoded `z = 1.0`.
- [byzMean.py](../SignGuard/attacks/byzMean.py): two coordinated Byzantine sub-groups to confuse aggregators that expect *identical* Byzantine gradients.
- [max_sum.py](../SignGuard/attacks/max_sum.py): `minmax_attack` / `minsum_attack` — numerical search for the attack vector that maximizes the GAR's output deviation.
- [adaptive_attack.py](../SignGuard/attacks/adaptive_attack.py): three white-box attacks (`std`, `sign`, `uv`) that binary-search over `γ` to maximize either Byzantine selection rate or distance to the benign mean. The metric depends on the defense being attacked (Krum/Bulyan/DnC/SignGuard use selection rate; simple defenses use distance).

## 4. Capabilities and limitations

**Can simulate:**
- Federated learning with `--num_users`, `--num_byzs`, `--frac` (participation), `--local_iter`, `--local_bs`, `--lr`, `--momentum`, `--epochs` knobs.
- IID or non-IID data partitions (`--iid`, `--unbalance`).
- MNIST, Fashion-MNIST, CIFAR-10.
- White-box adaptive attacks via `γ` search.
- Tracks loss, training accuracy, test accuracy, Byzantine selection rate, benign selection rate.

**Cannot do:**
- Asynchronous / decentralized learning.
- Differential privacy.
- Multi-GPU.
- Client dropout beyond fractional participation.
- The 17 BM study metrics.

## 5. Library-readiness

- No `setup.py` / `pyproject.toml`.
- No tests.
- No `requirements.txt` (imports imply torch, scikit-learn for MeanShift/DBSCAN, NumPy, TorchVision, `geom_median` package).
- README is **one sentence**.
- **Has a runtime import bug** (§8).

## 6. Comparison with ByzantineMomentum

### 6.1 Lineage evidence — *very thin*

- **Only [attacks/empire.py](../SignGuard/attacks/empire.py) is verbatim EPFL code.** Its header reads: "Copyright © 2019-2020 École Polytechnique Fédérale de Lausanne (EPFL), @author Sébastien Rouault." This is the only file carrying the BM authorship lineage. The function was renamed (`attack` → `fall_empires_attack`).
- **Ironically, `empire.py` is NOT registered** in [attacks/__init__.py](../SignGuard/attacks/__init__.py), so `--attack empire` would fail at runtime. It sits in the repo unused.
- **Classical GARs (Krum, Bulyan, Median, TrMean) are reimplemented from scratch**, not copied from BM. Different class names (`class mean`, `class krum`), different factory pattern (dict lookup, not `register()`).
- **No `tools/pytorch.py` with `flatten`/`relink`.** SignGuard has a single flat [tools.py](../SignGuard/tools.py) (258 lines) with its own utilities, no EPFL header.
- **No `experiments/` wrapper directory.**

The lineage here is almost accidental — one file (empire.py) carries the EPFL copyright, the rest is independent work.

### 6.2 What this repo adds

- **The SignGuard defense itself** — the two-phase L2-norm + sign-pattern MeanShift filter. Genuinely new approach to gradient filtering: uses the *sign distribution* of gradient coordinates rather than L2 distance or coordinate-wise stats.
- **Federated learning simulation.** Proper client abstraction (`benignWorker`, `byzantineWorker` in [running.py](../SignGuard/running.py)) — BM has no client abstraction, just a `for i in range(nb_honests)` loop.
- **Adaptive attacks that search against the defense.** `adaptive_attack.py` is the first example in this folder of an attack that runs a numerical search *calibrated to the chosen GAR*. This is more sophisticated than BM's `empire-strict` which uses a single factor.
- **MeanShift / DBSCAN clustering at aggregation time.** Brings scikit-learn into the hot path — new dependency, new runtime cost (MeanShift with `estimate_bandwidth(quantile=0.5, n_samples=50)` is not cheap for many clients).

### 6.3 What it drops

- The `register()` factory (dict lookup instead).
- `upper_bound()` / `influence()` introspection.
- `flatten`/`relink` (gradients are flattened ad-hoc each call).
- `experiments/` wrappers (`Model`, `Dataset`, `Loss`, `Criterion`, `Optimizer`, `Checkpoint`).
- The 17 study metrics.
- Momentum positions (`--momentum-at`).

## 7. Takeaways for the library plan

1. **SignGuard is the strongest case for "make the classical GARs easy to import."** A non-EPFL team wanted Krum/Bulyan/Median and reimplemented all three instead of depending on BM. If the library ships them as clean primitives, future SignGuard-style work can drop 150 lines of reimplementation.
2. **Adaptive attacks deserve a first-class abstraction.** `adaptive_attack.py` is a pattern: "run a search over a scalar parameter `γ`, using a metric that depends on the defense, until optimum." This is more reusable than writing 3 versions of essentially the same attack. The library could offer `AdaptiveAttack(base_attack, metric, search=binary|line)`.
3. **The MeanShift step is a preview of byzfl's pre-aggregator concept.** SignGuard's two-phase filter is effectively `preagg_clip_by_median_norm → cluster_and_select_largest → mean`. If the library adopts pre-aggregators as a composable pipeline (like byzfl already does), SignGuard ports trivially.
4. **Watch out for SignGuard's runtime bug** before citing it as a reference: [aggregators/__init__.py](../SignGuard/aggregators/__init__.py) imports from `.signcheck` but the file is `signguard.py`. Function names also mismatch (`signcheck_multiclass` vs. `signguard_multiclass`). The repo may not run out of the box.
5. **Evidence that BM's API is not universally sticky.** Unlike robust-collaborative-learning or Byzantine-Robust-Gossip, SignGuard's authors did not adopt the `register()` factory or any BM scaffolding. For a library that wants broad adoption outside EPFL, the API needs to be obviously better than "write a function and put it in a dict" — otherwise people will pick the dict.
