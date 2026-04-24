# ByzantineMomentum — Independent Code Analysis

> Written from a direct read of the source.
> Goal: scope what the codebase can and cannot do, and chart the lineage to the other repos
> sitting next to it in this folder, so we can plan turning it into a real library.

---

## 1. What this repo actually is

Code accompanying El-Mhamdi, Guerraoui & Rouault, *Distributed Momentum for Byzantine-resilient
Stochastic Gradient Descent* (ICLR 2021). It is **not a library**. It is a single-process,
**single-machine simulator** of synchronous parameter-server SGD where:

- All workers are simulated sequentially inside one Python process.
- A fixed fraction of those workers can be replaced by a chosen attack that can read every honest gradient and the GAR before crafting its reply.
- A chosen Gradient Aggregation Rule (GAR) replaces the average at the server.

Everything is driven from one CLI entry point, [attack.py](../ByzantineMomentum/attack.py), which builds the model, optimizer, dataset, GAR, attack, and runs the loop. Two helpers wrap it:

- [reproduce.py](../ByzantineMomentum/reproduce.py): generates the full sweep of `attack.py`
  invocations for the ICLR paper, dispatches them across GPUs via [tools/jobs.py](../ByzantineMomentum/tools/jobs.py).
- [study.py](../ByzantineMomentum/study.py): not a script — an importable module with `Session`,
  `LinePlot`, `BoxPlot` classes that read the per-step CSVs `attack.py` writes and produce the paper's plots.

---

## 2. Module map

```
ByzantineMomentum/
├── attack.py            ← single training loop, ~885 lines, the only entry point
├── reproduce.py         ← experiment sweep launcher
├── study.py             ← post-hoc plotting library (imports matplotlib, pandas, GTK)
├── reproduce-appendix.py
├── aggregators/         ← GAR registry + implementations
│   ├── __init__.py      ← register(), make_gar(), gars dict
│   ├── template.py      ← copy-paste skeleton for a new GAR
│   ├── average.py, krum.py, bulyan.py, median.py, trmean.py (+ phocas, meamed),
│   │   aksel.py, brute.py, cge.py
├── attacks/             ← attack registry + implementations
│   ├── __init__.py      ← register() (no make_gar equivalent)
│   ├── template.py
│   ├── identical.py     ← `bulyan`, `empire`, `little` attacks built from a generator
│   ├── empire.py        ← strict variant of "Fall of Empires"
│   ├── anticge.py, nan.py
├── experiments/         ← model / dataset / loss / optimizer / checkpoint wrappers
│   ├── __init__.py      ← auto-imports submodules
│   ├── model.py         ← Model wrapper (the central abstraction)
│   ├── dataset.py       ← Dataset wrapper, make_datasets(), batch_dataset()
│   ├── loss.py          ← Loss + Criterion classes (algebra: + and *)
│   ├── optimizer.py     ← thin wrapper around torch.optim
│   ├── checkpoint.py    ← Checkpoint + Storage (state_dict-protocol dict)
│   ├── configuration.py ← immutable dtype/device/non_blocking holder
│   ├── datasets/svm.py  ← custom SVM dataset
│   └── models/          ← simples-conv, simples-full, empire-cnn, wide_resnet
├── tools/               ← non-domain helpers
│   ├── __init__.py      ← Context (per-thread colored logging), import_directory()
│   ├── pytorch.py       ← flatten / relink / grad_of / grads_of / compute_avg_dev_max
│   ├── misc.py          ← parse_keyval, line_maximize, pairwise, UnavailableException
│   └── jobs.py          ← Command + Jobs runner for reproduce.py
└── submodules/wide-resnet
```

The **central trick** that everything else is built on top of lives in
[tools/pytorch.py](../ByzantineMomentum/tools/pytorch.py) — `flatten()` and `relink()`. See §4.

---

## 3. The training loop in plain English

[attack.py:685-885](../ByzantineMomentum/attack.py#L685-L885) is the whole show. Per step:

1. **Honest gradients.** For each of `nb_honests` simulated workers, sample a fresh batch and
   `model.backprop()` to get a gradient. Append to `grad_sampleds`. Optional L2 clip
   ([attack.py:776-779](../ByzantineMomentum/attack.py#L776-L779)). Loop is sequential — no real
   parallelism. The hot loop calls `grad.clone().detach_()` to copy out before the next backprop
   overwrites the in-place tensor.
2. **Apply momentum (if `worker` or `server`).** This is the paper's contribution. See §6.
3. **Adversary sees everything.** `attack.checked(grad_honests=..., f_decl=..., f_real=..., model=..., defense=...)`
   ([attack.py:818](../ByzantineMomentum/attack.py#L818)). The attack receives the honest list, the
   GAR closure, and the model — so an omniscient, GAR-aware adversary is the default threat model.
3. **Aggregate.** `defense.checked(gradients=honest+attack, f=nb_decl_byz, model=model, **gar_args)`
   ([attack.py:821](../ByzantineMomentum/attack.py#L821)). Single gradient out.
4. **Apply momentum (if `update`) and step.** `model.update(grad_defense)`
   ([attack.py:832-839](../ByzantineMomentum/attack.py#L832-L839)).
5. **Logging.** If `--result-directory` is set, ~17 metrics per step go to a TSV (variances,
   norms, max coords, all 6 pairwise cosines among sampled/honest/attack/defense gradients,
   composite curvature). Also periodic test accuracy and checkpoints.

`grad_honests` and `grad_attacks` are **plain lists of `torch.Tensor`**, all flat 1-D, all
sized `D = total parameter count`. That's the GAR/attack contract.

---

## 4. The flatten/relink mechanism — why everything is a 1-D tensor

[tools/pytorch.py:30-64](../ByzantineMomentum/tools/pytorch.py#L30-L64).

`flatten(tensors)` concatenates each parameter's `.view(-1)` into one contiguous buffer, then
`relink()` rewrites every original tensor's `.data` to a slice of that buffer using `.view(*shape)`.
After `flatten`, `model.parameters()` still iterates the original tensors with their original
shapes — but every read/write hits the shared 1-D buffer.

This happens once in `Model.__init__` ([experiments/model.py:170](../ByzantineMomentum/experiments/model.py#L170))
and the buffer hangs off `self._params` with `linked_tensors` attribute attached.

Two consequences:

- **`model.get()` returns a 1-D `Tensor` of length `D`.** It is the model. Mutating it mutates the
  network. ([experiments/model.py:262-267](../ByzantineMomentum/experiments/model.py#L262-L267))
- **`model.get_gradient()` does the same trick on the `.grad` slots** lazily on first call
  ([experiments/model.py:285-296](../ByzantineMomentum/experiments/model.py#L285-L296)). After that,
  every backprop accumulates into the same flat buffer.

`Model.backprop()` ([experiments/model.py:333-366](../ByzantineMomentum/experiments/model.py#L333-L366))
manually zeros the per-parameter `.grad` (which are slices of the flat buffer), runs forward+backward,
and returns the flat gradient by reference. **You must `.clone()` if you want to keep it across calls**
— the next `backprop()` will overwrite. The training loop does exactly this
([attack.py:780](../ByzantineMomentum/attack.py#L780), [attack.py:795](../ByzantineMomentum/attack.py#L795)).

This design is what makes GARs trivially expressible: every aggregator just operates on
`list[Tensor]` of identical 1-D shape; no need to traverse `state_dict` or worry about per-layer
shapes. It is the single most important architectural decision in the codebase.

### 4.1 What flattening actually saves (and what it doesn't)

The memory wins from this trick are mostly about *allocator fragmentation* and *temporary peak
memory*, not about total footprint. Worth spelling out because it's easy to overclaim:

- **Per-worker momentum buffers in `--momentum-at worker`.** This mode keeps `n` persistent
  momentum buffers across *all* of training. Flattened, each is one contiguous tensor of size
  `D`. Unflattened, each is `len(parameters)` small tensors. Total raw bytes are the same —
  but the allocator sees `n · len(parameters)` small blocks instead of `n` big ones, which on
  the CUDA caching allocator can mean the difference between a clean working set and a
  fragmented one that fails to coalesce after thousands of steps.
- **Temporary flat copies for the GAR.** If the GAR expects 1-D tensors and the code does not use
  the flatten trick, the natural path is to call `parameters_to_vector(...)` per worker each step,
  which allocates a *new* `D`-sized tensor every call — on top of the per-parameter `.grad`
  tensors that already exist. That doubles peak memory for a brief window every step. With the
  flatten trick the `.grad` slots *are* slices of the shared buffer, so there is no "flat copy
  vs. scattered original" duplication.
- **CUDA caching allocator behavior in long sweeps.** Many small tensors of varying sizes
  fragment the cache; one big buffer doesn't. ICLR-scale sweeps run thousands of steps and
  notice this even when a single step would fit comfortably.

What the trick does **not** avoid: the `n` `.clone()` calls in the training loop still allocate
`n · D` floats per step — one contiguous copy per honest worker, because `backprop()` overwrites
the shared gradient buffer on the next call. So the raw bytes copied per step are comparable
with or without flatten; what changes is *who* allocates them (one clean allocation vs. a
scatter-gather cat) and how friendly the result is to the CUDA allocator.

In short: the trick matters most for long runs on large models, where fragmentation and peak
memory dominate — not because it shrinks the total bytes a step needs.

### 4.2 Pros and cons of the flatten/relink trick

| Pros | Cons |
|---|---|
| One allocation at startup; zero per-step parameter-gathering allocations. | By-reference semantics: `model.get()` mutating the model is unusual, and `backprop()` overwriting the previous gradient is a footgun (callers must `.clone()`). |
| `model.get()` and `model.get_gradient()` are O(1). | Anything that **reassigns** `param.data` (some custom inits, weight pruning, quantization, partial `load_state_dict`) silently breaks the link. |
| GAR contract collapses to `list[Tensor(D,)] → Tensor(D,)`. New GARs are ~30 lines. | Per-layer aggregation rules ("median for batch-norm, trimmed mean for conv") are awkward — the buffer hides layer structure. |
| `torch.stack(list[Tensor(D,)])` produces one contiguous `(n, D)` matrix; sort/median/topk hit memory linearly. | Sparse or non-contiguous parameters don't fit cleanly: `view(-1)` requires contiguous memory. |
| One CUDA kernel per op instead of one per parameter — big launch-overhead win on small layers. | Order matters: `flatten` runs *after* the optional `nn.DataParallel` wrap ([model.py:168-170](../ByzantineMomentum/experiments/model.py#L168-L170)). Wrapping the model in something else later can break the link. |
| Persistent momentum buffers (worker mode) stay as one tensor each — checkpointable and small in metadata. | Hides what's really going on from a casual reader; debugging is harder when "this `Tensor` is actually a slice of that other one". |

### 4.3 GARs don't *require* flat tensors — it's a design choice

A common confusion: "the GAR needs flat tensors." It doesn't. Both kinds of operations GARs use
work fine on per-parameter lists:

```python
# Coordinate-wise median (e.g., for the median GAR):
flat_version = torch.stack(grads).median(dim=0).values

per_param_version = [torch.stack([g[k] for g in grads]).median(dim=0).values
                     for k in range(len(grads[0]))]

# Pairwise distance (e.g., for Krum/Bulyan):
flat_version = (g_i - g_j).norm()

per_param_version = torch.sqrt(sum((p_i - p_j).pow(2).sum()
                                   for p_i, p_j in zip(g_i, g_j)))
```

Same math, more code in the per-parameter version. The flat representation is chosen because it
(a) gives every GAR a uniform contract, (b) collapses per-parameter Python loops into one big
tensor op, and (c) keeps memory contiguous for sort/topk. None of these are *requirements* — they
are ergonomic and performance choices.

### 4.4 What modern PyTorch ships (post-2022)

The codebase was written in the 2018–2021 window. Several PyTorch features that would change the
calculus today did not exist yet, or were not stable:

- **`torch._foreach_*` ops** (stabilized PyTorch 1.13, late 2022): fused kernels over a list of
  tensors. `_foreach_add_(list, list, alpha=…)`, `_foreach_norm`, `_foreach_mul_`, etc. PyTorch's
  built-in optimizers switched to these around then. They give "one CUDA kernel for all
  parameters" without flattening — most of the kernel-launch win of `flatten` without the
  by-reference fragility.
- **`torch.func.functional_call` / `torch.func.grad` / `vmap`** (stable in PyTorch 2.0, 2023): pure
  functional model evaluation with parameters passed as a `dict[str, Tensor]`. `vmap` can
  parallelize across simulated workers in a single call — the *whole honest-gradient sampling
  loop* of `attack.py` could become one `vmap`. No mutation, no flattening, no by-reference
  surprises.
- **`torch.utils._pytree`**: flatten/unflatten arbitrary nested structures while keeping a "tree
  spec" you reuse to reconstruct. Underlies `torch.func`. Lets a GAR walk `dict[str, Tensor]`
  inputs cleanly.
- **`torch.nn.utils.parameters_to_vector` / `vector_to_parameters`**: existed all along, but
  copies on every call. Fine for occasional use, too slow for the inner loop ByzantineMomentum
  runs.
- **DDP gradient bucketing**: `DistributedDataParallel` already does *exactly* the
  flatten-and-relink trick internally to make AllReduce hit contiguous memory. ByzantineMomentum
  is just exposing that pattern to user code, which DDP keeps private.

So the trick isn't exotic — it's the standard high-perf trick for handling many parameters as one
buffer. PyTorch internalized it for DDP; ByzantineMomentum surfaces it for GAR ergonomics.

For a library written in 2025, the realistic options are:

1. **Keep `flatten/relink`.** Maximum performance, simplest GAR contract, but you have to document
   the by-reference and `.data`-reassignment hazards loudly. Backwards-compatible with all
   downstream EPFL repos.
2. **Switch to `torch._foreach_*` over per-parameter lists.** Almost as fast, no by-reference
   surprises, but the GAR contract becomes `list[list[Tensor]] → list[Tensor]` and every GAR has
   to walk the structure (or use `tree_map`). This is roughly what byzfl does.
3. **Go full functional with `torch.func`.** Cleanest code, supports `vmap` over workers, plays
   nicely with `torch.compile`. Different mental model — users no longer "have a model"; they
   have a function and a dict of weights. Bigger break from the existing API.

My read: option 1 stays the right default for a research simulator that needs to be deterministic
and fast on tens of millions of parameters. The fragmentation argument still holds today. The case
for option 2 or 3 is strongest if you're willing to break compatibility with the descendant repos
and target a broader audience that doesn't already know the EPFL idioms.

---

## 5. The GAR contract

Defined in the docstring of [aggregators/__init__.py:14-32](../ByzantineMomentum/aggregators/__init__.py#L14-L32).

A GAR module exposes:

| Function | Required | Signature |
|---|---|---|
| `aggregate(gradients, f, **kwargs) -> Tensor` | yes | flat list → flat tensor |
| `check(gradients, f, **kwargs) -> str \| None` | yes | parameter validation |
| `upper_bound(n, f, d) -> float` | optional | theoretical σ/‖g‖ ratio bound |
| `influence(honests, attacks, f, **kwargs) -> float` | optional | accepted-Byzantine ratio for the study CSV |

…and calls `register(name, aggregate, check, upper_bound=…, influence=…)` at module load.
`make_gar()` ([aggregators/__init__.py:42-69](../ByzantineMomentum/aggregators/__init__.py#L42-L69))
wraps the four functions and binds them as `func.checked`, `func.unchecked`, `func.check`,
`func.upper_bound`, `func.influence`. In Python's `__debug__` mode the default callable is
`checked` (validates inputs); under `python -OO` it's `unchecked` (fast path). `aggregators/`
is auto-scanned at import via `tools.import_directory()` so dropping a new file in is enough.

Same pattern for [attacks/__init__.py](../ByzantineMomentum/attacks/__init__.py): each attack exposes
`attack(grad_honests, f_decl, f_real, defense, model, **kwargs) -> list[Tensor]` and `check(...)`.
The attack list length must equal `f_real`; tensors may all be references to the same one.

GARs implemented out of the box: `average`, `krum` (Multi-Krum), `bulyan` (Bulyan over Multi-Krum),
`median` (coord-wise), `trmean` / `phocas` / `meamed`, `aksel`, `brute`, `cge`. Plus optional
`native-*` variants if a C/CUDA `native` module is importable.

Attacks: `nan`, `bulyan`, `empire`, `little` (all from [identical.py](../ByzantineMomentum/attacks/identical.py)),
`empire-strict` ([empire.py](../ByzantineMomentum/attacks/empire.py)), `anticge`. The three
`identical.py` attacks share an `eval_factor` line search ([identical.py:67-77](../ByzantineMomentum/attacks/identical.py#L67-L77)) — they
can probe the GAR to find the strongest attack scalar.

---

## 6. Where momentum is applied — the paper's actual contribution

[attack.py:195-198](../ByzantineMomentum/attack.py#L195-L198) and [attack.py:799-839](../ByzantineMomentum/attack.py#L799-L839).

`--momentum-at` is one of `update`, `server`, `worker`:

- **`update`** — classical SGD-with-momentum: the GAR gets the raw sampled gradients, momentum is
  applied to its output before the optimizer step. `grad_momentum_server.mul_(μ).add_(g_def, α=1-d)`.
- **`server`** — momentum is precomputed (it equals the previous step's `grad_defense`) and *added
  to each honest gradient* before the GAR sees them. The GAR aggregates "momentum-boosted" honest
  gradients. The momentum buffer is just last-step's defense output.
- **`worker`** — every honest worker keeps its own momentum buffer (`grad_momentum_workers[i]`).
  Each worker's gradient is `μ·m_i + (1-d)·g_i`, and *that* is what the GAR aggregates. The list
  of `μ`-momentum buffers is `nb_honests` flat tensors; they survive across steps and are
  checkpointed.

This is why [attack.py:642-678](../ByzantineMomentum/attack.py#L642-L678) cares about whether the
checkpoint stores a `list` of momentum buffers or a single one. The paper's claim is that
`worker`-side momentum strictly improves robustness for several GAR/attack pairs vs. classical
`update`-side momentum.

PyTorch's own `optimizer.step()` is configured with `momentum=0`, `dampening=0` — the wrapper
does momentum manually so it can put it in the right place ([attack.py:544](../ByzantineMomentum/attack.py#L544)).

---

## 7. Checkpointing — what's saved, what's not

[experiments/checkpoint.py](../ByzantineMomentum/experiments/checkpoint.py).

`Checkpoint` is a thin wrapper around `torch.save`/`torch.load` of a `dict[str, state_dict]`.
`snapshot(instance)` calls `instance.state_dict().copy()` (or `deepcopy` if `deepcopy=True`),
keyed by the instance's fully-qualified class name. `Storage(dict)` is a plain dict that
implements the `state_dict` / `load_state_dict` protocol, so arbitrary Python state can ride
along by stuffing it into a `Storage`.

What `attack.py` actually checkpoints:

- `Model._model` (real `nn.Module`) → all parameters
- `Optimizer._optim` → optimizer state (mostly empty here since momentum is manual)
- `storage` → `{version, steps, datapoints, momentum, origin}` where:
  - `momentum` = either a single flat tensor (for `update`/`server`) or a list of flat tensors
    (for `worker`)
  - `origin` = the initial flat parameters, only kept if studying is enabled, used to compute
    `‖θ - θ₀‖₂`

A bumped `version = 4` ([attack.py:622](../ByzantineMomentum/attack.py#L622)) gates load
compatibility. The checkpoint file is written each `--checkpoint-delta` steps as
`<result-dir>/checkpoint-<step>` and load is via `--load-checkpoint <path>`.

**What is *not* in the checkpoint** ([README:104-105](../ByzantineMomentum/README.md#L104-L105)):
PRNG state and the dataset loader's iterator state. So resuming from a checkpoint **does not give
bit-identical runs**. The tooling acknowledges this: setting `--seed` together with
`--load-checkpoint` is silently downgraded to no seed ([attack.py:298-300](../ByzantineMomentum/attack.py#L298-L300)).

There is no train/val state in the checkpoint either (no validation loop in the loop), and the
result CSVs are append-only files separate from the checkpoint.

---

## 8. Capabilities checklist (what works / what doesn't)

| Capability | Supported? | Notes |
|---|---|---|
| Plug a custom GAR | Yes | drop a file in `aggregators/`, `register(...)` it |
| Plug a custom attack | Yes | same pattern in `attacks/` |
| Plug a custom model | Yes | drop a constructor in `experiments/models/`, `__all__` it |
| Plug a custom dataset | Yes | drop a constructor in `experiments/datasets/`, `__all__` it |
| Get a flat gradient by reference | Yes | `model.get_gradient()` after `model.backprop()` |
| Get/set flat params | Yes | `model.get()` / `model.set(flat)` |
| Multiple GARs, mix-and-match per step | Yes | `--gars "krum,1;median,2;…"` randomly samples |
| Run GAR on a different device than model | Yes | `--device-gar`; tensors are migrated each step |
| Gradient clipping | Yes (L2 norm of the *sampled* honest gradient only) | `--gradient-clip` |
| Multiple local SGD steps before averaging | **Disabled** | `tools.fatal("Multi-steps SGD disabled until code review")` ([attack.py:798](../ByzantineMomentum/attack.py#L798)) |
| Nesterov momentum | Yes (manual, careful) | the comment at [attack.py:543](../ByzantineMomentum/attack.py#L543) warns the PyTorch impl is suspect |
| Checkpoint / resume | Yes (best-effort, not bit-exact) | see §7 |
| Bit-identical reproducibility from `--seed` | Yes if no checkpoint loaded | sets `cudnn.deterministic`, seeds torch+numpy |
| Real distributed execution (multiple processes / RPC) | **No** | everything is a Python `for` loop in one process |
| Decentralized / gossip topology | **No** | star/parameter-server only |
| Federated learning (clients with non-IID local data, multiple local epochs) | **No** | every honest "worker" samples from the same shared trainset |
| Differential privacy | **No** | no noise injection |
| Asynchronous / stragglers | **No** | strictly synchronous |
| Per-parameter / per-layer aggregation | Implicit via flatten | layers are recoverable via `params.linked_tensors` shapes, but no GAR uses this |
| Pip-installable / `setup.py` | **No** | no packaging metadata |
| Tests | **No** | no `tests/` directory |
| Public Python API for embedding in another script | Sort of | you can `import experiments, aggregators, attacks` from inside the source tree, but not from a sibling project |

---

## 9. Lineage to the other repos

These are findings from a sweep of the sibling directories. The marker for "this code is descended
from ByzantineMomentum" is concrete: same `register()` / `make_gar()` factory, same `flatten` /
`relink` helpers in `tools/pytorch.py`, EPFL/Rouault headers in source files.

### robust-collaborative-learning  ★ direct successor
The cleanest descendant. Same directory layout (`aggregators/`, `attacks/`, `experiments/`,
`tools/`), same registration machinery, same flatten/relink, same `Model`/`Checkpoint`/`Configuration`
wrappers, EPFL headers updated to 2018-2023. Adds ~8 GARs (cva, mva, rfa, mom, multikrum,
bucketing, filterL2…), more attacks (labelflipping, mimic, signflipping…), and **`peerToPeer.py`**
authored by John Stephan — the gossip extension that becomes the next repo.

### Byzantine-Robust-Gossip  ★ direct successor (gossip flavor)
Forked from robust-collaborative-learning (its README says so). Identical
`aggregators/__init__.py` registration. Adds `topology.py` + networkx, sparse peer-to-peer
scripts (`PeerToPeerSparseAVG.py`), and several gossip-specific GARs (robust_gossip, ios, mva,
cenna…). Multiple `reproduce_*.py` scripts per topology. ICML 2025.

### RPEL-BF2D  ◐ partial reuse
EPFL-authored (2023 headers). Keeps the `tools/` folder structurally. **Drops** the `register()`
factory: aggregators (`src/robust_aggregators.py`) and attacks (`src/byz_attacks.py`) are plain
free functions composed at the call site. No `experiments/` wrapper — uses raw `torch.nn.Module`s.
Adds gossip topology + Byzantine-epidemic peer-to-peer workers.

### SignGuard  ◐ partial / convergent
ICDCS 2022 federated learning code. Borrows the **EPFL `empire.py` attack** (still credited to
Rouault, 2019-2020). Aggregators reuse the names (Krum, Bulyan, Median…) but are **reimplemented**
in `aggregators/<Name>.py` with a simple dict factory, not the `register()` closure. No `tools/`,
no `experiments/`. New defense: SignGuard itself.

### DECOR  ◌ minimal reuse
Different problem (decentralized differential privacy via correlated noise). Only `tools/misc.py`
carries an EPFL header — no aggregator/attack registry, no GAR plumbing. Treat as parallel work
that touched the same toolbox briefly, not a descendant.

### byzfl  ★ rewrite as a real library
The DCL+INRIA pip-installable library. **Architectural inspiration only — no code reuse.**
Aggregators and attacks are **classes** (`TrMean(f=...)`, `SignFlipping()`), not registered free
functions. Has its own `fed_framework/` (server, client, byzantine_client, data_distributor)
instead of `experiments/`. Has `setup.py`, Sphinx docs, `tests/`, examples. This is what the
ByzantineMomentum codebase would look like if rebuilt from scratch as a library.

### Summary lineage diagram

```
ByzantineMomentum (ICLR 2021)
    │
    ├──► robust-collaborative-learning (ICML 2023)         ← extends, same plumbing
    │       │
    │       └──► Byzantine-Robust-Gossip (ICML 2025)       ← gossip extension
    │
    ├── ► RPEL-BF2D (2023)                                 ← keeps `tools/`, drops registry
    │
    ├── ► SignGuard (ICDCS 2022)                           ← borrows empire.py only
    │
    ├── ► DECOR                                            ← borrows tools/misc.py only
    │
    └──► byzfl (2025)                                      ← clean-slate rewrite as a library
```

---

## 10. Implications for "make ByzantineMomentum a library"

A few specific things stand out as friction or opportunities, in roughly decreasing order of
importance:

1. **byzfl already exists.** Before doing anything we should be honest about the overlap. The
   value-add of a new library has to be: (a) the *worker-momentum* trick that is this codebase's
   actual contribution and which byzfl does not appear to expose as a first-class option, (b) the
   17-metric per-step study CSV and the `study.py` plotting module, (c) the GAR `upper_bound` /
   `influence` introspection hooks, which byzfl's class API may or may not match.

2. **`attack.py` is monolithic.** 885 lines, all module-level code. To become a library, the
   training loop needs to become an iterator/class so the user can step it, inject hooks, and
   own the model. The current loop owns the result file descriptors, the signal handlers, the
   stdout printing, and the lr schedule — all need to be parameters or callbacks.

3. **The flatten/relink trick is the crown jewel.** Keep it, expose it, document it. It is what
   makes the GAR API trivial. byzfl's class-based aggregators effectively reproduce the same
   contract on shape-agnostic 1-D tensors.

4. **Single-process simulation is a feature, not a bug.** Researchers want determinism and small
   epoch times to sweep hyperparameters. Don't migrate to `torch.distributed` by default — keep
   the simulator and add an *optional* distributed backend.

5. **Drop the `import_directory` auto-loader for a real library.** Magic side-effect imports
   confuse static analysis and pip distribution. Replace with explicit registration in `__init__.py`
   or entry points (`importlib.metadata.entry_points('byzantine_momentum.gars')`), so plugins can
   live in third-party packages.

6. **Checkpoint reproducibility gap is fixable.** PRNG and dataloader iterator state are well-defined
   things; we just need to opt into saving them. This is a small, isolated change.

7. **The "multi local steps" path is dead code** ([attack.py:796-798](../ByzantineMomentum/attack.py#L796-L798))
   but federated learning fundamentally needs it. Before generalizing toward FL, that branch needs
   the promised review.

8. **GAR/attack interfaces are fine as-is.** Both contracts (`aggregate(gradients, f, **kwargs)` /
   `attack(grad_honests, f_decl, f_real, defense, model, **kwargs)`) are clean and have already
   proven extensible — three downstream repos kept them verbatim. No reason to break them.

9. **`tools.Context` (per-thread colored logging via stdout wrapping) and `tools.import_directory`
   are anti-library patterns.** They mutate `sys.stdout` and `sys.excepthook` at import time
   ([tools/__init__.py:215-216, 246](../ByzantineMomentum/tools/__init__.py#L215-L216)). Replace
   with the standard `logging` module before anyone else's app accidentally inherits these wrappers.

10. **`study.py` requires GTK 3.0 at import time.** It degrades gracefully but the dependency is
    surprising. A library version should split: `metrics.py` (CSV → DataFrame) with no GUI
    dependency, plus an optional `plotting` extra.
