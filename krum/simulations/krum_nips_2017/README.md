# Machine Learning with Adversaries

**Paper:** [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://papers.nips.cc/paper_files/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf)
— Blanchard, El Mhamdi, Guerraoui, Stainer — NIPS 2017

## Goal

Reproduce the experimental evaluation of the Krum aggregation rule against Byzantine
workers in a synchronous parameter-server distributed SGD setting, on MNIST.

## Architecture

```
Parameter Server ← gradients from n workers (up to f Byzantine)
       │
       │ broadcast parameters
       ▼
┌───────────────────────────────────────────────────────────┐
│   Worker 1   │   Worker 2   │   ...   │     Worker n      │
│   (honest)   │   (honest)   │         │ (maybe Byzantine) │
└───────────────────────────────────────────────────────────┘
       │
       ▼
  Local SGD on dataset shard
```

## Simulation Protocol

### Model

- **MNIST** — Multi-Layer Perceptron with **one hidden layer** of size 100
  (784 → 100 → 10). ReLU on the hidden layer, no softmax on output
  (included in CrossEntropyLoss). Total ≈ 8 × 10⁴ parameters.

- **SpamBase** — Multi-Layer Perceptron with **two hidden layers** of size 20
  (57 → 20 → 20 → 2). ReLU activations. Standardized features. Total ≈ 1.6 × 10³
  parameters.

### Workers

- Total workers: `n = 20` (fixed across all experiments).
- Each honest worker receives an IID shard of the full training set.
- Byzantine ratios: **33%** (f = 6) and **45%** (f = 9).
- Benchmark `f = 0` (no attack) as baseline.
- Each honest worker computes its gradient on **one mini-batch** per round
  from its local shard (stochastic).

### Attacker Knowledge

The adversary observes **all honest gradients at time t** before computing
the Byzantine gradients. The omniscient attacker additionally computes the
gradient on the **full training set** (not a mini-batch).

### Attacks

The paper tests two attack families:

#### Gaussian Byzantine (used in Fig. 4 and Fig. 6)

Byzantine workers send vectors drawn from a Gaussian distribution with **mean zero**
and isotropic covariance matrix with **standard deviation 200**:

```
byz_g = N(0, 200² · I_d)
```

This attack is independent of the honest gradients (non-omniscient).

#### Omniscient Byzantine (used in Fig. 5)

Each Byzantine worker computes the gradient on the **full dataset** (very accurate
estimate), then sends the **opposite vector scaled to a large length**:

```
byz_g = −κ · ∇L(full_dataset)
```

with `κ = 100` chosen large enough to dominate the aggregation.

### Aggregation Rules (compared)

| Aggregator | Description |
|---|---|
| **Average** | Plain average. `agg = mean(all gradients)` |
| **Krum** | Selects the gradient minimizing the sum of squared distances to its `n − f − 2` closest neighbors. |
| **Multi-Krum** | Averages the `m = n − f` gradients with smallest Krum scores. Interpolates between Krum (m = 1) and Average (m = n). |

The Krum score for gradient `i` is:

```
s(i) = Σ_{j ∈ N_i} ‖V_i − V_j‖²
```

where `N_i` is the set of `n − f − 2` closest neighbors of `V_i` (by Euclidean
distance, excluding self). The gradient with smallest score is selected (m = 1 for
Krum; the `m` smallest are averaged for Multi-Krum).

The Byzantine resilience guarantee requires `n ≥ 2f + 3`.

### Experiments

#### Experiment 1 — Resilience to Byzantine processes (Fig. 4)

- **Attack**: Gaussian Byzantine (`N(0, 200²)`), f = 33% (6 Byzantine out of 20)
- **Dataset**: SpamBase
- **Mini-batch size**: 3 per worker
- **Compare**: Average vs Krum, with and without Byzantine workers (f = 0, f = 6)
- **Metric**: cross-validation error over rounds
- **Expected**: Krum with 33% Byzantine matches Average with 0% Byzantine

#### Experiment 2 — Cost of Resilience (Fig. 5)

- **Attack**: Omniscient Byzantine, f = 45% (9 Byzantine out of 20)
- **Datasets**: SpamBase and MNIST
- **Mini-batch size**: swept from 3 to 160 (log₂ scale: 3, 5, 10, 20, 40, 80, 160)
- **Compare**: Average vs Krum, for f = 0 and f = 9
- **Metric**: cross-validation error at round 500 vs batch size
- **Expected**: with batch size ≥ 10, Krum with 45% Byzantine matches Average with 0% Byzantine

#### Experiment 3 — Multi-Krum (Fig. 6)

- **Attack**: Gaussian Byzantine (`N(0, 200²)`), f = 33% (6 Byzantine out of 20)
- **Dataset**: SpamBase
- **Mini-batch size**: 3 per worker
- **Multi-Krum parameter**: `m = n − f = 14`
- **Compare**: Average (f = 0), Krum (f = 6), Multi-Krum (f = 6)
- **Expected**: Multi-Krum with 33% Byzantine converges as fast as Average with 0% Byzantine

### Round Schedule

Synchronous rounds, executed sequentially:

```
for round t ∈ [0, T):
    # Parameter server broadcasts current model
    # (implicit — all workers share the same model in this simulation)

    # Each honest worker computes a gradient on its local shard
    for w in honest_workers:
        g_w = SGD_step(w.dataset, θ_t, batch_size)

    # Byzantine workers generate attack gradients
    if f > 0:
        if attack is Omniscient:
            attack.set_full_gradient(∇L(full_dataset, θ_t))
        byz_g = attack.generate(honest_gradients, f)

    # Parameter server aggregates and updates
    θ_{t+1} = θ_t − lr × aggregator([g_1, ..., g_n])

    # Evaluate on test set every eval_every rounds
    if t % eval_every == 0:
        error, loss = evaluate(model, test_set)
```

Training runs for up to `T = 500` rounds. Evaluation is performed on the **full
test set** (no mini-batching) at the start of training (round 0) and every
`eval_every` rounds (default: 10), plus the final round.

### Metrics (tracked every evaluation step)

1. **Cross-validation error** — fraction of misclassified samples on the test set, in `[0, 1]`.
2. **Cross-validation loss** — cross-entropy loss on the test set.

Both are computed on the full test set in a single forward pass (deterministic,
no dropout or batch normalization).

### Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 0.01 |
| Optimizer | SGD (no momentum, no weight decay) |
| Mini-batch size (base) | 3 |
| Mini-batch sweep (Fig. 5) | 3, 5, 10, 20, 40, 80, 160 |
| Number of workers `n` | 20 |
| Byzantine ratio `f/n` | 0%, 33% (f = 6), 45% (f = 9) |
| Rounds `T` | 500 |
| Multi-Krum `m` | `n − f` (14 for n = 20, f = 6) |
| Gaussian attack `σ` | 200 |
| Omniscient attack `κ` | 100 |
| Evaluation interval | every 10 rounds |
| Random seed | 42 |

Note: the paper does not specify the learning rate explicitly; 0.01 is extrapolated
from the AggregaThor reference implementation by the same lab.

## Output Artifacts

Per simulation run, when `results_dir` is configured:

### PyTorch trace

Saved as `{label}.pt` with a dictionary:

```python
{
    "traces": [(round, error, loss), ...],  # list of 3-tuples
    "label": "Krum_f6",                      # human-readable label
    "seed": 42                               # random seed used
}
```

### Figures

- **Figure 4** (`figure_4.png`): two side-by-side subplots — 0% and 33% Byzantine,
  error vs rounds, Average vs Krum.
- **Figure 5** (`figure_5.png`): two side-by-side subplots — SpamBase and MNIST,
  error at round 500 vs batch size (log₂ scale), 4 curves (Average/Krum × 0%/45%).
- **Figure 6** (`figure_6.png`): single plot — Average (0%), Krum (33%), Multi-Krum (33%),
  error vs rounds.

## Implementation Notes

- Uses the library's existing `Krum`, `MultiKrum`, and `Average` aggregators from
  `krum.primitives.aggregators`.
- Uses `GaussianAttack` and `OmniscientAttack` from `krum.primitives.attacks`.
- `MLPMnist` and `MLPSpambase` are defined locally in `models.py` (not part of
  the library core).
- The simulation loop is intentionally explicit — see `notes/adr-2026-05-05.md`
  for the design rationale.

## References

- [Krum paper (NIPS 2017)](https://papers.nips.cc/paper_files/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf)
- [AggregaThor — reference implementation by LPD-EPFL](https://github.com/LPD-EPFL/AggregaThor)
- [SpamBase dataset (UCI ML Repository)](https://archive.ics.uci.edu/ml/datasets/spambase)
- [ADRs for this project](../../notes/)
