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

### Workers

- Total workers: `n = 20` (fixed across all experiments).
- Each honest worker receives an IID shard of the full training set.
- Byzantine ratios: `33%` (f = 6–7) and `45%` (f = 9).
- Benchmark `f = 0` (no attack) as baseline.

### Attacker Knowledge

The adversary observes **all honest gradients at time t** before computing the Byzantine gradients.
The omniscient attacker additionally computes the gradient on the **full dataset** (not a mini-batch).

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

Each Byzantine worker computes the gradient on the **full dataset** (very accurate estimate),
then sends the **opposite vector scaled to a large length**:

```
byz_g = -κ · ∇L(full_dataset)
```

with κ chosen large enough to dominate the aggregation.

### Aggregation Rules (compared)

| Aggregator | Description |
|---|---|
| **Average** | Plain average. `agg = mean(all gradients)` |
| **Krum** | Single Krum. Selects the gradient minimizing the sum of squared distances to its `n - f - 2` closest neighbors. |
| **Multi-Krum** | Averages the `m = n - f` gradients with smallest Krum scores. Interpolates between Krum (m = 1) and Average (m = n). |

### Experiments

#### Experiment 1 — Resilience to Byzantine processes (Fig. 4)

- **Attack**: Gaussian Byzantine (`N(0, 200²)`), f = 33%
- **Mini-batch size**: 3 per worker
- **Compare**: Average vs Krum, with and without Byzantine workers
- **Metric**: cross-validation error over rounds
- **Expected**: Krum with 33% Byzantine matches Average with 0% Byzantine

#### Experiment 2 — Cost of Resilience (Fig. 5)

- **Attack**: Omniscient Byzantine, f = 45%
- **Mini-batch size**: swept from 3 to ~160
- **Compare**: Average vs Krum at round 500
- **Metric**: cross-validation error at round 500 vs batch size
- **Expected**: with batch size ≥ 10, Krum with 45% Byzantine matches Average with 0% Byzantine

#### Experiment 3 — Multi-Krum (Fig. 6)

- **Attack**: Gaussian Byzantine (`N(0, 200²)`), f = 33%
- **Mini-batch size**: 3 per worker
- **Multi-Krum parameter**: `m = n - f`
- **Compare**: Average (0%), Krum (33%), Multi-Krum (33%)
- **Expected**: Multi-Krum with 33% Byzantine converges as fast as Average with 0% Byzantine

### Round Schedule

Synchronous rounds:

```
for round t ∈ [1, T]:
    param_server.broadcast(θ_t)
    for each worker w:
        g_w = SGD_step(w.dataset, θ_t, batch_size)
        if w is Byzantine: g_w = attack.generate(...)
    θ_{t+1} = aggregator.aggregate([g_1, ..., g_n])
    θ_{t+1} = θ_t - lr × aggregated_gradient
    track_metrics(model, test_set)
```

Training runs for up to `T = 500` rounds.

### Metrics (tracked every round)

1. **Cross-validation error** — on a held-out validation split.
2. **Cross-validation loss** (cross-entropy).

### Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 0.01 |
| Mini-batch size (base) | 3 |
| Mini-batch sweep (Fig. 5) | 3–160 |
| Number of workers `n` | 20 |
| Byzantine ratio `f/n` | 0%, 33%, 45% |
| T (rounds) | 500 |
| Multi-Krum `m` | `n − f` |

Note: the paper does not specify the learning rate explicitly; 0.01 is extrapolated
from the AggregaThor reference implementation by the same lab.

## Output Artifacts

Per simulation run:

- CSV trace: `(round, cross_val_error, cross_val_loss)`.
- Metadata: aggregator, attack, n, f, dataset, batch_size, seed.
- Plots: error vs round (Fig. 4, Fig. 6), error at round 500 vs batch size (Fig. 5).

## Implementation Notes

- Use the library's existing `Krum`, `MultiKrum`, `Average` aggregators.
- Implement `GaussianAttack` and `OmniscientAttack` as new `Attack` subclasses.
- MNIST via `torchvision.datasets.MNIST`.
- The simulation loop is intentionally explicit — see `notes/adr-2026-05-05.md`.
