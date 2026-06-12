# Krum — NIPS 2017

Reproduces the experimental evaluation from:

> Blanchard, El Mhamdi, Guerraoui, Stainer.
> *"Machine learning with adversaries: Byzantine tolerant gradient descent."*
> NIPS 2017.

## Overview

This simulation package evaluates the **Krum** aggregation rule against Byzantine
workers in a synchronous parameter-server distributed SGD setting. It compares
Average, Krum, and Multi-Krum under Gaussian and Omniscient attacks on Spambase
and MNIST datasets.

## Usage

```bash
uv run python -m experiments.centralised.krum_nips_2017.experiment_1
uv run python -m experiments.centralised.krum_nips_2017.experiment_2
uv run python -m experiments.centralised.krum_nips_2017.experiment_3
```

## Code Structure

```
experiments/centralised/krum_nips_2017/
├── models.py          # MLPMnist, MLPSpambase
├── datasets.py        # MNIST and Spambase loaders
├── experiment_1.py    # Figure 4 — Resilience to Byzantine processes
├── experiment_2.py    # Figure 5 — Cost of resilience (batch size sweep)
└── experiment_3.py    # Figure 6 — Multi-Krum
```

## Models

| Model | Architecture | Parameters | Dataset |
|-------|-------------|------------|---------|
| `MLP` (from :file:`centralised/models.py`) | 784 → 100 (ReLU) → 10 | ≈ 8×10⁴ | MNIST |
| `MLPSpambase` | 57 → 20 (ReLU) → 20 (ReLU) → 2 | ≈ 1.6×10³ | Spambase |

## Experiments

### Experiment 1 — Resilience to Byzantine Processes (Fig. 4)

Compares Average and Krum under 0% and 33% Gaussian Byzantine workers on Spambase
(n=20, batch_size=3, 500 rounds).

**Attack:** Gaussian Byzantine (`std=200`), f = 33% (6 Byzantine out of 20)

### Experiment 2 — Cost of Resilience (Fig. 5)

Sweeps mini-batch sizes (3, 5, 10, 20, 40, 80, 160) comparing Average and Krum
under 0% and 45% Byzantine workers on both Spambase and MNIST.

**Attack:** Omniscient Byzantine (`kappa=100`), f = 45% (9 Byzantine out of 20)

### Experiment 3 — Multi-Krum (Fig. 6)

Compares Average (0%), Krum (33%), and Multi-Krum (`m = n - f - 2`) under
Gaussian Byzantine workers on Spambase (n=20, batch_size=3, 500 rounds).

**Attack:** Gaussian Byzantine (`std=200`), f = 33% (6 Byzantine out of 20)

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.01 (fixed, no decay) |
| Optimizer | Flat-tensor SGD (in-place, no momentum, no weight decay) |
| Number of workers `n` | 20 |
| Byzantine ratios `f/n` | 0%, 33% (f=6), 45% (f=9) |
| Rounds `T` | 500 |
| Evaluation interval | every 10 rounds |
| Random seed | 42 |

## Implementation Notes

- Uses a **fixed learning rate** (no scheduler, `lr_decay=None`).
- Reports a single **misclassification error rate** on the test set (inherits
  `evaluate_test_error_and_loss` from `CentralisedSimulation`).
- Simulation results are returned in-memory via `sim.run()`; no files are
  written to disk.
- `MLPSpambase` is defined in `models.py`; `MLP` (MNIST) is shared in
  :file:`centralised/models.py`.
