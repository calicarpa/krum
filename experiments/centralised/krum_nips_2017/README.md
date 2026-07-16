# Krum — NIPS 2017

Demonstrates Byzantine-resilient aggregation from:

> Blanchard, El Mhamdi, Guerraoui, Stainer.
> *"Machine learning with adversaries: Byzantine tolerant gradient descent."*
> NIPS 2017.

## Overview

Compares **MultiKrum** and **Mean** (plain averaging) under sign-flip Byzantine
attacks on Spambase. Four configurations are run: each aggregator with 0 and 6
Byzantine workers (f = n/3). MultiKrum converges regardless of attack; Mean
diverges catastrophically.

Runs in ~60 seconds on CPU.

## Usage

```bash
uv run python -m experiments.centralised.krum_nips_2017.experiment
```

## Code Structure

```
experiments/centralised/krum_nips_2017/
├── datasets.py     # Spambase loader
├── experiment.py   # MultiKrum vs Mean under sign-flip attack
└── run.py          # Shared simulation runner
```

## Experiment

Compares Mean and MultiKrum under sign-flip attack with:
- **Dataset:** Spambase (57 features, binary classification)
- **Model:** MLP 57 → 20 (ReLU) → 20 (ReLU) → 2 (~1.6k params)
- **Workers:** n = 20, Byzantine f = n/3 = 6
- **Attack:** Sign-flip (scale = 10.0)
- **Rounds:** 300

**Phenomenon:** MultiKrum resists Byzantine workers (reaches ~83% accuracy even
with 6 adversaries), while Mean diverges under attack — loss explodes and the
model collapses to random guessing.

## Hyperparameters

| Parameter           | Value             |
|---------------------|-------------------|
| Learning rate       | 0.01 (fixed)      |
| Number of workers n | 20                |
| Byzantine workers f | 0 or 6 (n/3)      |
| Rounds              | 300               |
| Batch size          | 3                 |
| Evaluation interval | every 15 rounds   |
| Attack scale        | 10.0              |
| Weight decay        | 1e-4              |
| Weight init         | Xavier uniform    |
| Random seed         | 42                |

## Results

Four curves across three panels: test loss, train loss, and test accuracy.

### Loss curves (test and train)

- **Mean_f0** (green, solid) and **MultiKrum_f0** (orange, solid): steady
  convergence — loss decreases smoothly from ~0.79 to ~0.42 over 300 rounds.
  Both aggregators perform equivalently when no attack is present.

- **Mean_f6** (red, dashed): diverges explosively. Loss climbs from 0.80 to
  infinity within ~75 rounds, then becomes NaN. The sign-flip attack amplifies
  gradients in the wrong direction, and plain averaging offers no protection.

- **MultiKrum_f6** (blue, dashed): converges normally despite 6 Byzantine
  workers. Loss decreases to ~0.44 — only marginally higher than the no-attack
  baseline. MultiKrum's scoring-based selection filters out adversarial
  gradients.

### Accuracy curve

- **Mean_f0** and **MultiKrum_f0**: both reach ~85% accuracy by step 285. The
  model is still improving — training has not fully converged.

- **Mean_f6**: stays at ~40% (the proportion of spam in the dataset) while
  weights remain finite — the exploding loss drives the model to always predict
  the spam class. After step 75, when weights become NaN, `argmax` on NaN
  tensors defaults to class 0 (non-spam, ~60% of the data), producing a
  spurious jump to ~60% accuracy with no actual learning.

- **MultiKrum_f6**: reaches ~83% accuracy, only 1-2 percentage points behind
  the attack-free runs. Demonstrates that MultiKrum is Byzantine-resilient up
  to f < n/2.
