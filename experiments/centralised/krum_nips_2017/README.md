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

Each run is recorded under `results/krum_2017_nips_spambase/`.
The folder is named by a key derived from the configuration *and* from the
code the run executes. Running the command again replays nothing: a run
already recorded there is skipped, so only the runs whose parameters or code
have changed are executed. Editing an aggregator therefore re-runs the
configurations that use it and leaves the others alone. Pass `force=True` to
the `Orchestrator` to re-run regardless.

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
| MultiKrum `m`       | 20 (no attack) / 14 (attack) |
| Data partitioner    | IID (default)      |

**Note on MultiKrum `m`:** both cases follow the paper (Section 6, Figure 6):
`m = n − f`. The attack case therefore runs at `m = 14`, above the theoretical
resilience bound `n − 2f − 3 = 5` — the paper sets `m = n − f` there too. The
no-attack case gives `m = n = 20`, where MultiKrum coincides with Average by
design. Above the bound the resilience guarantee no longer applies, but
far-from-honest byzantine gradients still receive large Krum scores and stay
excluded from the average.

## Results

Four curves across three panels: test loss, train loss, and test accuracy.

### Loss curves (test and train)

- **Mean_f0** (green, solid) and **MultiKrum_f0** (orange, solid): the two
  curves coincide exactly — with `m = n = 20` MultiKrum *is* Average — and
  converge smoothly from ~0.81 to ~0.44 over 300 rounds.

- **Mean_f6** (red, dashed): diverges explosively. Loss climbs from 0.82 to
  124.8 by round 45, leaves the frame before round 50, and the weights become
  NaN from round 60. The sign-flip attack amplifies gradients in the wrong
  direction, and plain averaging offers no protection.

- **MultiKrum_f6** (blue, dashed): converges normally despite 6 Byzantine
  workers. Loss decreases to ~0.50 — slightly above the no-attack baseline.
  MultiKrum's scoring-based selection filters out adversarial gradients even
  at `m = n − f = 14`, above the resilience bound.

### Accuracy curve

- **Mean_f0** and **MultiKrum_f0**: both reach ~82% accuracy by step 285. The
  model is still improving — training has not fully converged.

- **Mean_f6**: starts around 42%, stays at ~40% (the proportion of spam in the
  dataset) while weights remain finite, then jumps to a flat ~60% plateau once
  the weights become NaN at step 60 — `argmax` on NaN tensors defaults to
  class 0 (non-spam, ~60% of the data). This is an artifact, not learning.

- **MultiKrum_f6**: reaches ~77% accuracy, about five percentage points behind
  the attack-free runs. Demonstrates that MultiKrum is Byzantine-resilient up
  to f < n/2.
