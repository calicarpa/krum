# Krum — NIPS 2017

Demonstrates Byzantine-resilient aggregation from:

> Blanchard, El Mhamdi, Guerraoui, Stainer.
> *"Machine learning with adversaries: Byzantine tolerant gradient descent."*
> NIPS 2017.

## Overview

This experiment compares **MultiKrum** and **Mean** (Average) under sign-flip
Byzantine attacks on Spambase. It demonstrates that MultiKrum resists adversarial
workers while Mean diverges.

Runs in ~30 seconds on CPU.

## Usage

```bash
uv run python -m experiments.centralised.krum_nips_2017.experiment_1
```

## Code Structure

```
experiments/centralised/krum_nips_2017/
├── datasets.py        # Spambase loader
├── experiment_1.py    # MultiKrum vs Mean under sign-flip attack
└── run.py             # Shared simulation runner
```

## Experiment

Compares Mean and MultiKrum under sign-flip attack with:
- **Dataset:** Spambase (57 features, binary classification)
- **Model:** MLP 57 → 20 (ReLU) → 20 (ReLU) → 2 (~1.6k params)
- **Workers:** n=20, Byzantine f=n/3=6
- **Attack:** Sign-flip (scale=10.0)
- **Rounds:** 100

**Phenomenon:** Mean diverges under attack (loss increases 70%+), while MultiKrum
converges normally, demonstrating Byzantine resilience.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.01 (fixed) |
| Number of workers `n` | 20 |
| Byzantine workers `f` | 0 or 6 (n/3) |
| Rounds | 100 |
| Batch size | 3 |
| Evaluation interval | every 5 rounds |
| Attack scale | 10.0 |
| Random seed | 42 |
