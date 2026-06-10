"""Experiment 1 — NIPS 2017 (Blanchard et al.).

Resilience to Gaussian Byzantine workers on Spambase.
"""

from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.krum import Krum
from krum.primitives.attacks.gaussian import GaussianAttack

from ._common import run_one
from .datasets import spambase_dataset
from .models import MLPSpambase

# --- Configurable parameters ---
ROUNDS = 500
N = 20
BATCH_SIZE = 3
LR = 0.01
SEED = 42

# Attack configuration
ATTACK_STD = 200.0

# Byzantine fractions to test (0.0 = 0%, 0.33 = 33%)
BYZ_FRACTIONS = [0.0, 0.33]

# Aggregators to compare
AGGREGATORS = [
    (Average, "Average"),
    (Krum, "Krum"),
]
# --------------------------------------------


def main() -> None:
    """Run Experiment 1."""
    train_set, test_set = spambase_dataset()
    attack_kw = {"std": ATTACK_STD}

    for fraction in BYZ_FRACTIONS:
        f = int(N * fraction)
        for agg, agg_label in AGGREGATORS:
            label = f"{agg_label}_f{f}"
            run_one(
                label=label,
                model_cls=MLPSpambase,
                train_set=train_set,
                test_set=test_set,
                aggregator=agg,
                attack=GaussianAttack,
                attack_kwargs=attack_kw,
                n=N,
                f=f,
                rounds=ROUNDS,
                batch_size=BATCH_SIZE,
                lr=LR,
                seed=SEED,
            )

    print("\nExperiment 1 done.")


if __name__ == "__main__":
    main()
