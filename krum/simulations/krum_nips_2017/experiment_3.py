"""Experiment 3 — NIPS 2017 (Blanchard et al.).

Multi-Krum vs Krum and Average under Gaussian attack on Spambase.
"""

from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.krum import Krum
from krum.primitives.aggregators.multikrum import MultiKrum
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

# Byzantine fraction (33%)
BYZ_FRACTION = 0.33

# Attack configuration
ATTACK_STD = 200.0

# MultiKrum parameter
MULTIKRUM_M = None  # None = auto (N - F - 2)

# Configurations to run
CONFIGS = [
    (Average, "Average_f0", 0, None),
    (Krum, "Krum_f6", None, None),  # f computed from fraction
    (MultiKrum, "MultiKrum_f6", None, None),  # f and m computed
]
# --------------------------------------------


def main() -> None:
    """Run Experiment 3."""
    train_set, test_set = spambase_dataset()
    attack_kw = {"std": ATTACK_STD}

    f = int(N * BYZ_FRACTION)
    m = MULTIKRUM_M if MULTIKRUM_M is not None else (N - f - 2)

    configs = [
        (Average, "Average_f0", 0, None),
        (Krum, f"Krum_f{f}", f, None),
        (MultiKrum, f"MultiKrum_f{f}", f, {"m": m}),
    ]

    for agg, label, f_val, agg_kw in configs:
        run_one(
            label=label,
            model_cls=MLPSpambase,
            train_set=train_set,
            test_set=test_set,
            aggregator=agg,
            aggregator_kwargs=agg_kw,
            attack=GaussianAttack,
            attack_kwargs=attack_kw,
            n=N,
            f=f_val,
            rounds=ROUNDS,
            batch_size=BATCH_SIZE,
            lr=LR,
            seed=SEED,
        )

    print("\nExperiment 3 done.")


if __name__ == "__main__":
    main()
