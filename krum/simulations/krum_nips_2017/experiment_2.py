"""Experiment 2 — NIPS 2017 (Blanchard et al.).

Cost of Resilience: sweep batch sizes on Spambase and MNIST.
"""

from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.krum import Krum
from krum.primitives.attacks.full_gradient_negation import FullGradientNegationAttack
from krum.simulations.common.models import MLP as MLPMnist

from ._common import run_one_simulation
from .datasets import mnist_dataset, spambase_dataset
from .models import MLPSpambase

# --- Configurable parameters ---
ROUNDS = 500
N = 20
LR = 0.01
SEED = 42

# Attack configuration
ATTACK_KAPPA = 100.0

# Batch sizes to sweep
BATCH_SIZES = [3, 5, 10, 20, 40, 80, 160]

# Byzantine worker counts to test
F_VALUES = [0, 9]  # 0% and 45%

# Aggregators to compare
AGGREGATORS = [
    (Average, "Average"),
    (Krum, "Krum"),
]

# Datasets to test
DATASETS = [
    ("spambase", spambase_dataset, MLPSpambase),
    ("mnist", mnist_dataset, MLPMnist),
]
# --------------------------------------------


def main() -> None:
    """Run Experiment 2."""
    attack_kw = {"kappa": ATTACK_KAPPA}

    for ds_name, ds_fn, model_cls in DATASETS:
        train_set, test_set = ds_fn()
        for f in F_VALUES:
            for bs in BATCH_SIZES:
                for agg, agg_label in AGGREGATORS:
                    label = f"{ds_name}_{agg_label}_f{f}_bs{bs}"
                    run_one_simulation(
                        label=label,
                        model_cls=model_cls,
                        train_set=train_set,
                        test_set=test_set,
                        aggregator=agg,
                        attack=FullGradientNegationAttack,
                        attack_kwargs=attack_kw,
                        n=N,
                        f=f,
                        rounds=ROUNDS,
                        batch_size=bs,
                        lr=LR,
                        seed=SEED,
                    )

    print("\nExperiment 2 done.")


if __name__ == "__main__":
    main()
