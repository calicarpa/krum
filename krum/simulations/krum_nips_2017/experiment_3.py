"""Third Experiment for the Krum-NIPS-2017 simulation (Multi-Krum).

Reproduces Figure 6: compares Average (0% Byzantine), Krum (33%),
and Multi-Krum (33%) under Gaussian Byzantine workers on Spambase.
"""

from typing import Any

from krum.primitives.aggregators import Aggregator
from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.krum import Krum
from krum.primitives.aggregators.multikrum import MultiKrum
from krum.primitives.attacks.gaussian import GaussianAttack

from ._common import run_one
from .datasets import spambase_dataset
from .models import MLPSpambase


def main() -> None:
    """Run Experiment 3: Multi-Krum (Figure 6).

    Compares Average (0%), Krum (33%), and Multi-Krum with m=n-f (33%)
    under Gaussian Byzantine workers on Spambase (n=20, batch_size=3,
    500 rounds).
    """
    seed = 42
    rounds = 500
    n = 20
    f = int(n * 0.33)
    batch_size = 3
    lr = 0.01

    train_set, test_set = spambase_dataset()
    attack = GaussianAttack
    attack_kw: dict[str, Any] = {"std": 200.0}

    configs: list[tuple[type[Aggregator], str, int, dict[str, Any] | None]] = [
        (Average, "Average_f0", 0, None),
        (Krum, "Krum_f6", f, None),
        (MultiKrum, "MultiKrum_f6", f, {"m": n - f - 2}),
    ]

    for agg, label, f_val, agg_kw in configs:
        run_one(
            label=label,
            model_cls=MLPSpambase,
            train_set=train_set,
            test_set=test_set,
            aggregator=agg,
            aggregator_kwargs=agg_kw,
            attack=attack,
            attack_kwargs=attack_kw,
            n=n,
            f=f_val,
            rounds=rounds,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
        )

    print("\nExperiment 3 done.")


if __name__ == "__main__":
    main()
