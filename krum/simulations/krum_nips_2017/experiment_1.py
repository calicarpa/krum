"""First Experiment for the Krum-NIPS-2017 simulation (Resilience to Byzantine processes).

Reproduces Figure 4: compares Average and Krum under 0% and 33% Gaussian
Byzantine workers on Spambase.
"""

from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.krum import Krum
from krum.primitives.attacks.gaussian import GaussianAttack

from .datasets import spambase_dataset
from .models import MLPSpambase
from .simulation import KrumSimulation


def main() -> None:
    """Run Experiment 1: Resilience to Byzantine processes (Figure 4).

    Compares Average and Krum under 0% and 33% Gaussian Byzantine workers
    (n=20, batch_size=3, 500 rounds on Spambase).
    """
    seed = 42
    rounds = 500
    n = 20
    batch_size = 3
    lr = 0.01
    f_list = [0, int(n * 0.33)]

    train_set, test_set = spambase_dataset()
    attack = GaussianAttack(std=200.0)

    for f in f_list:
        for agg, agg_label in [
            (Average, "Average"),
            (Krum, "Krum"),
        ]:
            label = f"{agg_label}_f{f}"
            print(f"\n=== {label} ===")

            sim = KrumSimulation(
                model_cls=MLPSpambase,
                train_set=train_set,
                test_set=test_set,
                aggregator=agg,
                attack=attack,
                n=n,
                f=f,
                rounds=rounds,
                batch_size=batch_size,
                lr=lr,
                seed=seed,
            )
            sim.run()

    print("\nExperiment 1 done.")


if __name__ == "__main__":
    main()
