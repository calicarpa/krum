"""Second Experiment for the Krum-NIPS-2017 simulation (Cost of Resilience).

Reproduces Figure 5: compares cross-validation error at round 500 for
Average and Krum under 0% and 45% Byzantine workers across different
mini-batch sizes on both Spambase and MNIST.
"""

from typing import Any

import torch.nn as nn
from torch.utils.data import Dataset

from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.krum import Krum
from krum.primitives.attacks.full_gradient_negation import FullGradientNegationAttack
from krum.simulations.common.models import MLP as MLPMnist

from ._common import run_one
from .datasets import mnist_dataset, spambase_dataset
from .models import MLPSpambase


def main() -> None:
    """Run Experiment 2: Cost of Resilience (Figure 5).

    Sweeps mini-batch sizes from 3 to 160 on both Spambase and MNIST,
    comparing Average and Krum under 0% and 45% Byzantine workers.
    """
    rounds = 500
    n = 20
    f_list = [0, 9]
    batch_sizes = [3, 5, 10, 20, 40, 80, 160]
    lr = 0.01
    seed = 42

    datasets: list[tuple[str, tuple[Dataset[Any], Dataset[Any]], type[nn.Module]]] = [
        ("spambase", spambase_dataset(), MLPSpambase),
        ("mnist", mnist_dataset(), MLPMnist),
    ]

    for ds_name, (train_set, test_set), model_cls in datasets:
        for f in f_list:
            for bs in batch_sizes:
                for agg, agg_label in [
                    (Average, "Average"),
                    (Krum, "Krum"),
                ]:
                    attack = FullGradientNegationAttack
                    attack_kw: dict[str, Any] = {"kappa": 100.0}
                    label = f"{ds_name}_{agg_label}_f{f}_bs{bs}"
                    run_one(
                        label=label,
                        model_cls=model_cls,
                        train_set=train_set,
                        test_set=test_set,
                        aggregator=agg,
                        attack=attack,
                        attack_kwargs=attack_kw,
                        n=n,
                        f=f,
                        rounds=rounds,
                        batch_size=bs,
                        lr=lr,
                        seed=seed,
                    )

    print("\nExperiment 2 done.")


if __name__ == "__main__":
    main()
