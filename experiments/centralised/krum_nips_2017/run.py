"""Shared helper to build and run one KrumSimulation instance."""

from typing import Any

import torch.nn as nn

from krum.orchestration import Metric
from krum.primitives.aggregators import Aggregator
from krum.primitives.attacks import Attack
from krum.simulations.centralised.krum_nips_2017 import KrumSimulation

from .datasets import make_datasets


def krum_experiment(
    *,
    label: str,
    dataset: str,
    model_cls: type[nn.Module],
    aggregator: type[Aggregator],
    attack: type[Attack],
    attack_kwargs: dict[str, Any] | None = None,
    n: int,
    f: int,
    rounds: int,
    batch_size: int,
    lr: float,
    seed: int,
    eval_every: int = 10,
    train_size: int = 0,
    test_size: int = 0,
    aggregator_kwargs: dict[str, Any] | None = None,
) -> None:
    """Build and run one KrumSimulation instance.

    The datasets are built from the hashable ``dataset`` name (and optional
    ``train_size``/``test_size``) *inside* this function, so the run is
    identified by those parameters rather than by the dataset objects.
    """
    print(f"\n=== {label} ===")
    train_set, test_set = make_datasets(dataset, train_size, test_size)
    krum_simulation = KrumSimulation(
        model_cls=model_cls,
        train_set=train_set,
        test_set=test_set,
        aggregator=aggregator,
        aggregator_kwargs=aggregator_kwargs,
        attack=attack,
        attack_kwargs=attack_kwargs,
        n=n,
        f=f,
        rounds=rounds,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        eval_every=eval_every,
    )

    krum_simulation.setup()

    test_loss = Metric("test_loss", float)
    test_accuracy = Metric("test_accuracy", float)
    train_loss = Metric("train_loss", float)

    for step in range(rounds):
        krum_simulation.step()

        if step % eval_every == 0:
            test_loss, test_accuracy = krum_simulation.evaluate()
            train_loss = krum_simulation.evaluate_train()

            test_loss.push(step, test_loss)
            test_accuracy.push(step, test_accuracy)
            train_loss.push(step, train_loss)

            print(f"step {step}")
            print(f"test_loss: {test_loss:.4f}, test_accuracy: {test_accuracy:.4f}, train_loss: {train_loss:.4f}")
