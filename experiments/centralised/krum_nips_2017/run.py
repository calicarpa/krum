"""Shared helper to build and run one KrumSimulation instance."""

from typing import Any

import torch.nn as nn
from torch.utils.data import Dataset

from krum.orchestration import Metric
from krum.primitives.aggregators import Aggregator
from krum.primitives.attacks import Attack
from krum.simulations.centralised import KrumSimulation


def krum_experiment(
    *,
    label: str,
    model_cls: type[nn.Module],
    train_set: Dataset[Any],
    test_set: Dataset[Any],
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
    aggregator_kwargs: dict[str, Any] | None = None,
) -> list[tuple[int, Any]]:
    """Build and run one KrumSimulation instance."""
    print(f"\n=== {label} ===")
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

    loss = Metric("loss", float)
    error = Metric("error", float)

    for step in range(rounds):
        krum_simulation.step()
        loss_value, error_value = krum_simulation.evaluate()

        loss.push(step, loss_value)
        error.push(step, error_value)
