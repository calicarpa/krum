"""Shared helper to build and run one KrumSimulation instance."""

from typing import Any

import torch.nn as nn
from torch.utils.data import Dataset

from krum.primitives.aggregators import Aggregator
from krum.primitives.attacks import Attack
from krum.simulations.centralised import KrumSimulation


def run_one_simulation(
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
    aggregator_f: int | None = None,
) -> list[tuple[int, Any]]:
    """Build and run one KrumSimulation instance."""
    print(f"\n=== {label} ===")
    sim = KrumSimulation(
        model_cls=model_cls,
        train_set=train_set,
        test_set=test_set,
        aggregator=aggregator,
        aggregator_kwargs=aggregator_kwargs,
        attack=attack,
        attack_kwargs=attack_kwargs,
        n=n,
        f=f,
        aggregator_f=aggregator_f,
        rounds=rounds,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        eval_every=eval_every,
    )
    return sim.run()
