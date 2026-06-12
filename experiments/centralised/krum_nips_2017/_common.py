"""Shared utilities for the NIPS 2017 experiments."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset

from krum.primitives.aggregators import Aggregator
from krum.simulations.centralised import KrumSimulation


def run_one_simulation(
    *,
    label: str,
    model_cls: type[nn.Module],
    train_set: TorchDataset[Any],
    test_set: TorchDataset[Any],
    aggregator: type[Aggregator],
    attack: Any,
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
    """Build and run one :class:`KrumSimulation` instance.

    Returns:
        List of ``(round, ...)`` tuples containing evaluation results.
    """
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
