"""Build, run, and evaluate one MoNNA simulation as an orchestrator experiment."""

import random
from typing import Any

import torch
from torch import nn

from krum.orchestration import Metric
from krum.primitives.aggregators import Aggregator
from krum.primitives.attacks import Attack
from krum.primitives.data_partitioners import DataPartitioner
from krum.primitives.models import Model
from krum.simulations.decentralised.monna_icml_2023 import MonnaSimulation

from ..datasets import make_datasets, make_worker_streams


def detect_device() -> torch.device:
    """Detect the best available torch device.

    Returns:
        CUDA if available, otherwise MPS, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def monna_experiment(
    *,
    label: str,
    dataset: str,
    data_dir: str,
    model_cls: type[nn.Module],
    n: int,
    f: int,
    learning_rate: float,
    beta: float,
    weight_decay: float = 0.0,
    attack: type[Attack] | None = None,
    attack_kwargs: dict[str, Any] | None = None,
    aggregator: type[Aggregator] | None = None,
    aggregator_kwargs: dict[str, Any] | None = None,
    rounds: int,
    eval_every: int,
    train_batch_size: int,
    test_batch_size: int,
    train_size: int,
    test_size: int,
    partitioner: type[DataPartitioner],
    partitioner_kwargs: dict[str, Any] | None = None,
    seed: int,
    byzantine_reach: str = "all",
    device: torch.device | None = None,
) -> None:
    """Run one MoNNA simulation and record its per-round metrics.

    Intended to be driven by :class:`~krum.orchestration.Orchestrator`. The
    datasets are built from configuration *inside* this function, so the run is
    identified by hashable parameters only (the dataset name and sizes, not the
    dataset objects). Honest workers run local momentum-SGD then mix their models
    by nearest-neighbor averaging (or ``aggregator``, if given, e.g. ``Average``
    for a non-robust baseline).

    On evaluated rounds (the first, every ``eval_every``, and the last) three
    metric channels are pushed, keyed by the round number as ``step``:
    ``train_loss`` (mean over honest workers), ``test_loss`` and
    ``test_accuracy`` (mean over honest worker models on the test set, via
    :meth:`~krum.simulations.decentralised.MonnaSimulation.evaluate`).

    The model and every batch are placed on ``device`` (CUDA, MPS, or CPU,
    auto-detected when ``device`` is left as ``None``).
    """
    print(f"\n=== {label} ===")
    resolved_device = device or detect_device()
    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    train_set, test_set = make_datasets(
        dataset=dataset,
        data_dir=data_dir,
        train_size=train_size,
        test_size=test_size,
        n=n,
        train_batch_size=train_batch_size,
        seed=seed,
    )
    worker_datasets = make_worker_streams(
        train_set,
        n=n,
        partitioner=partitioner,
        partitioner_kwargs=partitioner_kwargs,
        seed=seed,
    )

    model = Model(model_cls().to(resolved_device))
    loss_fn = nn.CrossEntropyLoss()

    simulation = MonnaSimulation(
        model=model,
        train_datasets=worker_datasets,
        train_batch_size=train_batch_size,
        test_set=test_set,
        test_batch_size=test_batch_size,
        loss_fn=loss_fn,
        n=n,
        f=f,
        learning_rate=learning_rate,
        beta=beta,
        weight_decay=weight_decay,
        attack=attack,
        attack_kwargs=attack_kwargs,
        aggregator=aggregator,
        aggregator_kwargs=aggregator_kwargs,
        byzantine_reach=byzantine_reach,
        seed=seed,
    )

    train_loss = Metric("train_loss", float)
    test_loss = Metric("test_loss", float)
    test_accuracy = Metric("test_accuracy", float)

    for step in range(1, rounds + 1):
        result = simulation.step()
        if step == 1 or step % eval_every == 0 or step == rounds:
            evaluated_loss, evaluated_accuracy = simulation.evaluate()
            train_loss.push(step, result["losses"].mean().item())
            test_loss.push(step, evaluated_loss)
            test_accuracy.push(step, evaluated_accuracy)
