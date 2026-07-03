"""Build, run, and evaluate one MoNNA simulation as an orchestrator experiment."""

import random
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from krum.orchestration import Metric
from krum.primitives import Model
from krum.primitives.attacks import Attack
from krum.simulations.decentralised.monna_icml_2023 import MonnaSimulation

from ..datasets import make_datasets, make_worker_streams


def copy_parameters(model: Model, parameters: torch.Tensor) -> None:
    """Load one flat parameter vector into the shared model."""
    with torch.no_grad():
        model.parameters.copy_(parameters)


@torch.no_grad()
def evaluate_parameters(
    model: Model, parameters: torch.Tensor, loader: DataLoader, loss_fn: nn.Module
) -> tuple[float, float]:
    """Evaluate one worker parameter vector; return ``(loss, accuracy)``."""
    copy_parameters(model, parameters)
    model.module.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for inputs, targets in loader:
        logits = model.module(inputs)
        loss = loss_fn(logits, targets)
        total_loss += loss.item() * targets.numel()
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.numel()
    return total_loss / total, total_correct / total


def evaluate_workers(
    model: Model, parameters: torch.Tensor, loader: DataLoader, loss_fn: nn.Module
) -> tuple[float, float]:
    """Evaluate the mean loss and accuracy over honest worker parameter vectors."""
    losses = []
    accuracies = []
    for worker_parameters in parameters:
        loss, accuracy = evaluate_parameters(model, worker_parameters, loader, loss_fn)
        losses.append(loss)
        accuracies.append(accuracy)
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)


def monna_experiment(
    *,
    dataset: str,
    data_dir: str,
    model_cls: type[nn.Module],
    n: int,
    f: int,
    learning_rate: float,
    beta: float,
    attack: type[Attack] | None = None,
    attack_kwargs: dict[str, Any] | None = None,
    rounds: int,
    eval_every: int,
    batch_size: int,
    train_size: int,
    test_size: int,
    partition: str,
    dirichlet_alpha: float,
    num_workers: int,
    seed: int,
    byzantine_reach: str = "all",
) -> None:
    """Run one MoNNA simulation and record its per-round metrics.

    Intended to be driven by :class:`~krum.orchestration.Orchestrator`. The
    datasets are built from configuration *inside* this function, so the run is
    identified by hashable parameters only (the dataset name and sizes, not the
    dataset objects). Honest workers run local momentum-SGD then mix their models
    by nearest-neighbor averaging.

    On evaluated rounds (the first, every ``eval_every``, and the last) three
    metric channels are pushed, keyed by the round number as ``step``:
    ``train_loss`` (mean over honest workers), ``test_loss`` and
    ``test_accuracy`` (mean over honest worker models on the test set).
    """
    torch.manual_seed(seed)
    random.seed(seed)

    train_set, test_set = make_datasets(
        dataset=dataset,
        data_dir=data_dir,
        train_size=train_size,
        test_size=test_size,
        num_honest=n - f,
        batch_size=batch_size,
        seed=seed,
    )
    worker_streams = make_worker_streams(
        train_set,
        num_honest=n - f,
        batch_size=batch_size,
        partition=partition,
        dirichlet_alpha=dirichlet_alpha,
        seed=seed,
        num_workers=num_workers,
    )
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=num_workers)

    model = Model(model_cls())
    loss_fn = nn.CrossEntropyLoss()

    simulation = MonnaSimulation(
        model=model,
        data=worker_streams,
        loss_fn=loss_fn,
        n=n,
        f=f,
        learning_rate=learning_rate,
        beta=beta,
        attack=attack,
        attack_kwargs=attack_kwargs,
        byzantine_reach=byzantine_reach,
        seed=seed,
    )

    train_loss = Metric("train_loss", float)
    test_loss = Metric("test_loss", float)
    test_accuracy = Metric("test_accuracy", float)

    for step in range(1, rounds + 1):
        result = simulation.step()
        if step == 1 or step % eval_every == 0 or step == rounds:
            evaluated_loss, evaluated_accuracy = evaluate_workers(model, simulation.parameters, test_loader, loss_fn)
            train_loss.push(step, result["losses"].mean().item())
            test_loss.push(step, evaluated_loss)
            test_accuracy.push(step, evaluated_accuracy)
