"""Build, run, and evaluate one MoNNA simulation."""

import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from krum.primitives import Model
from krum.primitives.attacks import Attack
from krum.simulations.decentralised import MonnaSimulation

from ..datasets import make_worker_streams


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


def run_monna_simulation(
    *,
    model_cls: type[nn.Module],
    train_set: Dataset,
    test_set: Dataset,
    n: int,
    f: int,
    learning_rate: float,
    beta: float,
    attack: type[Attack] | None,
    attack_kwargs: dict[str, float] | None,
    rounds: int,
    eval_every: int,
    batch_size: int,
    partition: str,
    dirichlet_alpha: float,
    num_workers: int,
    seed: int,
    byzantine_reach: str = "all",
) -> list[dict[str, float]]:
    """Build one MoNNA simulation, train it, and return per-round metrics.

    Honest workers run local momentum-SGD then mix their models by
    nearest-neighbor averaging; metrics are the mean over the honest workers.

    Returns:
        One dict per evaluated round with keys ``round``, ``train_loss``,
        ``test_loss``, and ``test_accuracy``. The caller decides how to report
        them (print, plot, assert on in a test, sweep, ...).
    """
    torch.manual_seed(seed)
    random.seed(seed)

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

    metrics: list[dict[str, float]] = []
    for step in range(1, rounds + 1):
        result = simulation.step()
        if step == 1 or step % eval_every == 0 or step == rounds:
            test_loss, test_accuracy = evaluate_workers(model, simulation.parameters, test_loader, loss_fn)
            train_loss = result["losses"].mean().item()
            metrics.append({
                "round": step,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            })
    return metrics
