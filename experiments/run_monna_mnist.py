"""Run a small MoNNA experiment on MNIST or FakeData.

This script is intentionally small: it demonstrates the library simulation path
without trying to reproduce every figure from the paper.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from krum.primitives import Model
from krum.primitives.attacks.sign_flip import SignFlipAttack
from krum.simulations.monna import MonnaSimulation

Batch = tuple[torch.Tensor, torch.Tensor]


class SmallMnistNet(nn.Module):
    """Small CPU-friendly classifier for 28x28 images."""

    def __init__(self) -> None:
        """Initialize the network."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute logits."""
        return self.net(inputs)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mnist", "fake"), default="mnist")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--num-honest", type=int, default=8)
    parser.add_argument("--num-byzantine", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.99)
    parser.add_argument("--attack", choices=("none", "sign-flip"), default="sign-flip")
    parser.add_argument("--partition", choices=("iid", "dirichlet"), default="dirichlet")
    parser.add_argument("--dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--sign-flip-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes per honest worker")
    return parser.parse_args()


def make_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    """Create train and test datasets."""
    transform = transforms.Compose([transforms.ToTensor()])
    if args.dataset == "mnist":
        train = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
        test = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    else:
        train = datasets.FakeData(
            size=max(args.train_size, args.num_honest * args.batch_size),
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
            random_offset=args.seed,
        )
        test = datasets.FakeData(
            size=max(args.test_size, args.batch_size),
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
            random_offset=args.seed + 10_000,
        )
    return limit_dataset(train, args.train_size), limit_dataset(test, args.test_size)


def limit_dataset(dataset: Dataset, size: int) -> Dataset:
    """Limit a dataset to its first ``size`` samples."""
    if size <= 0 or size >= len(dataset):
        return dataset
    return Subset(dataset, list(range(size)))


def split_iid(dataset: Dataset, *, num_parts: int, seed: int) -> list[Subset]:
    """Create deterministic IID worker shards."""
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    shards = [indices[i::num_parts] for i in range(num_parts)]
    return [Subset(dataset, shard) for shard in shards]


def dataset_labels(dataset: Dataset) -> list[int]:
    """Read integer labels from a dataset or subset."""
    labels = []
    for index in range(len(dataset)):
        _, target = dataset[index]
        labels.append(int(target))
    return labels


def split_dirichlet(dataset: Dataset, *, num_parts: int, alpha: float, seed: int) -> list[Subset]:
    """Create deterministic non-IID worker shards using class-wise Dirichlet sampling."""
    if alpha <= 0:
        raise ValueError(f"Expected positive Dirichlet alpha, got {alpha!r}")

    rng = np.random.default_rng(seed)
    labels = dataset_labels(dataset)
    classes = sorted(set(labels))
    shards: list[list[int]] = [[] for _ in range(num_parts)]

    for label in classes:
        class_indices = [index for index, target in enumerate(labels) if target == label]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(num_parts, alpha))
        cut_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
        for worker_id, split in enumerate(np.split(np.array(class_indices), cut_points)):
            shards[worker_id].extend(int(index) for index in split.tolist())

    for shard in shards:
        rng.shuffle(shard)
    return [Subset(dataset, shard) for shard in shards]


def split_dataset(args: argparse.Namespace, dataset: Dataset) -> list[Subset]:
    """Split the training dataset across honest workers."""
    if args.partition == "iid":
        return split_iid(dataset, num_parts=args.num_honest, seed=args.seed)
    return split_dirichlet(dataset, num_parts=args.num_honest, alpha=args.dirichlet_alpha, seed=args.seed)


def cycle_loader(loader: DataLoader) -> Iterator[Batch]:
    """Yield batches forever from a finite DataLoader."""
    while True:
        yield from loader


def make_worker_streams(args: argparse.Namespace, dataset: Dataset) -> list[Iterator[Batch]]:
    """Create one infinite batch stream per honest worker."""
    shards = split_dataset(args, dataset)
    streams = []
    for worker_id, shard in enumerate(shards):
        generator = torch.Generator().manual_seed(args.seed + worker_id)
        if len(shard) < args.batch_size:
            raise ValueError(
                f"Worker {worker_id} shard has {len(shard)} samples, fewer than batch size {args.batch_size}. "
                "Increase --train-size, decrease --batch-size, or use --partition iid."
            )
        loader = DataLoader(
            shard,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            generator=generator,
        )
        streams.append(cycle_loader(loader))
    return streams


def copy_parameters(model: Model, parameters: torch.Tensor) -> None:
    """Load one flat parameter vector into the shared model."""
    with torch.no_grad():
        model.parameters.copy_(parameters)


@torch.no_grad()
def evaluate_parameters(
    model: Model, parameters: torch.Tensor, loader: DataLoader, loss_fn: nn.Module
) -> tuple[float, float]:
    """Evaluate one worker parameter vector."""
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


def evaluate_workers(model: Model, parameters: torch.Tensor, loader: DataLoader, loss_fn: nn.Module) -> tuple[float, float]:
    """Evaluate the mean loss and accuracy over honest worker parameter vectors."""
    losses = []
    accuracies = []
    for worker_parameters in parameters:
        loss, accuracy = evaluate_parameters(model, worker_parameters, loader, loss_fn)
        losses.append(loss)
        accuracies.append(accuracy)
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)


def main() -> None:
    """Run the experiment."""
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_dataset, test_dataset = make_datasets(args)
    worker_streams = make_worker_streams(args, train_dataset)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=args.num_workers)

    module = SmallMnistNet()
    model = Model(module)
    loss_fn = nn.CrossEntropyLoss()
    attack = None if args.attack == "none" else partial(SignFlipAttack.generate, scale=args.sign_flip_scale)

    simulation = MonnaSimulation(
        model=model,
        data=worker_streams,
        loss_fn=loss_fn,
        num_honest=args.num_honest,
        num_byzantine=args.num_byzantine,
        learning_rate=args.learning_rate,
        beta=args.beta,
        attack=attack,
        seed=args.seed,
    )

    print(
        "round,train_loss_mean,test_loss_mean,test_accuracy_mean",
        flush=True,
    )

    for step in range(1, args.rounds + 1):
        result = simulation.step()

        if step == 1 or step % args.eval_every == 0 or step == args.rounds:
            test_loss, test_accuracy = evaluate_workers(model, simulation.parameters, test_loader, loss_fn)
            train_loss = result["losses"].mean().item()
            print(f"{step},{train_loss:.6f},{test_loss:.6f},{test_accuracy:.4f}", flush=True)


if __name__ == "__main__":
    main()
