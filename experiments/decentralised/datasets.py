"""Datasets and per-worker batch streams for the MoNNA decentralised experiment."""

import random
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

Batch = tuple[torch.Tensor, torch.Tensor]


@lru_cache(maxsize=8)
def make_datasets(
    *,
    dataset: str,
    data_dir: str,
    train_size: int,
    test_size: int,
    num_honest: int,
    batch_size: int,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Create the train and test datasets (MNIST download or synthetic FakeData).

    Memoized on its (hashable) configuration so repeated runs with the same data
    configuration reuse the loaded datasets instead of reloading them. The cache
    keeps the 8 most recent configurations; the returned datasets are read-only
    and shared across runs, so callers must not mutate them.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == "mnist":
        train = datasets.MNIST(Path(data_dir), train=True, download=True, transform=transform)
        test = datasets.MNIST(Path(data_dir), train=False, download=True, transform=transform)
    else:
        train = datasets.FakeData(
            size=max(train_size, num_honest * batch_size),
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
            random_offset=seed,
        )
        test = datasets.FakeData(
            size=max(test_size, batch_size),
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
            random_offset=seed + 10_000,
        )
    return limit_dataset(train, train_size), limit_dataset(test, test_size)


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


def split_dataset(
    dataset: Dataset, *, partition: str, num_parts: int, dirichlet_alpha: float, seed: int
) -> list[Subset]:
    """Split the training dataset across honest workers."""
    if partition == "iid":
        return split_iid(dataset, num_parts=num_parts, seed=seed)
    return split_dirichlet(dataset, num_parts=num_parts, alpha=dirichlet_alpha, seed=seed)


def cycle_loader(loader: DataLoader) -> Iterator[Batch]:
    """Yield batches forever from a finite DataLoader."""
    while True:
        yield from loader


def make_worker_streams(
    dataset: Dataset,
    *,
    num_honest: int,
    batch_size: int,
    partition: str,
    dirichlet_alpha: float,
    seed: int,
    num_workers: int,
) -> list[Iterator[Batch]]:
    """Create one infinite batch stream per honest worker."""
    shards = split_dataset(
        dataset, partition=partition, num_parts=num_honest, dirichlet_alpha=dirichlet_alpha, seed=seed
    )
    streams = []
    for worker_id, shard in enumerate(shards):
        generator = torch.Generator().manual_seed(seed + worker_id)
        if len(shard) < batch_size:
            raise ValueError(
                f"Worker {worker_id} shard has {len(shard)} samples, fewer than batch size {batch_size}. "
                "Increase TRAIN_SIZE, decrease BATCH_SIZE, or set PARTITION = 'iid'."
            )
        loader = DataLoader(
            shard,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            generator=generator,
        )
        streams.append(cycle_loader(loader))
    return streams
