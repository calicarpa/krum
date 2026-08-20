"""Datasets and per-worker dataloaders for the MoNNA decentralised experiment."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from krum.primitives.data_partitioners import DataPartitioner


@lru_cache(maxsize=8)
def make_datasets(
    *,
    dataset: str,
    data_dir: str,
    train_size: int,
    test_size: int,
    n: int,
    train_batch_size: int,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Create the train and test datasets (MNIST/CIFAR-10 download, or synthetic FakeData).

    Memoized on its (hashable) configuration so repeated runs with the same data
    configuration reuse the loaded datasets instead of reloading them. The cache
    keeps the 8 most recent configurations; the returned datasets are read-only
    and shared across runs, so callers must not mutate them.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == "mnist":
        train = datasets.MNIST(Path(data_dir), train=True, download=True, transform=transform)
        test = datasets.MNIST(Path(data_dir), train=False, download=True, transform=transform)
    elif dataset == "cifar10":
        train = datasets.CIFAR10(Path(data_dir), train=True, download=True, transform=transform)
        test = datasets.CIFAR10(Path(data_dir), train=False, download=True, transform=transform)
    else:
        train = datasets.FakeData(
            size=max(train_size, n * train_batch_size),
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
            random_offset=seed,
        )
        test = datasets.FakeData(
            size=max(test_size, train_batch_size),
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


def make_worker_streams(
    dataset: Dataset,
    *,
    n: int,
    partitioner: type[DataPartitioner],
    partitioner_kwargs: dict[str, Any] | None = None,
    seed: int,
) -> list[Dataset]:
    """Split the training dataset into one dataset per worker (honest and Byzantine).

    ``partitioner`` is invoked directly (e.g.
    :class:`~krum.primitives.data_partitioners.iid.IidPartitioner` or
    :class:`~krum.primitives.data_partitioners.dirichlet.DirichletPartitioner`),
    with ``partitioner_kwargs`` forwarded as its strategy-specific keyword
    arguments (e.g. ``{"alpha": ...}`` for ``DirichletPartitioner``). The
    returned datasets are handed directly to
    :class:`~krum.simulations.decentralised.MonnaSimulation` as
    ``train_datasets``, which wraps each honest worker's dataset into its own
    ``DataLoader`` (batch size, shuffling) and re-iterates it automatically
    once its epoch is exhausted.
    """
    return partitioner.partition(dataset, n=n, seed=seed, **(partitioner_kwargs or {}))
