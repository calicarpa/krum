"""Datasets and per-worker dataloaders for the MoNNA decentralised experiment."""

from functools import lru_cache
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from krum.primitives.data_partitioners.dirichlet import DirichletPartitioner
from krum.primitives.data_partitioners.iid import IidPartitioner


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


def make_worker_streams(
    dataset: Dataset,
    *,
    num_honest: int,
    batch_size: int,
    partition: str,
    dirichlet_alpha: float,
    seed: int,
) -> list[DataLoader]:
    """Split the training dataset into one ``DataLoader`` per honest worker.

    ``"iid"`` uses :class:`~krum.primitives.data_partitioners.iid.IidPartitioner`;
    anything else uses
    :class:`~krum.primitives.data_partitioners.dirichlet.DirichletPartitioner`
    with the given ``dirichlet_alpha``. The returned loaders are handed
    directly to :class:`~krum.simulations.decentralised.MonnaSimulation`,
    which re-iterates each one automatically once its epoch is exhausted.
    """
    if partition == "iid":
        return IidPartitioner.partition(dataset, n=num_honest, batch_size=batch_size, seed=seed)
    return DirichletPartitioner.partition(
        dataset, n=num_honest, alpha=dirichlet_alpha, batch_size=batch_size, seed=seed
    )
