"""Standard dataset loaders shared across simulation protocols."""

from collections.abc import Sequence
from typing import Any

from torch.utils.data import Dataset, Subset
from torchvision import datasets
from torchvision import transforms as T

from krum.primitives.data_partitioners import DataPartitioner


def mnist_dataset() -> tuple[datasets.MNIST, datasets.MNIST]:
    """Download and return the MNIST dataset.

    Returns:
        Tuple of (train, test) datasets with standard normalization.
    """
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST("data", train=True, download=True, transform=transform)
    test = datasets.MNIST("data", train=False, download=True, transform=transform)
    return train, test


def limit_dataset(dataset: Dataset, size: int) -> Dataset:
    """Limit a dataset to its first ``size`` samples (``size <= 0`` keeps all)."""
    if size <= 0 or size >= len(dataset):  # type: ignore[arg-type]
        return dataset
    return Subset(dataset, list(range(size)))


def make_worker_streams(
    dataset: Dataset,
    *,
    n: int,
    partitioner: type[DataPartitioner],
    partitioner_kwargs: dict[str, Any] | None = None,
    seed: int,
) -> Sequence[Dataset]:
    """Split the training dataset into one dataset per worker (honest and Byzantine).

    ``partitioner`` is invoked directly (e.g.
    :class:`~krum.primitives.data_partitioners.iid.IidPartitioner` or
    :class:`~krum.primitives.data_partitioners.dirichlet.DirichletPartitioner`),
    with ``partitioner_kwargs`` forwarded as its strategy-specific keyword
    arguments (e.g. ``{"alpha": ...}`` for ``DirichletPartitioner``). The
    returned datasets are handed directly to
    :class:`~krum.simulations.centralised.CentralisedSimulation` as
    ``train_datasets``, which wraps each honest worker's dataset into its own
    ``DataLoader`` (batch size, shuffling).
    """
    return partitioner.partition(dataset, n=n, seed=seed, **(partitioner_kwargs or {}))
