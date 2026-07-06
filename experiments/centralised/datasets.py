"""Standard dataset loaders shared across simulation protocols."""

from torch.utils.data import Dataset, Subset
from torchvision import datasets
from torchvision import transforms as T


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
