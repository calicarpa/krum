"""Standard dataset loaders shared across simulation protocols."""

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
