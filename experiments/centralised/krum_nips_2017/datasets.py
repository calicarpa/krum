"""Datasets for the Krum-NIPS-2017 simulation."""

import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from ..datasets import mnist_dataset as mnist_dataset


def spambase_dataset(*, test_size: float = 0.2, seed: int = 42) -> tuple[TensorDataset, TensorDataset]:
    """Download and return the Spambase dataset from the UCI repository.

    Spambase has 57 continuous features and a binary label (0 = not spam, 1 = spam).
    Features are standardized to zero mean and unit variance.

    Args:
        test_size: Fraction of data to use for the test split.
        seed: Random seed for the train/test split.

    Returns:
        Tuple of (train_dataset, test_dataset) as ``TensorDataset`` objects.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    cache_dir = Path("data/spambase")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "spambase.data"

    if not cache_path.exists():
        urllib.request.urlretrieve(url, cache_path)

    data = np.loadtxt(cache_path, delimiter=",")
    x = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.int64)

    x = (x - x.mean(axis=0)) / x.std(axis=0)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    split = int(len(x) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]

    train_ds = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]))
    test_ds = TensorDataset(torch.from_numpy(x[test_idx]), torch.from_numpy(y[test_idx]))
    return (train_ds, test_ds)
