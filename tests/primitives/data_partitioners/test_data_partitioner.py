"""Tests for the base DataPartitioner class."""

import unittest
from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from krum.primitives.data_partitioners import DataPartitioner


class _ConcreteDataPartitioner(DataPartitioner):
    """Minimal concrete partitioner for testing the base class."""

    @classmethod
    def partition(
        cls,
        dataset: Dataset[Any],
        /,
        *,
        n: int,
        batch_size: int,
        seed: int = 42,
        **specialized: Any,
    ) -> Sequence[DataLoader[Any]]:
        return [DataLoader(dataset, batch_size=batch_size) for _ in range(n)]


class DataPartitionerTest(unittest.TestCase):
    """Test DataPartitioner base class."""

    def test_partition_returns_n_loaders(self) -> None:
        """Partition returns one loader per worker."""
        dataset = TensorDataset(torch.randn(10, 3), torch.randint(0, 2, (10,)))
        loaders = _ConcreteDataPartitioner.partition(dataset, n=4, batch_size=2)
        self.assertEqual(len(loaders), 4)


if __name__ == "__main__":
    unittest.main()
