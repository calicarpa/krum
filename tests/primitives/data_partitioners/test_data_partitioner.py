"""Tests for the base DataPartitioner class."""

import unittest
from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import Dataset, Subset, TensorDataset

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
        seed: int = 42,
        **specialized: Any,
    ) -> Sequence[Dataset[Any]]:
        return [Subset(dataset, []) for _ in range(n)]


class DataPartitionerTest(unittest.TestCase):
    """Test DataPartitioner base class."""

    def test_partition_returns_n_datasets(self) -> None:
        """Partition returns one dataset per worker."""
        dataset = TensorDataset(torch.randn(10, 3), torch.randint(0, 2, (10,)))
        datasets = _ConcreteDataPartitioner.partition(dataset, n=4)
        self.assertEqual(len(datasets), 4)


if __name__ == "__main__":
    unittest.main()
