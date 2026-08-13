"""Tests for the IID partitioner."""

import unittest

import torch
from torch.utils.data import Subset, TensorDataset

from krum.primitives.data_partitioners.iid import IidPartitioner


def _dummy_dataset(size: int = 100) -> TensorDataset:
    x = torch.arange(size, dtype=torch.float32).unsqueeze(1)
    y = torch.zeros(size, dtype=torch.int64)
    return TensorDataset(x, y)


def _dataset_indices(dataset: Subset) -> set[int]:
    return {int(i) for i in dataset.indices}


class IidPartitionerTest(unittest.TestCase):
    """Test IidPartitioner."""

    def test_partition_returns_n_datasets(self) -> None:
        """Partition returns one dataset per worker."""
        dataset = _dummy_dataset(100)
        datasets = IidPartitioner.partition(dataset, n=10)
        self.assertEqual(len(datasets), 10)

    def test_partition_shards_are_equal_size(self) -> None:
        """Each worker gets an equal-size shard."""
        dataset = _dummy_dataset(100)
        datasets = IidPartitioner.partition(dataset, n=10)
        for ds in datasets:
            self.assertEqual(len(_dataset_indices(ds)), 10)

    def test_partition_drops_remainder(self) -> None:
        """Samples that don't evenly divide across workers are dropped."""
        dataset = _dummy_dataset(103)
        datasets = IidPartitioner.partition(dataset, n=10)
        covered = set().union(*(_dataset_indices(ds) for ds in datasets))
        self.assertEqual(len(covered), 100)

    def test_partition_shards_are_disjoint(self) -> None:
        """Worker shards do not overlap."""
        dataset = _dummy_dataset(100)
        datasets = IidPartitioner.partition(dataset, n=10)
        all_indices = [i for ds in datasets for i in _dataset_indices(ds)]
        self.assertEqual(len(all_indices), len(set(all_indices)))

    def test_partition_covers_dataset_ignoring_remainder(self) -> None:
        """Shards jointly cover the whole (divisible) dataset."""
        dataset = _dummy_dataset(100)
        datasets = IidPartitioner.partition(dataset, n=10)
        covered = set().union(*(_dataset_indices(ds) for ds in datasets))
        self.assertEqual(covered, set(range(100)))

    def test_partition_is_deterministic_given_seed(self) -> None:
        """Same seed produces the same shard assignment."""
        dataset = _dummy_dataset(100)
        datasets_a = IidPartitioner.partition(dataset, n=10, seed=7)
        datasets_b = IidPartitioner.partition(dataset, n=10, seed=7)
        for a, b in zip(datasets_a, datasets_b, strict=True):
            self.assertEqual(_dataset_indices(a), _dataset_indices(b))

    def test_partition_differs_across_seeds(self) -> None:
        """Different seeds produce different shard assignments."""
        dataset = _dummy_dataset(100)
        datasets_a = IidPartitioner.partition(dataset, n=10, seed=1)
        datasets_b = IidPartitioner.partition(dataset, n=10, seed=2)
        shards_a = [_dataset_indices(ds) for ds in datasets_a]
        shards_b = [_dataset_indices(ds) for ds in datasets_b]
        self.assertNotEqual(shards_a, shards_b)

    def test_rejects_n_less_than_one(self) -> None:
        """Check raises ValueError when n < 1."""
        dataset = _dummy_dataset(100)
        with self.assertRaises(ValueError):
            IidPartitioner.partition(dataset, n=0)


if __name__ == "__main__":
    unittest.main()
