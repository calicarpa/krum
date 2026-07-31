"""Tests for the IID partitioner."""

import unittest

import torch
from torch.utils.data import TensorDataset

from krum.primitives.data_partitioners.iid import IidPartitioner


def _dummy_dataset(size: int = 100) -> TensorDataset:
    x = torch.arange(size, dtype=torch.float32).unsqueeze(1)
    y = torch.zeros(size, dtype=torch.int64)
    return TensorDataset(x, y)


def _loader_indices(loader: torch.utils.data.DataLoader) -> set[int]:
    subset = loader.dataset
    assert isinstance(subset, torch.utils.data.Subset)
    return {int(i) for i in subset.indices}


class IidPartitionerTest(unittest.TestCase):
    """Test IidPartitioner."""

    def test_partition_returns_n_loaders(self) -> None:
        """Partition returns one dataloader per worker."""
        dataset = _dummy_dataset(100)
        loaders = IidPartitioner.partition(dataset, n=10, batch_size=5)
        self.assertEqual(len(loaders), 10)

    def test_partition_shards_are_equal_size(self) -> None:
        """Each worker gets an equal-size shard."""
        dataset = _dummy_dataset(100)
        loaders = IidPartitioner.partition(dataset, n=10, batch_size=5)
        for loader in loaders:
            self.assertEqual(len(_loader_indices(loader)), 10)

    def test_partition_drops_remainder(self) -> None:
        """Samples that don't evenly divide across workers are dropped."""
        dataset = _dummy_dataset(103)
        loaders = IidPartitioner.partition(dataset, n=10, batch_size=5)
        covered = set().union(*(_loader_indices(loader) for loader in loaders))
        self.assertEqual(len(covered), 100)

    def test_partition_shards_are_disjoint(self) -> None:
        """Worker shards do not overlap."""
        dataset = _dummy_dataset(100)
        loaders = IidPartitioner.partition(dataset, n=10, batch_size=5)
        all_indices = [i for loader in loaders for i in _loader_indices(loader)]
        self.assertEqual(len(all_indices), len(set(all_indices)))

    def test_partition_covers_dataset_ignoring_remainder(self) -> None:
        """Shards jointly cover the whole (divisible) dataset."""
        dataset = _dummy_dataset(100)
        loaders = IidPartitioner.partition(dataset, n=10, batch_size=5)
        covered = set().union(*(_loader_indices(loader) for loader in loaders))
        self.assertEqual(covered, set(range(100)))

    def test_partition_is_deterministic_given_seed(self) -> None:
        """Same seed produces the same shard assignment."""
        dataset = _dummy_dataset(100)
        loaders_a = IidPartitioner.partition(dataset, n=10, batch_size=5, seed=7)
        loaders_b = IidPartitioner.partition(dataset, n=10, batch_size=5, seed=7)
        for a, b in zip(loaders_a, loaders_b, strict=True):
            self.assertEqual(_loader_indices(a), _loader_indices(b))

    def test_partition_differs_across_seeds(self) -> None:
        """Different seeds produce different shard assignments."""
        dataset = _dummy_dataset(100)
        loaders_a = IidPartitioner.partition(dataset, n=10, batch_size=5, seed=1)
        loaders_b = IidPartitioner.partition(dataset, n=10, batch_size=5, seed=2)
        shards_a = [_loader_indices(loader) for loader in loaders_a]
        shards_b = [_loader_indices(loader) for loader in loaders_b]
        self.assertNotEqual(shards_a, shards_b)

    def test_partition_respects_batch_size(self) -> None:
        """Each dataloader batches with the requested batch_size."""
        dataset = _dummy_dataset(100)
        loaders = IidPartitioner.partition(dataset, n=10, batch_size=4)
        batch, _ = next(iter(loaders[0]))
        self.assertEqual(batch.shape[0], 4)

    def test_rejects_n_less_than_one(self) -> None:
        """Check raises ValueError when n < 1."""
        dataset = _dummy_dataset(100)
        with self.assertRaises(ValueError):
            IidPartitioner.partition(dataset, n=0, batch_size=5)


if __name__ == "__main__":
    unittest.main()
