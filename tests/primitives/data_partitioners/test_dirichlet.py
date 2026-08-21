"""Tests for the Dirichlet partitioner."""

import unittest

import torch
from torch.utils.data import Dataset, Subset, TensorDataset

from krum.primitives.data_partitioners.dirichlet import DirichletPartitioner


class _TargetsDataset(Dataset):
    """Minimal dataset exposing ``.targets``, like the torchvision datasets."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.data = x
        self.targets = y

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[i], self.targets[i]


def _balanced_dataset(samples_per_class: int = 50, num_classes: int = 4) -> _TargetsDataset:
    x = torch.arange(samples_per_class * num_classes, dtype=torch.float32).unsqueeze(1)
    y = torch.arange(num_classes).repeat_interleave(samples_per_class)
    return _TargetsDataset(x, y)


def _dataset_indices(dataset: Subset) -> list[int]:
    return [int(i) for i in dataset.indices]


class DirichletPartitionerTest(unittest.TestCase):
    """Test DirichletPartitioner."""

    def test_partition_returns_n_datasets(self) -> None:
        """Partition returns one dataset per worker."""
        dataset = _balanced_dataset()
        datasets = DirichletPartitioner.partition(dataset, n=5, alpha=0.5)
        self.assertEqual(len(datasets), 5)

    def test_partition_uses_every_sample_exactly_once(self) -> None:
        """Every sample is assigned to exactly one worker, unlike IidPartitioner."""
        dataset = _balanced_dataset(samples_per_class=50, num_classes=4)
        datasets = DirichletPartitioner.partition(dataset, n=5, alpha=0.5)
        all_indices = [i for ds in datasets for i in _dataset_indices(ds)]
        self.assertEqual(sorted(all_indices), list(range(200)))

    def test_partition_is_deterministic_given_seed(self) -> None:
        """Same seed produces the same shard assignment."""
        dataset = _balanced_dataset()
        datasets_a = DirichletPartitioner.partition(dataset, n=5, alpha=0.5, seed=7)
        datasets_b = DirichletPartitioner.partition(dataset, n=5, alpha=0.5, seed=7)
        for a, b in zip(datasets_a, datasets_b, strict=True):
            self.assertEqual(_dataset_indices(a), _dataset_indices(b))

    def test_partition_differs_across_seeds(self) -> None:
        """Different seeds produce different shard assignments."""
        dataset = _balanced_dataset()
        datasets_a = DirichletPartitioner.partition(dataset, n=5, alpha=0.5, seed=1)
        datasets_b = DirichletPartitioner.partition(dataset, n=5, alpha=0.5, seed=2)
        shards_a = [_dataset_indices(ds) for ds in datasets_a]
        shards_b = [_dataset_indices(ds) for ds in datasets_b]
        self.assertNotEqual(shards_a, shards_b)

    def test_partition_works_without_targets_attribute(self) -> None:
        """Falls back to per-sample indexing when dataset has no .targets attribute."""
        x = torch.randn(40, 3)
        y = torch.randint(0, 4, (40,))
        dataset = TensorDataset(x, y)
        datasets = DirichletPartitioner.partition(dataset, n=4, alpha=0.5)
        all_indices = [i for ds in datasets for i in _dataset_indices(ds)]
        self.assertEqual(sorted(all_indices), list(range(40)))

    def test_high_alpha_is_approximately_balanced_per_class(self) -> None:
        """Large alpha yields an approximately even split of each class across workers."""
        dataset = _balanced_dataset(samples_per_class=1000, num_classes=2)
        datasets = DirichletPartitioner.partition(dataset, n=4, alpha=1000.0, seed=1)
        for ds in datasets:
            self.assertAlmostEqual(len(ds), 500, delta=50)

    def test_low_alpha_is_highly_skewed(self) -> None:
        """Small alpha concentrates most of each class onto very few workers."""
        dataset = _balanced_dataset(samples_per_class=1000, num_classes=2)
        datasets = DirichletPartitioner.partition(dataset, n=4, alpha=0.01, seed=1)
        sizes = sorted((len(ds) for ds in datasets), reverse=True)
        self.assertGreater(sizes[0] + sizes[1], 1900)

    def test_handles_worker_with_empty_shard(self) -> None:
        """Extreme alpha can leave a worker with zero samples; it gets a valid, empty dataset."""
        dataset = _balanced_dataset(samples_per_class=10, num_classes=2)
        datasets = DirichletPartitioner.partition(dataset, n=10, alpha=1e-6, seed=3)
        empty_datasets = [ds for ds in datasets if len(ds) == 0]
        self.assertGreater(len(empty_datasets), 0)

    def test_handles_fully_empty_dataset(self) -> None:
        """An empty input dataset yields n empty datasets rather than crashing.

        torch.distributions.Dirichlet rejects a zero-sized batch dimension, so
        this is handled with an explicit early return before ever constructing
        one; regression test for that path.
        """
        dataset = Subset(_balanced_dataset(), [])
        datasets = DirichletPartitioner.partition(dataset, n=5, alpha=0.5)
        self.assertEqual(len(datasets), 5)
        for ds in datasets:
            self.assertEqual(len(ds), 0)

    def test_rejects_n_less_than_one(self) -> None:
        """Check raises ValueError when n < 1."""
        dataset = _balanced_dataset()
        with self.assertRaises(ValueError):
            DirichletPartitioner.partition(dataset, n=0, alpha=0.5)

    def test_rejects_non_positive_alpha(self) -> None:
        """Check raises ValueError when alpha <= 0."""
        dataset = _balanced_dataset()
        with self.assertRaises(ValueError):
            DirichletPartitioner.partition(dataset, n=5, alpha=0.0)
        with self.assertRaises(ValueError):
            DirichletPartitioner.partition(dataset, n=5, alpha=-1.0)


if __name__ == "__main__":
    unittest.main()
