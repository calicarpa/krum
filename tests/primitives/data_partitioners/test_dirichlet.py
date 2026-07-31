"""Tests for the Dirichlet partitioner."""

import unittest

import torch
from torch.utils.data import Dataset, TensorDataset

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


def _loader_indices(loader: torch.utils.data.DataLoader) -> list[int]:
    subset = loader.dataset
    assert isinstance(subset, torch.utils.data.Subset)
    return [int(i) for i in subset.indices]


class DirichletPartitionerTest(unittest.TestCase):
    """Test DirichletPartitioner."""

    def test_partition_returns_n_loaders(self) -> None:
        """Partition returns one dataloader per worker."""
        dataset = _balanced_dataset()
        loaders = DirichletPartitioner.partition(dataset, n=5, alpha=0.5, batch_size=4)
        self.assertEqual(len(loaders), 5)

    def test_partition_uses_every_sample_exactly_once(self) -> None:
        """Every sample is assigned to exactly one worker, unlike IidPartitioner."""
        dataset = _balanced_dataset(samples_per_class=50, num_classes=4)
        loaders = DirichletPartitioner.partition(dataset, n=5, alpha=0.5, batch_size=4)
        all_indices = [i for loader in loaders for i in _loader_indices(loader)]
        self.assertEqual(sorted(all_indices), list(range(200)))

    def test_partition_is_deterministic_given_seed(self) -> None:
        """Same seed produces the same shard assignment."""
        dataset = _balanced_dataset()
        loaders_a = DirichletPartitioner.partition(dataset, n=5, alpha=0.5, batch_size=4, seed=7)
        loaders_b = DirichletPartitioner.partition(dataset, n=5, alpha=0.5, batch_size=4, seed=7)
        for a, b in zip(loaders_a, loaders_b, strict=True):
            self.assertEqual(_loader_indices(a), _loader_indices(b))

    def test_partition_differs_across_seeds(self) -> None:
        """Different seeds produce different shard assignments."""
        dataset = _balanced_dataset()
        loaders_a = DirichletPartitioner.partition(dataset, n=5, alpha=0.5, batch_size=4, seed=1)
        loaders_b = DirichletPartitioner.partition(dataset, n=5, alpha=0.5, batch_size=4, seed=2)
        shards_a = [_loader_indices(loader) for loader in loaders_a]
        shards_b = [_loader_indices(loader) for loader in loaders_b]
        self.assertNotEqual(shards_a, shards_b)

    def test_partition_respects_batch_size(self) -> None:
        """Each dataloader batches with the requested batch_size."""
        dataset = _balanced_dataset()
        loaders = DirichletPartitioner.partition(dataset, n=5, alpha=0.5, batch_size=4)
        for loader in loaders:
            if len(loader.dataset) >= 4:
                batch, _ = next(iter(loader))
                self.assertEqual(batch.shape[0], 4)
                break
        else:
            self.fail("no worker had at least one full batch")

    def test_partition_works_without_targets_attribute(self) -> None:
        """Falls back to per-sample indexing when dataset has no .targets attribute."""
        x = torch.randn(40, 3)
        y = torch.randint(0, 4, (40,))
        dataset = TensorDataset(x, y)
        loaders = DirichletPartitioner.partition(dataset, n=4, alpha=0.5, batch_size=4)
        all_indices = [i for loader in loaders for i in _loader_indices(loader)]
        self.assertEqual(sorted(all_indices), list(range(40)))

    def test_high_alpha_is_approximately_balanced_per_class(self) -> None:
        """Large alpha yields an approximately even split of each class across workers."""
        dataset = _balanced_dataset(samples_per_class=1000, num_classes=2)
        loaders = DirichletPartitioner.partition(dataset, n=4, alpha=1000.0, batch_size=4, seed=1)
        for loader in loaders:
            self.assertAlmostEqual(len(loader.dataset), 500, delta=50)

    def test_low_alpha_is_highly_skewed(self) -> None:
        """Small alpha concentrates most of each class onto very few workers."""
        dataset = _balanced_dataset(samples_per_class=1000, num_classes=2)
        loaders = DirichletPartitioner.partition(dataset, n=4, alpha=0.01, batch_size=4, seed=1)
        sizes = sorted((len(loader.dataset) for loader in loaders), reverse=True)
        self.assertGreater(sizes[0] + sizes[1], 1900)

    def test_handles_worker_with_empty_shard(self) -> None:
        """Extreme alpha can leave a worker with zero samples; it gets an empty, iterable loader."""
        dataset = _balanced_dataset(samples_per_class=10, num_classes=2)
        loaders = DirichletPartitioner.partition(dataset, n=10, alpha=1e-6, batch_size=4, seed=3)
        empty_loaders = [loader for loader in loaders if len(loader.dataset) == 0]
        self.assertGreater(len(empty_loaders), 0)
        self.assertEqual(list(empty_loaders[0]), [])

    def test_rejects_n_less_than_one(self) -> None:
        """Check raises ValueError when n < 1."""
        dataset = _balanced_dataset()
        with self.assertRaises(ValueError):
            DirichletPartitioner.partition(dataset, n=0, alpha=0.5, batch_size=4)

    def test_rejects_non_positive_alpha(self) -> None:
        """Check raises ValueError when alpha <= 0."""
        dataset = _balanced_dataset()
        with self.assertRaises(ValueError):
            DirichletPartitioner.partition(dataset, n=5, alpha=0.0, batch_size=4)
        with self.assertRaises(ValueError):
            DirichletPartitioner.partition(dataset, n=5, alpha=-1.0, batch_size=4)


if __name__ == "__main__":
    unittest.main()
