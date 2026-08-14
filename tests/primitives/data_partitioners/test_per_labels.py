"""Tests for the per-labels partitioner."""

import unittest
from collections import Counter

import torch
from torch.utils.data import Dataset, Subset, TensorDataset

from krum.primitives.data_partitioners.per_labels import PerLabelsPartitioner


class _TargetsDataset(Dataset):
    """Minimal dataset exposing ``.targets``, like the torchvision datasets."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.data = x
        self.targets = y

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[i], self.targets[i]


def _balanced_dataset(samples_per_class: int = 6000, num_classes: int = 10) -> _TargetsDataset:
    x = torch.arange(samples_per_class * num_classes, dtype=torch.float32).unsqueeze(1)
    y = torch.arange(num_classes).repeat_interleave(samples_per_class)
    return _TargetsDataset(x, y)


def _dataset_indices(dataset: Subset) -> list[int]:
    return [int(i) for i in dataset.indices]


def _top_class_fraction(dataset: _TargetsDataset, worker_dataset: Subset) -> float:
    labels = [int(dataset.targets[i].item()) for i in worker_dataset.indices]
    if not labels:
        return 0.0
    return Counter(labels).most_common(1)[0][1] / len(labels)


class PerLabelsPartitionerTest(unittest.TestCase):
    """Test PerLabelsPartitioner."""

    def test_partition_returns_n_datasets(self) -> None:
        """Partition returns one dataset per worker."""
        dataset = _balanced_dataset()
        datasets = PerLabelsPartitioner.partition(dataset, n=10, lambda_=0.5)
        self.assertEqual(len(datasets), 10)

    def test_lambda_zero_is_maximally_skewed(self) -> None:
        """lambda_=0 gives n_shards=n; each worker's shard is (almost) a single class."""
        dataset = _balanced_dataset()
        datasets = PerLabelsPartitioner.partition(dataset, n=10, lambda_=0.0, seed=1)
        for ds in datasets:
            self.assertGreater(_top_class_fraction(dataset, ds), 0.99)

    def test_lambda_one_recovers_iid_like_distribution(self) -> None:
        """lambda_=1 gives n_shards=N; every worker's class mix is close to uniform."""
        dataset = _balanced_dataset()
        datasets = PerLabelsPartitioner.partition(dataset, n=10, lambda_=1.0, seed=1)
        for ds in datasets:
            # 10 balanced classes: a uniform mix has each class at ~10% share.
            self.assertAlmostEqual(_top_class_fraction(dataset, ds), 0.1, delta=0.03)

    def test_round_robin_balances_shard_remainder(self) -> None:
        """When n_shards doesn't divide n evenly, worker sizes differ by at most one shard."""
        dataset = _balanced_dataset()
        # lambda_=0.01 -> n_shards=11 for n=10, N=60000 (not a multiple of n).
        datasets = PerLabelsPartitioner.partition(dataset, n=10, lambda_=0.01, seed=1)
        sizes = sorted(len(ds) for ds in datasets)
        self.assertLessEqual(sizes[-1] - sizes[0], sizes[0])  # at most one shard's worth apart

    def test_partition_is_deterministic_given_seed(self) -> None:
        """Same seed produces the same shard assignment."""
        dataset = _balanced_dataset()
        datasets_a = PerLabelsPartitioner.partition(dataset, n=10, lambda_=0.3, seed=7)
        datasets_b = PerLabelsPartitioner.partition(dataset, n=10, lambda_=0.3, seed=7)
        for a, b in zip(datasets_a, datasets_b, strict=True):
            self.assertEqual(_dataset_indices(a), _dataset_indices(b))

    def test_partition_differs_across_seeds(self) -> None:
        """Different seeds produce different shard assignments."""
        dataset = _balanced_dataset()
        datasets_a = PerLabelsPartitioner.partition(dataset, n=10, lambda_=0.3, seed=1)
        datasets_b = PerLabelsPartitioner.partition(dataset, n=10, lambda_=0.3, seed=2)
        shards_a = [_dataset_indices(ds) for ds in datasets_a]
        shards_b = [_dataset_indices(ds) for ds in datasets_b]
        self.assertNotEqual(shards_a, shards_b)

    def test_partition_works_without_targets_attribute(self) -> None:
        """Falls back to per-sample indexing when dataset has no .targets attribute."""
        x = torch.randn(400, 3)
        y = torch.randint(0, 4, (400,))
        dataset = TensorDataset(x, y)
        datasets = PerLabelsPartitioner.partition(dataset, n=4, lambda_=0.5)
        all_indices = [i for ds in datasets for i in _dataset_indices(ds)]
        self.assertEqual(len(set(all_indices)), len(all_indices))  # no duplicates

    def test_handles_fully_empty_dataset(self) -> None:
        """An empty input dataset yields n empty datasets rather than crashing."""
        dataset = Subset(_balanced_dataset(), [])
        datasets = PerLabelsPartitioner.partition(dataset, n=5, lambda_=0.5)
        self.assertEqual(len(datasets), 5)
        for ds in datasets:
            self.assertEqual(len(ds), 0)

    def test_rejects_n_less_than_one(self) -> None:
        """Check raises ValueError when n < 1."""
        dataset = _balanced_dataset()
        with self.assertRaises(ValueError):
            PerLabelsPartitioner.partition(dataset, n=0, lambda_=0.5)

    def test_rejects_lambda_out_of_range(self) -> None:
        """Check raises ValueError when lambda_ is outside [0, 1]."""
        dataset = _balanced_dataset()
        with self.assertRaises(ValueError):
            PerLabelsPartitioner.partition(dataset, n=10, lambda_=-0.1)
        with self.assertRaises(ValueError):
            PerLabelsPartitioner.partition(dataset, n=10, lambda_=1.1)

    def test_rejects_nonempty_dataset_smaller_than_n(self) -> None:
        """A nonempty dataset with fewer than n samples is rejected, unlike the empty case."""
        dataset = Subset(_balanced_dataset(), [0, 1, 2])
        with self.assertRaises(ValueError):
            PerLabelsPartitioner.partition(dataset, n=10, lambda_=0.5)

    def test_num_shards_at_endpoints(self) -> None:
        """n_shards(lambda_=0) == n and n_shards(lambda_=1) == dataset_size."""
        self.assertEqual(PerLabelsPartitioner._num_shards(0.0, n=10, dataset_size=60000), 10)
        self.assertEqual(PerLabelsPartitioner._num_shards(1.0, n=10, dataset_size=60000), 60000)

    def test_num_shards_clamped_when_dataset_smaller_than_n(self) -> None:
        """_num_shards itself stays defensively clamped even for an input partition() now rejects."""
        n_shards = PerLabelsPartitioner._num_shards(0.0, n=10, dataset_size=3)
        self.assertLessEqual(n_shards, 3)
        self.assertGreaterEqual(n_shards, 1)


if __name__ == "__main__":
    unittest.main()
