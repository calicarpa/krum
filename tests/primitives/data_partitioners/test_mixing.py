"""Tests for the mixing partitioner."""

import unittest
from typing import Any, Sized, cast

import torch
from torch.utils.data import Dataset, TensorDataset

from krum.primitives.data_partitioners.dirichlet import DirichletPartitioner
from krum.primitives.data_partitioners.iid import IidPartitioner
from krum.primitives.data_partitioners.mixing import MixingPartitioner


def _dummy_dataset(size: int = 100) -> TensorDataset:
    x = torch.arange(size, dtype=torch.float32).unsqueeze(1)
    y = torch.zeros(size, dtype=torch.int64)
    return TensorDataset(x, y)


def _values(dataset: Dataset) -> list[int]:
    """Read back the distinctive x-value of every sample, regardless of nesting."""
    return [int(dataset[i][0].item()) for i in range(len(cast(Sized, dataset)))]


class MixingPartitionerTest(unittest.TestCase):
    """Test MixingPartitioner."""

    def test_partition_returns_n_datasets(self) -> None:
        """Partition returns one dataset per worker."""
        dataset = _dummy_dataset(100)
        datasets = MixingPartitioner.partition(
            dataset, n=10, p1=IidPartitioner, p2=DirichletPartitioner, gamma=0.5, p2_kwargs={"alpha": 1.0}
        )
        self.assertEqual(len(datasets), 10)

    def test_partition_covers_dataset_without_duplication(self) -> None:
        """No sample is assigned to two workers, and (with two exact partitioners) none is dropped."""
        dataset = _dummy_dataset(100)
        datasets = MixingPartitioner.partition(
            dataset,
            n=10,
            p1=DirichletPartitioner,
            p1_kwargs={"alpha": 1.0},
            p2=DirichletPartitioner,
            p2_kwargs={"alpha": 1.0},
            gamma=0.5,
        )
        all_values = [v for ds in datasets for v in _values(ds)]
        self.assertEqual(sorted(all_values), list(range(100)))

    def test_gamma_zero_uses_only_p1(self) -> None:
        """gamma=0 routes the entire dataset through p1; p2 never sees a sample."""
        dataset = _dummy_dataset(100)
        datasets = MixingPartitioner.partition(
            dataset, n=10, p1=IidPartitioner, p2=DirichletPartitioner, gamma=0.0, p2_kwargs={"alpha": 1.0}
        )
        all_values = [v for ds in datasets for v in _values(ds)]
        self.assertEqual(sorted(all_values), list(range(100)))

    def test_gamma_one_uses_only_p2(self) -> None:
        """gamma=1 routes the entire dataset through p2; p1 never sees a sample."""
        dataset = _dummy_dataset(100)
        datasets = MixingPartitioner.partition(
            dataset, n=10, p1=IidPartitioner, p2=DirichletPartitioner, gamma=1.0, p2_kwargs={"alpha": 1.0}
        )
        all_values = [v for ds in datasets for v in _values(ds)]
        self.assertEqual(sorted(all_values), list(range(100)))

    def test_partition_is_deterministic_given_seed(self) -> None:
        """Same seed produces the same per-worker assignment."""
        dataset = _dummy_dataset(100)
        kwargs: dict[str, Any] = {
            "n": 10,
            "p1": IidPartitioner,
            "p2": DirichletPartitioner,
            "gamma": 0.5,
            "p2_kwargs": {"alpha": 1.0},
        }
        datasets_a = MixingPartitioner.partition(dataset, seed=7, **kwargs)
        datasets_b = MixingPartitioner.partition(dataset, seed=7, **kwargs)
        for a, b in zip(datasets_a, datasets_b, strict=True):
            self.assertEqual(_values(a), _values(b))

    def test_partition_differs_across_seeds(self) -> None:
        """Different seeds produce different assignments."""
        dataset = _dummy_dataset(100)
        kwargs: dict[str, Any] = {
            "n": 10,
            "p1": IidPartitioner,
            "p2": DirichletPartitioner,
            "gamma": 0.5,
            "p2_kwargs": {"alpha": 1.0},
        }
        datasets_a = MixingPartitioner.partition(dataset, seed=1, **kwargs)
        datasets_b = MixingPartitioner.partition(dataset, seed=2, **kwargs)
        values_a = [_values(ds) for ds in datasets_a]
        values_b = [_values(ds) for ds in datasets_b]
        self.assertNotEqual(values_a, values_b)

    def test_forwards_kwargs_to_p1_and_p2(self) -> None:
        """p1_kwargs/p2_kwargs actually reach the sub-partitioners' partition() calls."""
        dataset = _dummy_dataset(100)
        # DirichletPartitioner.alpha has no default: omitting it via p2_kwargs
        # must surface as a TypeError from DirichletPartitioner.partition itself,
        # proving p2_kwargs is genuinely forwarded rather than silently dropped.
        with self.assertRaises(TypeError):
            MixingPartitioner.partition(dataset, n=10, p1=IidPartitioner, p2=DirichletPartitioner, gamma=0.5)

    def test_rejects_n_less_than_one(self) -> None:
        """Check raises ValueError when n < 1."""
        dataset = _dummy_dataset(100)
        with self.assertRaises(ValueError):
            MixingPartitioner.partition(dataset, n=0, p1=IidPartitioner, p2=IidPartitioner, gamma=0.5)

    def test_rejects_gamma_out_of_range(self) -> None:
        """Check raises ValueError when gamma is outside [0, 1]."""
        dataset = _dummy_dataset(100)
        with self.assertRaises(ValueError):
            MixingPartitioner.partition(dataset, n=10, p1=IidPartitioner, p2=IidPartitioner, gamma=-0.1)
        with self.assertRaises(ValueError):
            MixingPartitioner.partition(dataset, n=10, p1=IidPartitioner, p2=IidPartitioner, gamma=1.1)

    def test_rejects_non_partitioner_p1(self) -> None:
        """Check raises TypeError when p1 is not a DataPartitioner subclass."""
        dataset = _dummy_dataset(100)
        with self.assertRaises(TypeError):
            MixingPartitioner.partition(dataset, n=10, p1=object, p2=IidPartitioner, gamma=0.5)  # ty:ignore[invalid-argument-type]

    def test_rejects_non_partitioner_p2(self) -> None:
        """Check raises TypeError when p2 is not a DataPartitioner subclass."""
        dataset = _dummy_dataset(100)
        with self.assertRaises(TypeError):
            MixingPartitioner.partition(dataset, n=10, p1=IidPartitioner, p2=object, gamma=0.5)  # ty:ignore[invalid-argument-type]


if __name__ == "__main__":
    unittest.main()
