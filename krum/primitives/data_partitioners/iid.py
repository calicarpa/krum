"""IID dataset partitioning: shuffle, then split into equal-size shards."""

from typing import Any, Sized, cast

import torch
from torch.utils.data import Dataset, Subset

from . import DataPartitioner


class IidPartitioner(DataPartitioner):
    """IID partitioner: shuffle the dataset, then split into ``n`` equal shards.

    The dataset is shuffled and cut into ``n`` equal-size, disjoint,
    uniformly random shards, one per worker. Any remainder
    (``len(dataset) % n`` samples) is dropped.
    """

    @classmethod
    def partition(
        cls,
        dataset: Dataset[Any],
        /,
        *,
        n: int,
        seed: int = 42,
        **specialized: Any,
    ) -> list[Subset[Any]]:
        r"""Shuffle ``dataset`` and split it into ``n`` equal-size shards.

        Args:
            dataset: Full dataset to partition across workers.
            n: Number of workers to split the dataset across.
            seed: Random seed for the shard permutation.
            **specialized: Additional keyword arguments (unused).

        Returns:
            List of ``n`` datasets, each an equal-size, disjoint, uniformly
            random shard.

        Raises:
            ValueError: If ``n < 1``, or ``dataset`` is nonempty but has
                fewer than ``n`` samples.
        """
        if n < 1:
            msg = f"Invalid number of workers, got {n=!r}, expected n >= 1"
            raise ValueError(msg)

        dataset_size = len(cast(Sized, dataset))
        if 0 < dataset_size < n:
            raise ValueError(
                f"Expected at least n={n} samples to split across n workers, got dataset_size={dataset_size}"
            )
        shard_size = dataset_size // n
        shard_indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(seed))

        return [Subset(dataset, shard_indices[w * shard_size : (w + 1) * shard_size].tolist()) for w in range(n)]
