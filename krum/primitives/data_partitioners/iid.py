"""IID dataset partitioning: shuffle, then split into equal-size shards."""

from typing import Any, Sized, cast

import torch
from torch.utils.data import Dataset, Subset

from . import DataPartitioner


class IidPartitioner(DataPartitioner):
    """IID partitioner: shuffle the dataset, then split into ``n`` equal shards.

    Matches the IID baseline of McMahan et al. (AISTATS 2017): the dataset
    is shuffled and cut into ``n`` equal-size, disjoint, uniformly random
    shards, one per worker. Any remainder (``len(dataset) % n`` samples) is
    dropped.
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
            ValueError: If ``n < 1``.
        """
        if n < 1:
            raise ValueError(f"Invalid number of workers, got {n=!r}, expected n >= 1")

        dataset_size = len(cast(Sized, dataset))
        shard_size = dataset_size // n
        shard_indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(seed))

        return [
            Subset(dataset, shard_indices[w * shard_size : (w + 1) * shard_size].tolist()) for w in range(n)
        ]
