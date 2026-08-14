"""Per-labels dataset partitioning: shard-granularity interpolation from pathological skew to IID."""

from typing import Any

import torch
from torch.utils.data import Dataset, Subset

from . import DataPartitioner, _extract_labels


class PerLabelsPartitioner(DataPartitioner):
    r"""Per-labels partitioner: one sort-by-label mechanism spanning pathological skew to IID.

    Sorts the dataset by label, cuts it into ``n_shards`` equal-size
    contiguous shards, shuffles the shard order, then deals shards to
    workers round-robin (worker :math:`w` gets shards :math:`w, w+n, w+2n,
    \dots`). Round-robin — rather than a fixed block of shards per worker —
    means any ``n_shards % n`` remainder shards are spread one at a time
    across the first few workers instead of being dropped, so worker
    dataset sizes never differ by more than one shard.

    ``n_shards`` is itself controlled by :math:`\lambda \in [0, 1]` ("iid-ness"),
    interpolating geometrically between the two extremes:

    .. math::
        \text{n\_shards} = n \cdot \left(\frac{N}{n}\right)^{\lambda}

    where :math:`N` is the dataset size. At :math:`\lambda = 0`, ``n_shards
    = n`` — one giant, near-single-label shard per worker, the most
    pathological split this mechanism can produce. At :math:`\lambda = 1`,
    ``n_shards = N`` — every shard is a single sample, sorting becomes
    irrelevant at that granularity, and shuffle-then-round-robin reduces to
    exactly what :class:`~krum.primitives.data_partitioners.iid.IidPartitioner`
    already does — recovering IID as a special case of the same mechanism,
    rather than needing a separate algorithm for it. Geometric (rather than
    linear) interpolation is deliberate: :math:`n` and :math:`N` typically
    span several orders of magnitude, and empirically, linear interpolation
    spends almost its entire range indistinguishable from IID, with the
    only interesting transition crammed into a tiny sliver near
    :math:`\lambda = 0`.

    This is not a reproduction of any published scheme — it is an original
    design, built to unify IID and pathological sort-by-label skew under
    one continuously tunable mechanism instead of two separate
    partitioners.
    """

    @classmethod
    def partition(
        cls,
        dataset: Dataset[Any],
        /,
        *,
        n: int,
        lambda_: float,
        seed: int = 42,
        **specialized: Any,
    ) -> list[Subset[Any]]:
        r"""Split ``dataset`` across ``n`` workers via shard-granularity interpolation.

        Args:
            dataset: Full dataset to partition across workers. Labels are
                read from ``dataset.targets`` when available (as for the
                torchvision datasets), otherwise by indexing every sample.
            n: Number of workers to split the dataset across.
            lambda_: Iid-ness in ``[0, 1]``. ``0`` is the most pathological
                split this mechanism can produce (one shard per worker);
                ``1`` recovers plain IID (one sample per shard).
            seed: Random seed for the shard-order shuffle.
            **specialized: Additional keyword arguments (unused).

        Returns:
            List of ``n`` datasets, one per worker. Any remainder
            (``len(dataset) % n_shards`` samples, past the last full shard)
            is dropped, as in ``IidPartitioner``.

        Raises:
            ValueError: If ``n < 1``, ``lambda_`` is not in ``[0, 1]``, or
                ``dataset`` is nonempty but has fewer than ``n`` samples.
        """
        if n < 1:
            raise ValueError(f"Invalid number of workers, got {n=!r}, expected n >= 1")
        if not 0 <= lambda_ <= 1:
            raise ValueError(f"Invalid lambda_, got {lambda_=!r}, expected 0 <= lambda_ <= 1")

        labels = _extract_labels(dataset)
        dataset_size = labels.numel()
        if dataset_size == 0:
            return [Subset(dataset, []) for _ in range(n)]
        if dataset_size < n:
            raise ValueError(
                f"Expected at least n={n} samples to split across n workers, got dataset_size={dataset_size}"
            )

        n_shards = cls._num_shards(lambda_, n, dataset_size)
        sorted_indices = torch.argsort(labels, stable=True)

        shard_size = dataset_size // n_shards
        generator = torch.Generator().manual_seed(seed)
        shard_order = torch.randperm(n_shards, generator=generator)

        worker_indices: list[list[int]] = [[] for _ in range(n)]
        for position, shard_idx in enumerate(shard_order.tolist()):
            start = shard_idx * shard_size
            end = start + shard_size
            worker_indices[position % n].extend(sorted_indices[start:end].tolist())

        return [Subset(dataset, worker_indices[w]) for w in range(n)]

    @staticmethod
    def _num_shards(lambda_: float, n: int, dataset_size: int) -> int:
        r"""Compute the shard count for a given :math:`\lambda`.

        Args:
            lambda_: Iid-ness in ``[0, 1]``.
            n: Number of workers.
            dataset_size: Total number of samples (:math:`N`), at least 1.

        Returns:
            ``n_shards``, geometrically interpolated between ``n`` (at
            ``lambda_ = 0``) and ``dataset_size`` (at ``lambda_ = 1``),
            clamped to ``[1, dataset_size]`` as a safety net against
            floating-point rounding at the extremes (``partition`` itself
            guarantees ``dataset_size >= n``, so the clamp should not
            otherwise engage).
        """
        raw = n * (dataset_size / n) ** lambda_
        return max(1, min(dataset_size, round(raw)))
