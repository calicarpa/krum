"""Mixing dataset partitioning: interpolate between any two partitioners."""

from typing import Any, Sized, cast

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from . import DataPartitioner


class MixingPartitioner(DataPartitioner):
    r"""Mixing partitioner: interpolates between any two partitioners by a ratio.

    Shuffles the dataset, splits it into a :math:`(1 - \gamma)` fraction and
    a :math:`\gamma` fraction, partitions each fraction independently with
    ``p1`` and ``p2`` respectively (both across the same :math:`n` workers),
    then gives worker :math:`w` the concatenation of its ``p1``-slice and its
    ``p2``-slice.

    :math:`\gamma = 0` recovers ``p1`` alone; :math:`\gamma = 1` recovers
    ``p2`` alone. This generalizes the "gamma-similarity" scheme of
    Karimireddy, Kale, Mohri, Reddi, Stich & Suresh (ICML 2020, SCAFFOLD,
    Section 7.1) — there, ``p1`` is always an IID split and ``p2`` is always
    a sort-by-label split — to *any* pair of partitioners.

    Since the initial split accounts for every sample exactly once, and
    worker :math:`w`'s dataset is the concatenation of its (disjoint) slice
    of each half, no sample is ever assigned to two workers. Whether a
    sample can be dropped depends on ``p1``/``p2`` themselves: mixing in
    :class:`~krum.primitives.data_partitioners.iid.IidPartitioner` (which
    drops a remainder within its own slice) can still drop samples, while
    :class:`~krum.primitives.data_partitioners.dirichlet.DirichletPartitioner`
    never does.
    """

    @classmethod
    def partition(
        cls,
        dataset: Dataset[Any],
        /,
        *,
        n: int,
        p1: type[DataPartitioner],
        p2: type[DataPartitioner],
        gamma: float,
        p1_kwargs: dict[str, Any] | None = None,
        p2_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
        **specialized: Any,
    ) -> list[ConcatDataset[Any]]:
        r"""Split ``dataset`` across ``n`` workers by mixing ``p1`` and ``p2``.

        Args:
            dataset: Full dataset to partition across workers.
            n: Number of workers to split the dataset across.
            p1: Partitioner applied to the :math:`(1 - \gamma)` fraction of
                the shuffled dataset.
            p2: Partitioner applied to the :math:`\gamma` fraction.
            gamma: Mixing ratio in ``[0, 1]`` — the fraction of the dataset
                routed to ``p2`` instead of ``p1``.
            p1_kwargs: Extra keyword arguments forwarded to ``p1.partition``
                (e.g. ``{"alpha": 0.5}`` when ``p1`` is ``DirichletPartitioner``).
            p2_kwargs: Extra keyword arguments forwarded to ``p2.partition``.
            seed: Random seed for the initial shuffle, forwarded unchanged
                to both ``p1.partition`` and ``p2.partition``.
            **specialized: Additional keyword arguments (unused).

        Returns:
            List of ``n`` datasets, one per worker, each the concatenation
            of that worker's ``p1``-slice and ``p2``-slice.

        Raises:
            ValueError: If ``n < 1`` or ``gamma`` is not in ``[0, 1]``.
            TypeError: If ``p1`` or ``p2`` is not a :class:`DataPartitioner` subclass.
        """
        if n < 1:
            raise ValueError(f"Invalid number of workers, got {n=!r}, expected n >= 1")
        if not 0 <= gamma <= 1:
            raise ValueError(f"Invalid gamma, got {gamma=!r}, expected 0 <= gamma <= 1")
        if not (isinstance(p1, type) and issubclass(p1, DataPartitioner)):
            raise TypeError(f"Expected p1 to be a DataPartitioner subclass, got {p1!r}")
        if not (isinstance(p2, type) and issubclass(p2, DataPartitioner)):
            raise TypeError(f"Expected p2 to be a DataPartitioner subclass, got {p2!r}")

        dataset_size = len(cast(Sized, dataset))
        shuffled = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(seed))
        split_point = int(dataset_size * (1 - gamma))
        p1_dataset = Subset(dataset, shuffled[:split_point].tolist())
        p2_dataset = Subset(dataset, shuffled[split_point:].tolist())

        p1_datasets = p1.partition(p1_dataset, n=n, seed=seed, **(p1_kwargs or {}))
        p2_datasets = p2.partition(p2_dataset, n=n, seed=seed, **(p2_kwargs or {}))

        return [ConcatDataset([p1_datasets[w], p2_datasets[w]]) for w in range(n)]
