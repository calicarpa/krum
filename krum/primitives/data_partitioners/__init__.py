"""Dataset-to-worker partitioning strategies for simulations.

A :class:`DataPartitioner` turns one dataset into ``n`` per-worker
:class:`~torch.utils.data.Dataset` instances. Both
:class:`~krum.simulations.centralised.CentralisedSimulation` and
:class:`~krum.simulations.decentralised.DecentralisedSimulation` consume
this same shape, so partitioning is entirely the caller's responsibility,
IID or not. Wrapping each worker's dataset into a
:class:`~torch.utils.data.DataLoader` (batch size, shuffling) is the
simulation's job, not the partitioner's — that separation is what lets
partitioners compose (e.g. mixing two partitioners' outputs) without
reaching back into a ``DataLoader`` to get at the underlying samples.

Like :mod:`~krum.primitives.aggregators` and :mod:`~krum.primitives.attacks`,
each strategy is **stateless**: a ``@classmethod`` invoked directly on the
class. The dataset is the sole positional argument; ``n``, ``seed``, and any
partitioner-specific hyperparameters are keyword-only.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Sized, cast

import torch
from torch.utils.data import Dataset


class DataPartitioner(ABC):
    """Abstract base class for stateless dataset-to-worker partitioners.

    Subclasses implement :meth:`partition` as a ``@classmethod`` — no
    instance state is required, and the caller invokes the strategy
    directly on the class. The dataset is the sole positional argument;
    ``n``, ``seed``, and any partitioner-specific hyperparameters are
    keyword-only.
    """

    @classmethod
    @abstractmethod
    def partition(
        cls,
        dataset: Dataset[Any],
        /,
        *,
        n: int,
        seed: int = 42,
        **specialized: Any,
    ) -> Sequence[Dataset[Any]]:
        r"""Split ``dataset`` into ``n`` per-worker datasets.

        Args:
            dataset: Full dataset to partition across workers.
            n: Number of workers to split the dataset across.
            seed: Random seed for reproducibility.
            **specialized: Keyword-only arguments specific to each
                partitioning strategy (e.g. :math:`\alpha` for label skew).

        Returns:
            Sequence of ``n`` datasets, one per worker.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError


def _extract_labels(dataset: Dataset[Any]) -> torch.Tensor:
    """Read the per-sample class label of every example in ``dataset``.

    Uses ``dataset.targets`` when available (as for the torchvision
    datasets), avoiding a full pass through ``__getitem__`` (which would
    needlessly apply any configured transform). Falls back to indexing
    every sample otherwise. Shared by every label-aware partitioner (e.g.
    :class:`~krum.primitives.data_partitioners.dirichlet.DirichletPartitioner`,
    :class:`~krum.primitives.data_partitioners.per_labels.PerLabelsPartitioner`).

    Args:
        dataset: Dataset to read labels from.

    Returns:
        1-D tensor of length ``len(dataset)`` with one label per sample.
    """
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return torch.as_tensor(targets)
    return torch.tensor([dataset[i][1] for i in range(len(cast(Sized, dataset)))])
