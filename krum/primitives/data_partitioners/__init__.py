"""Dataset-to-worker partitioning strategies for simulations.

A :class:`DataPartitioner` turns one dataset into ``n`` per-worker
:class:`~torch.utils.data.DataLoader` instances. Both
:class:`~krum.simulations.centralised.CentralisedSimulation` and
:class:`~krum.simulations.decentralised.DecentralisedSimulation` consume
this same shape, so partitioning is entirely the caller's responsibility,
IID or not.

Like :mod:`~krum.primitives.aggregators` and :mod:`~krum.primitives.attacks`,
each strategy is **stateless**: a ``@classmethod`` invoked directly on the
class. The dataset is the sole positional argument; ``n``, ``batch_size``,
``seed``, and any partitioner-specific hyperparameters are keyword-only.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from torch.utils.data import DataLoader, Dataset


class DataPartitioner(ABC):
    """Abstract base class for stateless dataset-to-worker partitioners.

    Subclasses implement :meth:`partition` as a ``@classmethod`` — no
    instance state is required, and the caller invokes the strategy
    directly on the class. The dataset is the sole positional argument;
    ``n``, ``batch_size``, ``seed``, and any partitioner-specific
    hyperparameters are keyword-only.
    """

    @classmethod
    @abstractmethod
    def partition(
        cls,
        dataset: Dataset[Any],
        /,
        *,
        n: int,
        batch_size: int,
        seed: int = 42,
        **specialized: Any,
    ) -> Sequence[DataLoader[Any]]:
        """Split ``dataset`` into ``n`` per-worker dataloaders.

        Args:
            dataset: Full dataset to partition across workers.
            n: Number of workers to split the dataset across.
            batch_size: Mini-batch size for every worker's ``DataLoader``.
            seed: Random seed for reproducibility.
            **specialized: Keyword-only arguments specific to each
                partitioning strategy.

        Returns:
            Sequence of ``n`` dataloaders, one per worker.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError
