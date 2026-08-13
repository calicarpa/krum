"""Dirichlet dataset partitioning: per-class label-skew split."""

from typing import Any, Sized, cast

import torch
from torch.utils.data import Dataset, Subset

from . import DataPartitioner


class DirichletPartitioner(DataPartitioner):
    r"""Dirichlet partitioner: per-class label-skew split, from near-IID to extreme imbalance.

    For each class :math:`k`, draws a proportion vector
    :math:`p_k \sim \mathrm{Dirichlet}(\alpha, \dots, \alpha)` of dimension
    :math:`n` (one entry per worker, summing to 1), then gives worker
    :math:`w` a :math:`p_{k,w}` fraction of class :math:`k`'s samples. Every
    sample is assigned to exactly one worker (no remainder is dropped,
    unlike :class:`~krum.primitives.data_partitioners.iid.IidPartitioner`).

    :math:`\alpha` controls the skew: as :math:`\alpha \to \infty`, every
    :math:`p_k` collapses to :math:`(1/n, \dots, 1/n)` (near-IID); as
    :math:`\alpha \to 0`, every :math:`p_k` collapses to a one-hot vector
    (each class goes almost entirely to a single worker). Matches the
    scheme of Hsu, Qi & Brown (2019), "Measuring the Effects of
    Non-Identical Data Distribution for Federated Visual Classification".

    A worker can legitimately end up with zero samples of a class, or
    (for small enough :math:`\alpha` and small :math:`n`) even zero samples
    overall — an intentional consequence of extreme skew. Such a worker
    gets an empty (but valid) dataset.
    """

    @classmethod
    def partition(
        cls,
        dataset: Dataset[Any],
        /,
        *,
        n: int,
        alpha: float,
        seed: int = 42,
        **specialized: Any,
    ) -> list[Subset[Any]]:
        r"""Split ``dataset`` across ``n`` workers via per-class Dirichlet skew.

        Args:
            dataset: Full dataset to partition across workers. Labels are
                read from ``dataset.targets`` when available (as for the
                torchvision datasets), otherwise by indexing every sample.
            n: Number of workers to split the dataset across.
            alpha: Concentration parameter of the per-class
                :math:`\mathrm{Dirichlet}(\alpha, \dots, \alpha)` draw.
                Smaller values produce more extreme label skew.
            seed: Random seed for the per-class Dirichlet draws and the
                within-class shuffle.
            **specialized: Additional keyword arguments (unused).

        Returns:
            List of ``n`` datasets, one per worker.

        Raises:
            ValueError: If ``n < 1`` or ``alpha <= 0``.
        """
        if n < 1:
            raise ValueError(f"Invalid number of workers, got {n=!r}, expected n >= 1")
        if alpha <= 0:
            raise ValueError(f"Invalid alpha, got {alpha=!r}, expected alpha > 0")

        labels = cls._extract_labels(dataset)
        classes = torch.unique(labels)

        proportions = cls._sample_proportions(alpha, classes.numel(), n, seed)
        generator = torch.Generator().manual_seed(seed)

        worker_indices: list[list[int]] = [[] for _ in range(n)]
        for class_idx, label in enumerate(classes.tolist()):
            class_positions = torch.nonzero(labels == label, as_tuple=True)[0]
            shuffled = class_positions[torch.randperm(class_positions.numel(), generator=generator)]

            boundaries = (proportions[class_idx].cumsum(0) * shuffled.numel()).long()
            boundaries[-1] = shuffled.numel()

            start = 0
            for w in range(n):
                end = int(boundaries[w])
                worker_indices[w].extend(shuffled[start:end].tolist())
                start = end

        return [Subset(dataset, worker_indices[w]) for w in range(n)]

    @staticmethod
    def _sample_proportions(alpha: float, num_classes: int, n: int, seed: int) -> torch.Tensor:
        r"""Draw the per-class :math:`\mathrm{Dirichlet}(\alpha, \dots, \alpha)` proportions.

        :class:`torch.distributions.Dirichlet` has no ``generator`` parameter — it always
        draws from the global RNG — so the seeding is scoped with
        :func:`torch.random.fork_rng` instead: the global RNG state is saved, seeded,
        sampled from, then restored, leaving it untouched on return. This keeps the
        draw reproducible without perturbing global RNG state, matching the local
        :class:`torch.Generator` convention used for the within-class shuffle below and
        throughout :mod:`~krum.primitives.data_partitioners`.

        Args:
            alpha: Concentration parameter of the symmetric Dirichlet distribution.
            num_classes: Number of classes (rows of the returned tensor).
            n: Number of workers (columns of the returned tensor; each row sums to 1).
            seed: Random seed for the draw.

        Returns:
            Tensor of shape ``(num_classes, n)``.
        """
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            concentration = torch.full((num_classes, n), alpha, dtype=torch.float64)
            return torch.distributions.Dirichlet(concentration).sample()

    @staticmethod
    def _extract_labels(dataset: Dataset[Any]) -> torch.Tensor:
        """Read the per-sample class label of every example in ``dataset``.

        Uses ``dataset.targets`` when available (as for the torchvision
        datasets), avoiding a full pass through ``__getitem__`` (which
        would needlessly apply any configured transform). Falls back to
        indexing every sample otherwise.

        Args:
            dataset: Dataset to read labels from.

        Returns:
            1-D tensor of length ``len(dataset)`` with one label per sample.
        """
        targets = getattr(dataset, "targets", None)
        if targets is not None:
            return torch.as_tensor(targets)
        return torch.tensor([dataset[i][1] for i in range(len(cast(Sized, dataset)))])
