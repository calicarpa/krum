"""Nearest-neighbor averaging aggregator."""

import torch

from .aggregator import Aggregator


class NearestNeighborAverage(Aggregator):
    """Average the ``num_closest`` vectors nearest to a call-specific pivot.

    The rule keeps the ``num_closest`` vectors with smallest Euclidean distance
    to the pivot, then returns their mean. The pivot belongs to a single
    aggregation call, so it is passed to :meth:`aggregate` rather than stored on
    the aggregator.

    How many vectors to keep is a caller policy (e.g. ``n - f`` for plain
    nearest-neighbor averaging, ``n - 2f`` for MoNNA-style model mixing); this
    rule only needs the resulting count, not ``n`` or ``f``.

    Args:
        num_closest: Number of nearest vectors to average.
    """

    def __init__(self, *, num_closest: int):
        """Initialize the nearest-neighbor averaging aggregator.

        Args:
            num_closest: Number of nearest vectors to average.

        Raises:
            ValueError: If ``num_closest`` is not positive.
        """
        super().__init__()
        if num_closest < 1:
            raise ValueError(f"Expected num_closest to be at least 1, got {num_closest!r}")
        self.num_closest = num_closest

    def aggregate(self, gradients: torch.Tensor, *, pivot: torch.Tensor) -> torch.Tensor:
        """Average the ``num_closest`` vectors nearest to the pivot.

        Args:
            gradients: Tensor of shape ``(m, d)`` containing candidate vectors.
            pivot: Tensor of shape ``(d,)`` used as the distance reference.

        Returns:
            Mean of the ``num_closest`` closest vectors, shape ``(d,)``.

        Raises:
            ValueError: If the input shapes are invalid or fewer than
                ``num_closest`` candidates are supplied.
        """
        if gradients.ndim != 2:
            raise ValueError(f"Expected gradients with shape (m, d), got shape {tuple(gradients.shape)!r}")
        if gradients.shape[0] < self.num_closest:
            raise ValueError(f"Expected at least num_closest={self.num_closest} gradients, got {gradients.shape[0]!r}")
        if pivot.shape != gradients.shape[1:]:
            raise ValueError(f"Expected pivot with shape {tuple(gradients.shape[1:])!r}, got {tuple(pivot.shape)!r}")

        distances = torch.linalg.vector_norm(gradients - pivot, dim=1)
        closest = torch.argsort(distances, stable=True)[: self.num_closest]
        return gradients[closest].mean(0)
