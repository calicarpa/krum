"""Trimmed Mean aggregator."""

from collections.abc import Sequence

import torch

from . import Aggregator


class TrimmedMean(Aggregator):
    """Trimmed Mean aggregator.

    Computes the coordinate-wise trimmed mean by removing the ``f`` smallest
    and ``f`` largest values per coordinate, then averaging the remaining ones.

    Args:
        f: Number of Byzantine workers to tolerate.
    """

    def __init__(self, *, f: int):
        """Initialize the Trimmed Mean aggregator.

        Args:
            f: Number of Byzantine workers to tolerate.
        """
        super().__init__()
        if f < 0:
            raise ValueError(f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 0 ≤ f")
        self.f = f

    def aggregate(self, gradients: Sequence[torch.Tensor]) -> torch.Tensor:
        """Aggregate the gradients by computing the coordinate-wise trimmed mean.

        Args:
            gradients: Sequence of Tensors containing gradients from workers.

        Returns:
            Coordinate-wise trimmed mean.

        Raises:
            ValueError: If the number of gradients is insufficient.
        """
        if len(gradients) <= 2 * self.f:
            raise ValueError(f"At least 2f+1 = {2 * self.f + 1} gradients required, got {len(gradients)}")
        stacked = torch.stack(list(gradients))
        return stacked.sort(dim=0).values[self.f : -self.f].mean(dim=0)
