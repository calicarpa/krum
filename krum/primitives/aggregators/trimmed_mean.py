"""Trimmed Mean aggregator."""

from collections.abc import Sequence

import torch

from . import Aggregator


class TrimmedMean(Aggregator):
    """Trimmed Mean aggregator.

    Computes the coordinate-wise trimmed mean by removing the ``f`` smallest
    and ``f`` largest values per coordinate, then averaging the remaining ones.
    """

    @classmethod
    def aggregate(cls, gradients: Sequence[torch.Tensor], /, *, f: int) -> torch.Tensor:
        """Aggregate the gradients by computing the coordinate-wise trimmed mean.

        Args:
            gradients: Sequence of Tensors containing gradients from workers.
            f: Number of Byzantine workers to tolerate.

        Returns:
            Coordinate-wise trimmed mean.

        Raises:
            ValueError: If the number of gradients is insufficient.
        """
        if f < 0:
            raise ValueError(f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 0 ≤ f")
        if len(gradients) <= 2 * f:
            raise ValueError(f"At least 2f+1 = {2 * f + 1} gradients required, got {len(gradients)}")
        stacked = torch.stack(list(gradients))
        return stacked.sort(dim=0).values[f:-f].mean(dim=0)
