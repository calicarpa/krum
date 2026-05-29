"""Trimmed Mean aggregator."""

import torch

from .aggregator import Aggregator


class TrimmedMean(Aggregator):
    """Trimmed Mean aggregator.

    Computes the coordinate-wise trimmed mean by removing the ``f`` smallest
    and ``f`` largest values per coordinate, then averaging the remaining ones.

    Args:
        n: Total number of workers.
        f: Number of Byzantine workers to tolerate.
    """

    def __init__(self, *, n: int, f: int):
        """Initialize the Trimmed Mean aggregator.

        Args:
            n: Total number of workers.
            f: Number of Byzantine workers to tolerate.
        """
        if n < 1:
            raise ValueError(f"Expected a list of at least one gradient to aggregate, got {n!r}")
        if f < 0:
            raise ValueError(f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 0 ≤ f")
        if f > n:
            raise ValueError(
                f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected f ≤ n = {n!r}"
            )
        self.n = n
        self.f = f
        super().__init__()

    def aggregate(self, gradients: torch.Tensor) -> torch.Tensor:
        """Aggregate the gradients by computing the coordinate-wise trimmed mean.

        Args:
            gradients: Tensor of shape (n, d) containing gradients from workers.

        Returns:
            Coordinate-wise trimmed mean of shape (d,).
        """
        return gradients.sort(dim=0).values[self.f : -self.f].mean(dim=0)
