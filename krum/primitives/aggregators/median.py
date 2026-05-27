"""Median aggregator."""

import torch

from .aggregator import Aggregator


class Median(Aggregator):
    """Median aggregator.

    Args:
        n: Total number of workers.
        f: Number of Byzantine workers to tolerate.
    """

    def __init__(self, *, n: int, f: int):
        """Initialize the Median aggregator.

        Args:
            n: Total number of workers.
            f: Number of Byzantine workers to tolerate.
        """
        super().__init__(n=n, f=f)

    def aggregate(self, gradients: torch.Tensor) -> torch.Tensor:
        """Aggregate the gradients by computing the coordinate-wise median.

        Args:
            gradients: Tensor of shape (n, d) containing gradients from workers.

        Returns:
            Coordinate-wise median of shape (d,).
        """
        return gradients.median(dim=0).values
