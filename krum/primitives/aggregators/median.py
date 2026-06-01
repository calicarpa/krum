"""Median aggregator."""

import torch

from . import Aggregator


class Median(Aggregator):
    """Median aggregator."""

    def __init__(self) -> None:
        """Initialize the Median aggregator."""
        super().__init__()

    def aggregate(self, gradients: torch.Tensor) -> torch.Tensor:
        """Aggregate the gradients by computing the coordinate-wise median.

        Args:
            gradients: Tensor of shape (n, d) containing gradients from workers.

        Returns:
            Coordinate-wise median of shape (d,).
        """
        return gradients.median(dim=0).values
