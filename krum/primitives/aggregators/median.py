"""Median aggregator."""

from collections.abc import Sequence

import torch

from . import Aggregator


class Median(Aggregator):
    """Median aggregator."""

    def __init__(self) -> None:
        """Initialize the Median aggregator."""
        super().__init__()

    def aggregate(self, gradients: Sequence[torch.Tensor]) -> torch.Tensor:
        """Aggregate the gradients by computing the coordinate-wise median.

        Args:
            gradients: Sequence of Tensors containing gradients from workers.

        Returns:
            Coordinate-wise median.
        """
        return torch.stack(list(gradients)).median(dim=0).values
