"""Average aggregator that computes the mean of the gradients."""

import torch

from .aggregator import Aggregator


class Average(Aggregator):
    """Average aggregator that computes the mean of the gradients."""

    def __init__(self) -> None:
        """Initialize the Average aggregator."""
        super().__init__()

    def aggregate(self, gradients: torch.Tensor) -> torch.Tensor:
        """Aggregate the gradients by computing the mean.

        Args:
            gradients: Tensor of shape (n, d) containing gradients from workers.

        Returns:
            Mean of the gradients of shape (d,).
        """
        return gradients.mean(0)
