"""Average aggregator that computes the mean of the gradients."""

import torch

from .aggregator import Aggregator


class Average(Aggregator):
    """Average aggregator that computes the mean of the gradients.

    Args:
        n: Total number of workers.
        f: Number of Byzantine workers to tolerate.
    """

    def __init__(self, *, n: int, f: int):
        """Initialize the Average aggregator.

        Args:
            n: Total number of workers.
            f: Number of Byzantine workers to tolerate.
        """
        super().__init__(n=n, f=f)

    def aggregate(self, gradients: torch.Tensor) -> torch.Tensor:
        """Aggregate the gradients by computing the mean.

        Args:
            gradients: Tensor of shape (n, d) containing gradients from workers.

        Returns:
            Mean of the gradients of shape (d,).
        """
        return gradients.mean(0)
