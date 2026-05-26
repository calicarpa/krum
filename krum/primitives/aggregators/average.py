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
        self.check()

    def aggregate(self, gradients: torch.Tensor) -> torch.Tensor:
        """Aggregate the gradients by computing the mean.

        Args:
            gradients: Tensor of shape (n, d) containing gradients from workers.

        Returns:
            Mean of the gradients of shape (d,).
        """
        return gradients.mean(0)

    def influence_ratio(self, honest_gradients: torch.Tensor, byzantine_gradients: torch.Tensor) -> float:
        """Compute the ratio of accepted Byzantine gradients.

        Args:
            honest_gradients: Tensor of shape (h, d) containing gradients from honest workers.
            byzantine_gradients: Tensor of shape (b, d) containing gradients from Byzantine workers.

        Returns:
            Ratio of accepted Byzantine gradients.
        """
        total = honest_gradients.size(0) + byzantine_gradients.size(0)
        if total == 0:
            return 0.0
        return byzantine_gradients.size(0) / total
