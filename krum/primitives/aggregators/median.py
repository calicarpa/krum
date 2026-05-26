"""Median aggregator."""

import numpy as np
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
        self.check()

    def aggregate(self, gradients: torch.Tensor) -> torch.Tensor:
        """Aggregate the gradients by computing the coordinate-wise median.

        Args:
            gradients: Tensor of shape (n, d) containing gradients from workers.

        Returns:
            Coordinate-wise median of shape (d,).
        """
        return gradients.median(dim=0).values

    def influence_ratio(self, honest_gradients: torch.Tensor, byzantine_gradients: torch.Tensor) -> float:
        """Compute the ratio of accepted Byzantine gradients.

        Args:
            honest_gradients: Tensor of shape (h, d) containing gradients from honest workers.
            byzantine_gradients: Tensor of shape (b, d) containing gradients from Byzantine workers.

        Returns:
            Fraction of dimensions where the median is influenced by Byzantine gradients.
        """
        all_gradients = torch.cat([honest_gradients, byzantine_gradients], dim=0)

        indices = all_gradients.median(dim=0).indices

        num_honest = honest_gradients.size(0)
        byzantine_count = (indices >= num_honest).sum().item()

        return byzantine_count / indices.numel()

    def upper_bound(self) -> float:
        """Compute the theoretical upper bound on the ratio non-Byzantine standard deviation / norm.

        Returns:
            Theoretical upper bound.
        """
        return 1 / np.sqrt(self.n - self.f)
