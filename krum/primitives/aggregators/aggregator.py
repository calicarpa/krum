"""Base class for aggregators."""

from abc import ABC, abstractmethod

import torch


class Aggregator(ABC):
    """Base class for gradient aggregation rules in Byzantine-resilient distributed learning.

    Args:
        n: Total number of workers.
        f: Number of Byzantine workers to tolerate.
    """

    def __init__(self, *, n: int, f: int):
        """Initialize the aggregator.

        Args:
            n: Total number of workers.
            f: Number of Byzantine workers to tolerate.

        Raises:
            ValueError: If parameters are invalid.
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

    @abstractmethod
    def aggregate(self, gradients: torch.Tensor) -> torch.Tensor:
        """Aggregate the gradients.

        Args:
            gradients: Tensor of shape (n, d) containing gradients from workers.

        Returns:
            Aggregated gradient of shape (d,).
        """
        pass

    def upper_bound(self) -> float:
        """Compute the theoretical upper bound on the ratio non-Byzantine standard deviation / norm.

        Returns:
            Theoretical upper bound.
        """
        return float("nan")

    def influence_ratio(self, honest_gradients: torch.Tensor, byzantine_gradients: torch.Tensor) -> float:
        """Compute the ratio of accepted Byzantine gradients.

        Args:
            honest_gradients: Tensor of shape (h, d) containing gradients from honest workers.
            byzantine_gradients: Tensor of shape (b, d) containing gradients from Byzantine workers.

        Returns:
            Ratio of accepted Byzantine gradients.
        """
        return float("nan")

    def __call__(self, gradients: torch.Tensor) -> torch.Tensor:
        """Aggregate the gradients.

        Args:
            gradients: Tensor of shape (n, d) containing gradients from workers.

        Returns:
            Aggregated gradient of shape (d,).
        """
        return self.aggregate(gradients)
