"""Base class for aggregators."""

from abc import ABC, abstractmethod

import torch


class Aggregator(ABC):
    """Base class for gradient aggregation rules in Byzantine-resilient distributed learning."""

    @abstractmethod
    def aggregate(self, gradients: torch.Tensor) -> torch.Tensor:
        """Aggregate the gradients.

        Args:
            gradients: Tensor of shape (n, d) containing gradients from workers.

        Returns:
            Aggregated gradient of shape (d,).
        """
        pass

    def __call__(self, gradients: torch.Tensor) -> torch.Tensor:
        """Aggregate the gradients.

        Args:
            gradients: Tensor of shape (n, d) containing gradients from workers.

        Returns:
            Aggregated gradient of shape (d,).
        """
        return self.aggregate(gradients)
