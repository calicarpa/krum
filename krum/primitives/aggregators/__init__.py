"""Aggregators for Byzantine-resilient distributed learning."""

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


from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.bulyan import Bulyan
from krum.primitives.aggregators.krum import Krum
from krum.primitives.aggregators.median import Median
from krum.primitives.aggregators.multikrum import MultiKrum
from krum.primitives.aggregators.trimmed_mean import TrimmedMean

__all__ = ["Aggregator", "Average", "Bulyan", "Krum", "Median", "MultiKrum", "TrimmedMean"]
