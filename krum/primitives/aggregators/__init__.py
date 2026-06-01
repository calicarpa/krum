"""Aggregators for Byzantine-resilient distributed learning."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch


class Aggregator(ABC):
    """Base class for gradient aggregation rules in Byzantine-resilient distributed learning."""

    @classmethod
    @abstractmethod
    def aggregate(cls, gradients: Sequence[torch.Tensor], /, **specialized: Any) -> torch.Tensor:
        """Aggregate the gradients.

        Args:
            gradients: Sequence of Tensors containing gradients from workers.
            **specialized: Keyword-only arguments specific to each aggregation rule.

        Returns:
            Aggregated gradient of shape (d,).
        """
        pass
