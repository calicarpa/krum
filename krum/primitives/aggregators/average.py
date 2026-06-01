"""Average aggregator that computes the mean of the gradients."""

from collections.abc import Sequence

import torch

from . import Aggregator


class Average(Aggregator):
    """Average aggregator that computes the mean of the gradients."""

    def __init__(self) -> None:
        """Initialize the Average aggregator."""
        super().__init__()

    def aggregate(self, gradients: Sequence[torch.Tensor]) -> torch.Tensor:
        """Aggregate the gradients by computing the mean.

        Args:
            gradients: Sequence of Tensors containing gradients from workers.

        Returns:
            Mean of the gradients.
        """
        return torch.stack(list(gradients)).mean(0)
