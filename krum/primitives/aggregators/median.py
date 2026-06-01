"""Coordinate-wise median aggregation rule."""

from collections.abc import Sequence

import torch

from . import Aggregator


class Median(Aggregator):
    """Coordinate-wise geometric median of the worker gradients.

    The aggregated gradient is the coordinate-wise median of the worker
    gradients. This provides basic Byzantine resilience: a single
    adversarial worker can shift at most one sample per coordinate away
    from the true median.

    Args:
        gradients: Sequence of 1-D tensors, one per worker.

    Returns:
        Coordinate-wise median of the gradients, of shape ``(d,)``.
    """

    @classmethod
    def aggregate(cls, gradients: Sequence[torch.Tensor], /) -> torch.Tensor:
        """Aggregate the gradients by computing the coordinate-wise median.

        Args:
            gradients: Sequence of 1-D tensors containing gradients from workers.

        Returns:
            Coordinate-wise median of the gradients, of shape ``(d,)``.
        """
        return torch.stack(list(gradients)).median(dim=0).values
