"""Plain mean (arithmetic average) aggregation rule — non-robust baseline."""

from collections.abc import Sequence

import torch

from . import Aggregator


class Average(Aggregator):
    """Plain mean of all worker gradients (no Byzantine resilience).

    Included as a non-robust baseline. A single adversarial worker with an
    arbitrarily large gradient can drive the aggregated gradient arbitrarily
    far from the honest mean, so this rule has no Byzantine resilience
    guarantees.

    Args:
        gradients: Sequence of 1-D tensors, one per worker.

    Returns:
        Element-wise mean of the gradients, of shape ``(d,)``.
    """

    @classmethod
    def aggregate(cls, gradients: Sequence[torch.Tensor], /) -> torch.Tensor:
        """Aggregate the gradients by computing the element-wise mean.

        Args:
            gradients: Sequence of 1-D tensors containing gradients from workers.

        Returns:
            Element-wise mean of the gradients, of shape ``(d,)``.
        """
        return torch.stack(list(gradients)).mean(0)
