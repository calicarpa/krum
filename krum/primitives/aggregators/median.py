"""Median aggregator."""

from collections.abc import Sequence

import torch

from . import Aggregator


class Median(Aggregator):
    """Median aggregator."""

    @classmethod
    def aggregate(cls, gradients: Sequence[torch.Tensor], /) -> torch.Tensor:
        """Aggregate the gradients by computing the coordinate-wise median.

        Args:
            gradients: Sequence of Tensors containing gradients from workers.

        Returns:
            Coordinate-wise median.
        """
        return torch.stack(list(gradients)).median(dim=0).values
