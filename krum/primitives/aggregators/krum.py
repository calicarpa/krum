"""Krum aggregation rule."""

from collections.abc import Sequence

import torch

from .multikrum import MultiKrum


class Krum(MultiKrum):
    """Krum aggregation rule."""

    @classmethod
    def aggregate(cls, gradients: Sequence[torch.Tensor], /, *, n: int, f: int) -> torch.Tensor:
        """Aggregate gradients using Krum.

        Args:
            gradients: Sequence of Tensors containing gradients from workers.
            n: Total number of workers.
            f: Number of Byzantine workers to tolerate.

        Returns:
            Aggregated gradient of shape (d,).
        """
        return MultiKrum.aggregate(gradients, n=n, f=f, m=1)
