"""Nearest-neighbor averaging aggregator."""

import torch

from .aggregator import Aggregator


class NearestNeighbor(Aggregator):
    """Average the vectors closest to a call-specific pivot vector.

    The rule keeps the ``n - f`` vectors with smallest Euclidean distance to
    the pivot, then returns their mean. The pivot is provided to
    :meth:`aggregate` because it belongs to one aggregation call, not to the
    aggregator configuration.

    Args:
        n: Total number of candidate vectors.
        f: Number of Byzantine vectors to tolerate.
    """

    def __init__(self, *, n: int, f: int):
        """Initialize the nearest-neighbor aggregator.

        Args:
            n: Total number of candidate vectors.
            f: Number of Byzantine vectors to tolerate.

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__()
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

    def aggregate(self, gradients: torch.Tensor, *, pivot: torch.Tensor) -> torch.Tensor:
        """Aggregate the closest gradients to the pivot.

        Args:
            gradients: Tensor of shape ``(n, d)`` containing candidate vectors.
            pivot: Tensor of shape ``(d,)`` used as the distance reference.

        Returns:
            Mean of the ``n - f`` closest vectors, shape ``(d,)``.

        Raises:
            ValueError: If the input shapes do not match the aggregator
                configuration.
        """
        if gradients.ndim != 2:
            raise ValueError(f"Expected gradients with shape (n, d), got shape {tuple(gradients.shape)!r}")
        if gradients.shape[0] != self.n:
            raise ValueError(f"Expected {self.n} gradients, got {gradients.shape[0]!r}")
        if pivot.shape != gradients.shape[1:]:
            raise ValueError(f"Expected pivot with shape {tuple(gradients.shape[1:])!r}, got {tuple(pivot.shape)!r}")

        distances = torch.linalg.vector_norm(gradients - pivot, dim=1)
        closest = torch.argsort(distances, stable=True)[: self.n - self.f]
        return gradients[closest].mean(0)
