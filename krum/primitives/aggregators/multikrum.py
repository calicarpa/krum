"""MultiKrum aggregation rule."""

import torch

from .aggregator import Aggregator


class MultiKrum(Aggregator):
    """MultiKrum aggregation rule.

    Args:
        n: Total number of workers.
        f: Number of Byzantine workers to tolerate.
        m: Number of gradients to average (1 for Krum).
    """

    def __init__(self, *, n: int, f: int, m: int) -> None:
        """Initialize MultiKrum aggregator.

        Args:
            n: Total number of workers.
            f: Number of Byzantine workers to tolerate.
            m: Number of gradients to average (1 for Krum).

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
        if m < 1 or m > n - f - 2:
            raise ValueError(f"Invalid number of selected gradients, got m = {m!r}, expected 1 ≤ m ≤ {n - f - 2}")
        if n < 2 * f + 3:
            raise ValueError(
                f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 1 ≤ f ≤ {(n - 3) // 2}"
            )
        self.n = n
        self.f = f
        self.m = m

    def _compute_scores(self, gradients: torch.Tensor) -> torch.Tensor:
        """Internal helper to compute Krum scores."""
        distances = torch.cdist(gradients, gradients, p=2.0)
        distances.fill_diagonal_(float("inf"))

        sorted_distances, _ = torch.sort(distances, dim=1)
        return sorted_distances[:, : self.n - self.f - 1].sum(dim=1)

    def aggregate(self, gradients: torch.Tensor) -> torch.Tensor:
        """Aggregate gradients using MultiKrum.

        Args:
            gradients: Tensor of shape (n, d) containing gradients from workers.

        Returns:
            Aggregated gradient of shape (d,).

        Raises:
            ValueError: If the number of gradients does not match ``n``.
        """
        if gradients.shape[0] != self.n:
            raise ValueError(f"Expected {self.n} gradients, got {gradients.shape[0]}")
        scores = self._compute_scores(gradients)
        _, top_indices = torch.topk(scores, self.m, largest=False)

        return gradients[top_indices].mean(dim=0)
