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
        if m < 1 or m > n - f - 2:
            raise ValueError(
                f"Invalid number of selected gradients, got m = {self.m!r}, expected 1 ≤ m ≤ {self.n - self.f - 2}"
            )
        if n < 2 * f + 3:
            raise ValueError(
                f"Invalid number of Byzantine gradients to tolerate, got f = {self.f!r}, expected 1 ≤ f ≤ {(self.n - 3) // 2}"
            )
        self.m = m
        super().__init__(n=n, f=f)

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
        """
        scores = self._compute_scores(gradients)
        _, top_indices = torch.topk(scores, self.m, largest=False)

        return gradients[top_indices].mean(dim=0)
