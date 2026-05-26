"""MultiKrum aggregation rule."""

import math

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
        """
        self.m = m
        super().__init__(n=n, f=f)
        self.check()

    def check(self) -> None:
        """Check parameter validity for MultiKrum rule.

        Raises:
            ValueError: If the bounds are invalid.
        """
        super().check()
        if self.n < 2 * self.f + 3:
            raise ValueError(
                f"Invalid number of Byzantine gradients to tolerate, got f = {self.f!r}, expected 1 ≤ f ≤ {(self.n - 3) // 2}"
            )
        if self.m < 1 or self.m > self.n - self.f - 2:
            raise ValueError(
                f"Invalid number of selected gradients, got m = {self.m!r}, expected 1 ≤ m ≤ {self.n - self.f - 2}"
            )

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

    def influence_ratio(self, honest_gradients: torch.Tensor, byzantine_gradients: torch.Tensor) -> float:
        """Compute the ratio of accepted Byzantine gradients.

        Args:
            honest_gradients: Tensor of shape (h, d) containing gradients from honest workers.
            byzantine_gradients: Tensor of shape (b, d) containing gradients from Byzantine workers.

        Returns:
            Ratio of accepted Byzantine gradients.
        """
        all_gradients = torch.cat([honest_gradients, byzantine_gradients], dim=0)
        num_honest = honest_gradients.size(0)

        scores = self._compute_scores(all_gradients)
        _, top_indices = torch.topk(scores, self.m, largest=False)

        byzantine_selected = (top_indices >= num_honest).sum().item()

        return byzantine_selected / self.m

    def upper_bound(self) -> float:
        """Compute the theoretical upper bound on the ratio non-Byzantine standard deviation / norm.

        Returns:
            Theoretical upper bound.
        """
        return 1 / math.sqrt(
            2 * (self.n - self.f + self.f * (self.n + self.f * (self.n - self.f - 2) - 2) / (self.n - 2 * self.f - 2))
        )
