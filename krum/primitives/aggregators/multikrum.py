"""MultiKrum aggregation rule."""

from collections.abc import Sequence

import torch

from . import Aggregator


class MultiKrum(Aggregator):
    """MultiKrum aggregation rule."""

    @classmethod
    def aggregate(cls, gradients: Sequence[torch.Tensor], /, *, n: int, f: int, m: int) -> torch.Tensor:
        """Aggregate gradients using MultiKrum.

        Args:
            gradients: Sequence of Tensors containing gradients from workers.
            n: Total number of workers.
            f: Number of Byzantine workers to tolerate.
            m: Number of gradients to average (1 for Krum).

        Returns:
            Aggregated gradient of shape (d,).

        Raises:
            ValueError: If parameters are invalid.
        """
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
        if len(gradients) != n:
            raise ValueError(f"Expected {n} gradients, got {len(gradients)}")
        stacked = torch.stack(list(gradients))
        scores = cls._compute_scores(stacked, n=n, f=f)
        _, top_indices = torch.topk(scores, m, largest=False)

        return stacked[top_indices].mean(dim=0)

    @staticmethod
    def _compute_scores(stacked: torch.Tensor, *, n: int, f: int) -> torch.Tensor:
        distances = torch.cdist(stacked, stacked, p=2.0)
        distances.fill_diagonal_(float("inf"))
        sorted_distances, _ = torch.sort(distances, dim=1)
        return sorted_distances[:, : n - f - 1].sum(dim=1)
