"""MultiKrum: multi-gradient averaging rule (Blanchard et al., NIPS 2017)."""

from collections.abc import Sequence
from typing import Any

import torch

from . import Aggregator


class MultiKrum(Aggregator):
    """MultiKrum aggregation rule.

    Scores every worker gradient by the sum of its distances to its
    ``n - f - 1`` closest neighbors, picks the ``m`` gradients with the
    smallest scores, and returns their mean. With ``m = 1`` it reduces to
    :class:`~krum.primitives.aggregators.krum.Krum`.

    Reference:
        Blanchard, Peva, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien
        Stainer. "Machine learning with adversaries: Byzantine tolerant
        gradient descent." In Advances in Neural Information Processing
        Systems 30 (NIPS 2017).

    Args:
        gradients: Sequence of 1-D tensors, one per worker.
        n: Total number of workers.
        f: Number of Byzantine workers to tolerate. Must satisfy
            ``1 <= f <= (n - 3) // 2``.
        m: Number of selected gradients to average.

    Returns:
        Aggregated gradient of shape ``(d,)``.

    Raises:
        ValueError: If ``n``, ``f``, ``m``, or the gradients count is invalid.
    """

    @classmethod
    def aggregate(
        cls, gradients: Sequence[torch.Tensor], /, *, n: int, f: int, m: int, **specialized: Any
    ) -> torch.Tensor:
        """Aggregate gradients using MultiKrum.

        Args:
            gradients: Sequence of 1-D tensors containing gradients from workers.
            n: Total number of workers.
            f: Number of Byzantine workers to tolerate. Must satisfy
                ``1 <= f <= (n - 3) // 2``.
            m: Number of selected gradients to average. Must satisfy
                ``1 <= m <= n - f - 2``.
            **specialized: Additional keyword arguments.

        Returns:
            Aggregated gradient of shape ``(d,)``.

        Raises:
            ValueError: If ``n``, ``f``, ``m``, or the gradients count is invalid.
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
        """Score every stacked gradient by its sum of distances to its ``n - f - 1`` closest peers.

        The ``n - f - 1`` closest distance sum approximates how surrounded a
        gradient is by the (presumed honest) majority; lower scores are
        better.

        Args:
            stacked: Tensor of shape ``(n, d)`` containing the stacked worker gradients.
            n: Total number of workers (rows of ``stacked``).
            f: Number of Byzantine workers to tolerate.

        Returns:
            Tensor of shape ``(n,)`` containing the Krum score of each worker.
        """
        distances = torch.cdist(stacked, stacked, p=2.0)
        distances.fill_diagonal_(float("inf"))
        sorted_distances, _ = torch.sort(distances, dim=1)
        return sorted_distances[:, : n - f - 1].sum(dim=1)
