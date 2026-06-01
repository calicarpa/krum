"""GeoMed: geometric median (vector-level medoid) aggregation rule (El Mhamdi et al., ICML 2018)."""

from collections.abc import Sequence
from typing import Any

import torch

from . import Aggregator


class GeoMed(Aggregator):
    r"""Geometric median (vector-level medoid) of the worker gradients.

    The geometric median is the gradient :math:`V_i` that minimises
    :math:`\\sum_j \\|V_i - V_j\\|`. Ties are broken by the smallest index.
    This is a vector-level operator (one of the submitted vectors is
    selected as-is) — distinct from the coordinate-wise median, which
    computes a median per coordinate.

    Reference:
        El Mahdi El Mhamdi, Rachid Guerraoui, and Sébastien Rouault.
        "The Hidden Vulnerability of Distributed Learning in Byzantium."
        ICML 2018.

    Args:
        gradients: Sequence of 1-D tensors, one per worker.
        n: Total number of workers. Must satisfy :math:`n \\geq 2f + 1`.
        f: Number of Byzantine workers to tolerate. Must satisfy
            :math:`0 \\leq f \\leq (n - 1) // 2`.

    Returns:
        Selected worker gradient of shape ``(d,)``.

    Raises:
        ValueError: If ``n``, ``f``, or the gradients count is invalid.
    """

    @classmethod
    def aggregate(cls, gradients: Sequence[torch.Tensor], /, *, n: int, f: int, **specialized: Any) -> torch.Tensor:
        """Aggregate gradients by selecting the geometric median.

        Args:
            gradients: Sequence of 1-D tensors containing gradients from workers.
            n: Total number of workers.
            f: Number of Byzantine workers to tolerate. ``f`` is accepted for
                API uniformity with other aggregators but is not consulted
                here (the geometric median is defined for any ``n ≥ 1``).
            **specialized: Additional keyword arguments.

        Returns:
            Selected worker gradient of shape ``(d,)``.

        Raises:
            ValueError: If ``n``, ``f``, or the gradients count is invalid.
        """
        if n < 1:
            raise ValueError(f"Expected a list of at least one gradient to aggregate, got {n!r}")
        if f < 0:
            raise ValueError(f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 0 ≤ f")
        if f > n:
            raise ValueError(
                f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected f ≤ n = {n!r}"
            )
        if len(gradients) != n:
            raise ValueError(f"Expected {n} gradients, got {len(gradients)}")
        stacked = torch.stack(list(gradients))
        distances = torch.cdist(stacked, stacked, p=2.0)
        scores = distances.sum(dim=1)
        best_index = int(scores.argmin().item())
        return stacked[best_index]
