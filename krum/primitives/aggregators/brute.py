"""Brute: most-clumped-subset aggregation rule (El Mhamdi et al., ICML 2018)."""

from collections.abc import Sequence
from itertools import combinations
from typing import Any

from torch import Tensor, cdist, mean, stack

from . import Aggregator


class Brute(Aggregator):
    r"""Brute aggregation rule — pick the most-clumped :math:`n - f` subset.

    For every subset :math:`R` of size :math:`n - f` from the submitted
    gradients, define its clumping score as
    :math:`\max_{i, j \in R} \|V_i - V_j\|^2`. The aggregator picks the
    subset with the smallest clumping score and returns its mean. This
    enumerates :math:`\binom{n}{n-f}` subsets, so it is only feasible when
    that count is small (the paper uses :math:`6` honest + :math:`5`
    Byzantine workers, giving :math:`\binom{11}{6} = 462` subsets).

    Reference:
        El Mahdi El Mhamdi, Rachid Guerraoui, and Sébastien Rouault.
        "The Hidden Vulnerability of Distributed Learning in Byzantium."
        ICML 2018.

    Args:
        gradients: Sequence of 1-D tensors, one per worker.
        n: Total number of workers. Must satisfy :math:`n \geq 2f + 1`.
        f: Number of Byzantine workers to tolerate. Must satisfy
            :math:`1 \leq f \leq (n - 1) // 2`.
        out: Optional pre-allocated tensor to write the result into.

    Returns:
        Mean of the most-clumped :math:`n - f` subset, of shape ``(d,)``.

    Raises:
        ValueError: If ``n``, ``f``, or the gradients count is invalid.
    """

    @classmethod
    def aggregate(
        cls,
        gradients: Sequence[Tensor] | Tensor,
        /,
        out: Tensor | None = None,
        *,
        n: int,
        f: int,
        **specialized: Any,
    ) -> Tensor:
        """Aggregate gradients by selecting the most-clumped :math:`n - f` subset.

        Args:
            gradients: Sequence of 1-D tensors containing gradients from workers.
            out: Optional pre-allocated tensor to write the result into.
            n: Total number of workers.
            f: Number of Byzantine workers to tolerate.
            **specialized: Additional keyword arguments.

        Returns:
            Mean of the selected :math:`n - f` subset, of shape ``(d,)``.

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
        if f < 1 or n < 2 * f + 1:
            raise ValueError(
                f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 1 ≤ f ≤ {(n - 1) // 2}"
            )

        if not isinstance(gradients, Tensor):
            gradients = stack(list(gradients))

        if gradients.size(0) != n:
            raise ValueError(f"Expected {n} gradients, got {gradients.size(0)}")

        k = n - f
        best_subset: tuple[int, ...] | None = None
        best_score = float("inf")
        for subset in combinations(range(n), k):
            sub = gradients[list(subset)]
            diam = cdist(sub, sub, p=2.0).max().item()
            if diam < best_score:
                best_score = diam
                best_subset = subset
        assert best_subset is not None
        return mean(gradients[list(best_subset)], dim=0, out=out)
