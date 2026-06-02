"""Coordinate-wise trimmed mean aggregation rule."""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, mean, stack

from . import Aggregator


class TrimmedMean(Aggregator):
    """Coordinate-wise trimmed mean aggregation rule.

    For every coordinate, the ``f`` smallest and ``f`` largest values are
    dropped, then the remaining values are averaged. This requires at least
    ``2f + 1`` workers and provides basic Byzantine resilience: adversarial
    workers can only shift at most ``f`` samples per coordinate.

    Reference:
        Yin, Dong, Yudong Chen, Kannan Ramchandran, and Peter Bartlett.
        "Byzantine-Robust Distributed Learning: Towards Optimal Statistical
        Rates." In Proceedings of the 35th International Conference on
        Machine Learning (ICML 2018).

    Args:
        gradients: Sequence of 1-D tensors, one per worker.
        f: Number of Byzantine workers to tolerate. Must satisfy
            ``0 <= f`` and ``len(gradients) > 2f``.
        out: Optional pre-allocated tensor to write the result into.

    Returns:
        Coordinate-wise trimmed mean of the gradients, of shape ``(d,)``.

    Raises:
        ValueError: If ``f`` is negative or if there are not enough gradients
            to trim (``len(gradients) <= 2f``).
    """

    @classmethod
    def aggregate(
        cls,
        gradients: Sequence[Tensor],
        /,
        out: Tensor | None = None,
        *,
        f: int,
        **specialized: Any,
    ) -> Tensor:
        """Aggregate the gradients by computing the coordinate-wise trimmed mean.

        Args:
            gradients: Sequence of 1-D tensors containing gradients from workers.
            out: Optional pre-allocated tensor to write the result into.
            f: Number of Byzantine workers to tolerate. Must satisfy
                ``0 <= f`` and ``len(gradients) > 2f``.
            **specialized: Additional keyword arguments.

        Returns:
            Coordinate-wise trimmed mean of the gradients, of shape ``(d,)``.

        Raises:
            ValueError: If ``f`` is negative or if there are not enough gradients
                to trim (``len(gradients) <= 2f``).
        """
        if f < 0:
            raise ValueError(f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 0 ≤ f")

        grad_list = list(gradients)
        num_grads = len(grad_list)

        if num_grads <= 2 * f:
            raise ValueError(f"At least 2f+1 = {2 * f + 1} gradients required, got {num_grads}")

        stacked = stack(grad_list)
        sorted_values = stacked.sort(dim=0).values
        trimmed = sorted_values[f : num_grads - f]
        return mean(trimmed, dim=0, out=out)
