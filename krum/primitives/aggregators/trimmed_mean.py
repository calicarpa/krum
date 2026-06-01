"""Coordinate-wise trimmed mean aggregation rule."""

from collections.abc import Sequence

import torch

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

    Returns:
        Coordinate-wise trimmed mean of the gradients, of shape ``(d,)``.

    Raises:
        ValueError: If ``f`` is negative or if there are not enough gradients
            to trim (``len(gradients) <= 2f``).
    """

    @classmethod
    def aggregate(cls, gradients: Sequence[torch.Tensor], /, *, f: int) -> torch.Tensor:
        """Aggregate the gradients by computing the coordinate-wise trimmed mean.

        Args:
            gradients: Sequence of 1-D tensors containing gradients from workers.
            f: Number of Byzantine workers to tolerate. Must satisfy
                ``0 <= f`` and ``len(gradients) > 2f``.

        Returns:
            Coordinate-wise trimmed mean of the gradients, of shape ``(d,)``.

        Raises:
            ValueError: If ``f`` is negative or if there are not enough gradients
                to trim (``len(gradients) <= 2f``).
        """
        if f < 0:
            raise ValueError(f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 0 ≤ f")
        if len(gradients) <= 2 * f:
            raise ValueError(f"At least 2f+1 = {2 * f + 1} gradients required, got {len(gradients)}")
        stacked = torch.stack(list(gradients))
        return stacked.sort(dim=0).values[f:-f].mean(dim=0)
