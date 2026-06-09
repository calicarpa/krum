"""Coordinate-wise trimmed mean aggregation rule.

Reference:
    Dong Yin, Yudong Chen, Kannan Ramchandran, and Peter Bartlett.
    "Byzantine-Robust Distributed Learning: Towards Optimal Statistical
    Rates." In Proceedings of the 35th International Conference on
    Machine Learning (ICML 2018).
"""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, mean, stack

from . import Aggregator


class TrimmedMean(Aggregator):
    r"""Coordinate-wise trimmed mean aggregation rule.

    For every coordinate, the :math:`f` smallest and :math:`f` largest values are
    dropped, then the remaining values are averaged. This requires at least
    :math:`2f + 1` workers and provides basic Byzantine resilience: adversarial
    workers can only shift at most :math:`f` samples per coordinate.
    """

    @classmethod
    def aggregate(
        cls,
        gradients: Sequence[Tensor] | Tensor,
        /,
        out: Tensor | None = None,
        *,
        f: int,
        **specialized: Any,
    ) -> Tensor:
        r"""Aggregate the gradients.

        Args:
            gradients: Sequence of 1-D tensors containing gradients from workers.
            out: Optional pre-allocated tensor to write the result into.
            f: Number of Byzantine workers to tolerate. Must satisfy
                :math:`0 \le f` and ``len(gradients) > 2f``.
            **specialized: Additional keyword arguments.

        Returns:
            Coordinate-wise trimmed mean of the gradients, of shape ``(d,)``.

        Raises:
            ValueError: If :math:`f` is negative or if there are not enough gradients
                to trim (``len(gradients) <= 2f``).
        """
        if f < 0:
            raise ValueError(f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 0 ≤ f")

        if not isinstance(gradients, Tensor):
            gradients = stack(list(gradients))

        if gradients.size(0) <= 2 * f:
            raise ValueError(f"At least 2f+1 = {2 * f + 1} gradients required, got {gradients.size(0)}")

        sorted_values = gradients.sort(dim=0).values
        trimmed = sorted_values[f : gradients.size(0) - f]
        return mean(trimmed, dim=0, out=out)
