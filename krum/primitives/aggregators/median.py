"""Coordinate-wise median aggregation rule.

Reference:
    Dong Yin, Yudong Chen, Kannan Ramchandran, and Peter Bartlett.
    "Byzantine-Robust Distributed Learning: Towards Optimal Statistical
    Rates." In Proceedings of the 35th International Conference on
    Machine Learning (ICML 2018).
"""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, long, median, stack

from . import Aggregator


class Median(Aggregator):
    """Coordinate-wise median of the worker gradients.

    The aggregated gradient is the coordinate-wise median of the worker
    gradients. This provides basic Byzantine resilience: a single
    adversarial worker can shift at most one sample per coordinate away
    from the true median.
    """

    @classmethod
    def aggregate(
        cls,
        gradients: Sequence[Tensor] | Tensor,
        /,
        out: Tensor | None = None,
        **specialized: Any,
    ) -> Tensor:
        """Aggregate the gradients by computing the coordinate-wise median.

        Args:
            gradients: Sequence of 1-D tensors containing gradients from workers.
            out: Optional pre-allocated tensor to write the result into.
            **specialized: Additional keyword arguments.

        Returns:
            Coordinate-wise median of the gradients, of shape ``(d,)``.
        """
        if not isinstance(gradients, Tensor):
            gradients = stack(list(gradients))

        if out is not None:
            indices = out.new_empty(out.shape, dtype=long)
            median(gradients, dim=0, out=(out, indices))
            return out
        return gradients.median(dim=0).values
