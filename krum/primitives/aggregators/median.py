"""Coordinate-wise median aggregation rule."""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, long, median, stack

from . import Aggregator


class Median(Aggregator):
    """Coordinate-wise geometric median of the worker gradients.

    The aggregated gradient is the coordinate-wise median of the worker
    gradients. This provides basic Byzantine resilience: a single
    adversarial worker can shift at most one sample per coordinate away
    from the true median.

    Args:
        gradients: Sequence of 1-D tensors, one per worker.
        out: Optional pre-allocated tensor to write the result into.

    Returns:
        Coordinate-wise median of the gradients, of shape ``(d,)``.
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
        result = stack(list(gradients))
        if out is not None:
            indices = out.new_empty(out.shape, dtype=long)

            median(result, dim=0, out=(out, indices))
            return out
        return result.median(dim=0).values
