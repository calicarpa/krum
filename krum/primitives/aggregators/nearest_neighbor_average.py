"""Nearest-neighbor averaging aggregator."""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, argsort, mean, stack
from torch.linalg import vector_norm

from . import Aggregator


class NearestNeighborAverage(Aggregator):
    """Average the ``num_closest`` vectors nearest to a call-specific pivot.

    The rule keeps the ``num_closest`` vectors with smallest Euclidean distance
    to the pivot, then returns their mean. Both ``num_closest`` and the pivot
    belong to a single aggregation call, so they are passed to :meth:`aggregate`
    rather than stored on the (stateless) aggregator.

    How many vectors to keep is a caller policy (e.g. ``n - f`` for plain
    nearest-neighbor averaging, ``n - 2f`` for MoNNA-style model mixing); this
    rule only needs the resulting count, not ``n`` or ``f``.
    """

    @classmethod
    def aggregate(
        cls,
        gradients: Sequence[Tensor] | Tensor,
        /,
        out: Tensor | None = None,
        *,
        num_closest: int,
        pivot: Tensor,
        **specialized: Any,
    ) -> Tensor:
        """Average the ``num_closest`` vectors nearest to the pivot.

        Args:
            gradients: Sequence of ``m`` candidate vectors, each of shape ``(d,)``.
            out: Optional pre-allocated tensor to write the result into.
            num_closest: Number of nearest vectors to average.
            pivot: Tensor of shape ``(d,)`` used as the distance reference.
            **specialized: Additional keyword arguments.

        Returns:
            Mean of the ``num_closest`` closest vectors, shape ``(d,)``.

        Raises:
            ValueError: If ``num_closest`` is not positive, fewer than
                ``num_closest`` candidates are supplied, or the pivot shape is
                wrong.
        """
        if num_closest < 1:
            raise ValueError(f"Expected num_closest to be at least 1, got {num_closest!r}")
        stacked = stack(list(gradients))
        if stacked.shape[0] < num_closest:
            raise ValueError(f"Expected at least num_closest={num_closest} gradients, got {stacked.shape[0]!r}")
        if pivot.shape != stacked.shape[1:]:
            raise ValueError(f"Expected pivot with shape {tuple(stacked.shape[1:])!r}, got {tuple(pivot.shape)!r}")

        distances = vector_norm(stacked - pivot, dim=1)
        closest = argsort(distances, stable=True)[:num_closest]
        return mean(stacked[closest], 0, out=out)
