"""Plain mean (arithmetic average) aggregation rule — non-robust baseline.

Reference:
    McMahan, Brendan, Eamonn Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas.
    "Communication-Efficient Learning of Deep Networks from Decentralized Data."
    In Proceedings of the 20th International Conference on Artificial Intelligence
    and Statistics (AISTATS 2017).
"""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, mean, stack

from . import Aggregator


class Average(Aggregator):
    """Plain mean of all worker gradients (no Byzantine resilience).

    Included as a non-robust baseline. A single adversarial worker with an
    arbitrarily large gradient can drive the aggregated gradient arbitrarily
    far from the honest mean, so this rule has no Byzantine resilience
    guarantees.
    """

    @classmethod
    def aggregate(
        cls,
        gradients: Sequence[Tensor] | Tensor,
        /,
        out: Tensor | None = None,
        **specialized: Any,
    ) -> Tensor:
        """Aggregate the gradients by computing the element-wise mean.

        Args:
            gradients: Sequence of 1-D tensors containing gradients from workers,
                or a pre-stacked 2-D tensor of shape ``(n, d)``.
            out: Optional pre-allocated tensor to write the result into.
            **specialized: Additional keyword arguments.

        Returns:
            Element-wise mean of the gradients, of shape ``(d,)``.
        """
        if not isinstance(gradients, Tensor):
            gradients = stack(list(gradients))
        return mean(gradients, dim=0, out=out)
