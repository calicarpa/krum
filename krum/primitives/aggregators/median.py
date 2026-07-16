"""Median aggregation rule, coordinate-wise.

Reference:
    Dong Yin, Yudong Chen, Kannan Ramchandran, and Peter Bartlett.
    "Byzantine-Robust Distributed Learning: Towards Optimal Statistical
    Rates." In Proceedings of the 35th International Conference on
    Machine Learning (ICML 2018).
"""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, quantile, stack

from . import Aggregator


class Median(Aggregator):
    """Median aggregation rule, coordinate-wise.

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
        """Aggregate the gradients.

        Args:
            gradients: Sequence of 1-D tensors containing gradients from workers.
            out: Optional pre-allocated tensor to write the result into.
            **specialized: Additional keyword arguments.

        Returns:
            Coordinate-wise median of the gradients, of shape ``(d,)``.
        """
        if not isinstance(gradients, Tensor):
            gradients = stack(list(gradients))

        # Yin et al. define this as "the usual (one-dimensional) median," which
        # by standard convention averages the two middle values for an even
        # worker count. torch.median instead always returns one of the actual
        # submitted values (the lower of the two middles when even) — quantile
        # at q=0.5 is the variant that matches the paper's definition exactly,
        # and reduces to the same value as torch.median whenever the count is odd.
        return quantile(gradients, 0.5, dim=0, out=out)
