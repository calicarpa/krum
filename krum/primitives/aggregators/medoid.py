"""Medoid aggregation rule, vector-level medoid.

Reference:
    Cong Xie, Oluwasanmi Koyejo, and Indranil Gupta.
    "Generalized Byzantine-tolerant SGD."
    arXiv preprint arXiv:1802.10116 (2018).
"""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, cdist, stack

from . import Aggregator


class Medoid(Aggregator):
    r"""Medoid aggregation rule, vector-level medoid.

    The medoid is the submitted gradient :math:`V_i` that minimises
    :math:`\sum_j \|V_i - V_j\|`. Ties are broken by the smallest index.
    This is a vector-level operator (one of the submitted vectors is
    selected as-is) — distinct from the coordinate-wise median, which
    computes a median per coordinate, and from the true geometric median
    (:class:`~krum.primitives.aggregators.geometric_median.GeometricMedian`),
    which is the unconstrained minimiser in
    :math:`\mathbb{R}^d`. The medoid carries no
    :math:`(\alpha, f)`-Byzantine-resilience guarantee; it is kept here
    as a literature baseline.
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
        r"""Aggregate gradients by selecting the medoid.

        Args:
            gradients: Sequence of 1-D tensors containing gradients from workers.
            out: Optional pre-allocated tensor to write the result into.
            n: Total number of workers.
            f: Number of Byzantine workers to tolerate. :math:`f` is accepted for
                API uniformity with other aggregators but is not consulted
                here (the medoid is defined for any :math:`n \ge 1`).
            **specialized: Additional keyword arguments.

        Returns:
            Selected worker gradient of shape ``(d,)``.

        Raises:
            ValueError: If :math:`n`, :math:`f`, or the gradients count is invalid.
        """
        if n < 1:
            raise ValueError(f"Expected a list of at least one gradient to aggregate, got {n!r}")
        if f < 0:
            raise ValueError(f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 0 ≤ f")
        if f > n:
            raise ValueError(
                f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected f ≤ n = {n!r}"
            )

        if not isinstance(gradients, Tensor):
            gradients = stack(list(gradients))

        if gradients.size(0) != n:
            raise ValueError(f"Expected {n} gradients, got {gradients.size(0)}")

        distances = cdist(gradients, gradients, p=2.0)
        scores = distances.sum(dim=1)
        best_index = int(scores.argmin().item())
        if out is not None:
            return out.copy_(gradients[best_index])
        return gradients[best_index].clone()
