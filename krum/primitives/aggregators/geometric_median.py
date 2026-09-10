"""GeometricMedian aggregation rule, geometric median via smoothed Weiszfeld iterations.

Reference:
    Krishna Pillutla, Sham M. Kakade, and Zaid Harchaoui.
    "Robust Aggregation for Federated Learning."
    IEEE Transactions on Signal Processing 70 (2022): 1142-1154.
"""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, clamp, mean, stack
from torch.linalg import vector_norm

from . import Aggregator


class GeometricMedian(Aggregator):
    r"""GeometricMedian aggregation rule, geometric median of the gradients.

    The geometric median is the unconstrained minimiser
    :math:`\arg\min_{y \in \mathbb{R}^d} \sum_i \|y - V_i\|`, computed here
    with smoothed Weiszfeld iterations (RFA oracle): starting from the mean,
    each step reweights the gradients by the inverse of their distance to
    the current estimate, floored at :math:`\nu` for numerical stability.
    Unlike :class:`~krum.primitives.aggregators.medoid.Medoid`, the result is a synthetic point and generally
    not one of the submitted vectors.
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
        nu: float = 0.1,
        tol: float = 1e-6,
        max_iter: int = 100,
        **specialized: Any,
    ) -> Tensor:
        r"""Aggregate gradients by approximating their geometric median.

        Args:
            gradients: Sequence of 1-D tensors containing gradients from workers.
            out: Optional pre-allocated tensor to write the result into.
            n: Total number of workers.
            f: Number of Byzantine workers to tolerate. :math:`f` is accepted for
                API uniformity with other aggregators but is not consulted
                here (the geometric median is defined for any :math:`n \ge 1`
                and tolerates any minority of outliers).
            nu: Smoothing floor on distances. Must be positive.
            tol: Stop when an iteration moves the estimate by at most ``tol``.
            max_iter: Maximum number of Weiszfeld iterations. Must be at least 1.
            **specialized: Additional keyword arguments.

        Returns:
            Approximate geometric median of shape ``(d,)``.

        Raises:
            ValueError: If :math:`n`, :math:`f`, the gradients count, ``nu``,
                ``tol`` or ``max_iter`` is invalid.
        """
        if n < 1:
            raise ValueError(f"Expected a list of at least one gradient to aggregate, got {n!r}")
        if f < 0:
            raise ValueError(f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 0 ≤ f")
        if f > n:
            raise ValueError(
                f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected f ≤ n = {n!r}"
            )
        if nu <= 0:
            raise ValueError(f"Expected smoothing nu to be positive, got {nu!r}")
        if tol < 0:
            raise ValueError(f"Expected tolerance to be non-negative, got {tol!r}")
        if max_iter < 1:
            raise ValueError(f"Expected at least one Weiszfeld iteration, got {max_iter!r}")

        if not isinstance(gradients, Tensor):
            gradients = stack(list(gradients))

        if gradients.size(0) != n:
            raise ValueError(f"Expected {n} gradients, got {gradients.size(0)}")

        estimate = mean(gradients, dim=0)
        for _ in range(max_iter):
            distances = vector_norm(gradients - estimate, dim=1)
            weights = 1.0 / clamp(distances, min=nu)
            updated = (weights.unsqueeze(1) * gradients).sum(dim=0) / weights.sum()
            if vector_norm(updated - estimate).item() <= tol:
                estimate = updated
                break
            estimate = updated

        if out is not None:
            return out.copy_(estimate)
        return estimate
