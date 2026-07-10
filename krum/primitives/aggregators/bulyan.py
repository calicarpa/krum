"""Bulyan aggregation rule, two-stage Multi-Krum + trimmed mean.

Reference:
    El Mahdi El Mhamdi, Rachid Guerraoui, and Sébastien Rouault. "The
    Hidden Vulnerability of Distributed Learning in Byzantium." In
    Proceedings of the 35th International Conference on Machine
    Learning (ICML 2018).
"""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, stack, topk

from . import Aggregator
from .multikrum import MultiKrum
from .trimmed_mean import TrimmedMean


class Bulyan(Aggregator):
    r"""Bulyan aggregation rule, two-stage Multi-Krum + trimmed mean.

    Bulyan first iteratively applies Multi-Krum to select a set
    :math:`S` of :math:`\theta = n - 2f - 2` aggregated vectors.
    At each iteration the Multi-Krum output (average of the :math:`m`
    gradients with smallest Krum scores) is added to :math:`S`, and
    the gradient closest to that output is removed from the candidate
    pool. It then aggregates :math:`S` coordinate-wise via
    :class:`TrimmedMean` with the same :math:`f`, keeping
    :math:`\beta = \theta - 2f = n - 4f - 2` values per coordinate.

    This implementation uses ``Bulyan(MultiKrum)`` — i.e. the base
    aggregator is Multi-Krum with :math:`m = n - f - 2` by default.
    With :math:`m = 1` it reduces to ``Bulyan(Krum)``.

    .. note::

        Krum scores are computed once on the full candidate set and
        removed gradients are masked with ``inf`` rather than
        recomputing pairwise distances at every iteration. This is an
        approximation of Algorithm 1 in the paper: a removed gradient
        still appears in the distance matrix of the remaining workers,
        so individual scores do not get updated after each removal.
        The selection order may therefore differ slightly from the
        paper, though the impact on the final trimmed mean is minimal
        when the honest majority forms a tight cluster.
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
        m: int | None = None,
        **specialized: Any,
    ) -> Tensor:
        r"""Aggregate the gradients.

        Args:
            gradients: Sequence of 1-D tensors containing gradients from workers.
            out: Optional pre-allocated tensor to write the result into.
            n: Total number of workers. Must satisfy :math:`n \ge 4f + 3`.
            f: Number of Byzantine workers to tolerate. Must satisfy
                ``1 <= f <= (n - 3) // 4``.
            m: Number of gradients selected by Multi-Krum at each iteration.
                Defaults to :math:`n - f - 2`.
            **specialized: Additional keyword arguments.

        Returns:
            Aggregated gradient of shape ``(d,)``.

        Raises:
            ValueError: If :math:`n`, :math:`f`, :math:`m`, or the gradients count is invalid.
        """
        if not isinstance(n, int):
            raise TypeError(f"Invalid total number of workers, got {n=!r}, expected a positive int")
        if not isinstance(f, int):
            raise TypeError(
                f"Invalid number of Byzantine gradients to tolerate, got {f=!r}, expected a non-negative int"
            )
        if m is not None and not isinstance(m, int):
            raise TypeError(f"Invalid number of selected gradients, got {m=!r}, expected a positive int")
        if n < 1:
            raise ValueError(f"Expected a list of at least one gradient to aggregate, got {n=!r}")
        if f < 0:
            raise ValueError(f"Invalid number of Byzantine gradients to tolerate, got {f=!r}, expected 0 ≤ f")
        if f > n:
            raise ValueError(f"Invalid number of Byzantine gradients to tolerate, got {f=!r}, expected f ≤ n = {n!r}")
        if f < 1 or n < 4 * f + 3:
            raise ValueError(
                f"Invalid number of Byzantine gradients to tolerate, got {f=!r}, expected 1 ≤ f ≤ {(n - 3) // 4}"
            )
        m = m if m is not None else n - f - 2
        if m < 1 or m > n - f - 2:
            raise ValueError(f"Invalid number of selected gradients, got {m=!r}, expected 1 ≤ m ≤ {n - f - 2}")

        if not isinstance(gradients, Tensor):
            gradients = stack(list(gradients))

        if gradients.size(0) != n:
            raise ValueError(f"Expected {n} gradients, got {gradients.size(0)}")

        scores = MultiKrum.score(gradients, n=n, f=f, m=m)

        theta = n - 2 * f - 2
        selected = gradients.new_empty((theta, gradients.size(1)))

        for i in range(theta):
            m_cur = min(m, n - f - 2 - i)
            _, top = topk(scores, m_cur, largest=False)
            selected[i] = gradients[top].mean(dim=0)
            closest = top[(gradients[top] - selected[i]).norm(dim=1).argmin()]
            scores[closest] = float("inf")

        return TrimmedMean.aggregate(selected, out=out, f=f)
