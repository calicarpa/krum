"""Bulyan aggregation rule, two-stage Krum + trimmed mean.

Reference:
    El Mahdi El Mhamdi, Rachid Guerraoui, and Sébastien Rouault. "The
    Hidden Vulnerability of Distributed Learning in Byzantium." In
    Proceedings of the 35th International Conference on Machine
    Learning (ICML 2018).
"""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, mean, stack, topk

from . import Aggregator
from .multikrum import MultiKrum


class Bulyan(Aggregator):
    r"""Bulyan aggregation rule, two-stage Krum + trimmed mean.

    This implementation follows ``Bulyan(Krum)`` from El Mhamdi et al.
    It first builds a selection set :math:`S` of
    :math:`\theta = n - 2f` gradients. At each iteration, Krum is applied
    to the remaining candidate gradients, the selected gradient is added
    to :math:`S`, and that gradient is removed from the candidate pool.
    It then aggregates :math:`S` coordinate-wise by taking the median and
    averaging the :math:`\beta = \theta - 2f = n - 4f` values closest to
    the median for each coordinate.
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
        r"""Aggregate the gradients.

        Args:
            gradients: Sequence of 1-D tensors containing gradients from workers.
            out: Optional pre-allocated tensor to write the result into.
            n: Total number of workers. Must satisfy :math:`n \ge 4f + 3`.
            f: Number of Byzantine workers to tolerate. Must satisfy
                ``1 <= f <= (n - 3) // 4``.
            **specialized: Additional keyword arguments.

        Returns:
            Aggregated gradient of shape ``(d,)``.

        Raises:
            TypeError: If a non-paper ``m`` parameter is provided.
            ValueError: If :math:`n`, :math:`f`, or the gradients count is invalid.
        """
        if not isinstance(n, int):
            raise TypeError(f"Invalid total number of workers, got {n=!r}, expected a positive int")
        if not isinstance(f, int):
            raise TypeError(
                f"Invalid number of Byzantine gradients to tolerate, got {f=!r}, expected a non-negative int"
            )
        if "m" in specialized:
            raise TypeError("Bulyan(Krum) from the reference paper does not accept an m parameter")
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

        if not isinstance(gradients, Tensor):
            gradients = stack(list(gradients))

        if gradients.size(0) != n:
            raise ValueError(f"Expected {n} gradients, got {gradients.size(0)}")

        theta = n - 2 * f
        selected = gradients.new_empty((theta, gradients.size(1)))

        candidate_indices = list(range(n))
        for i in range(theta):
            candidates = gradients[candidate_indices]
            scores = MultiKrum.score(candidates, n=len(candidate_indices), f=f)
            winner_pos = int(scores.argmin().item())
            selected[i] = candidates[winner_pos]
            candidate_indices.pop(winner_pos)

        beta = theta - 2 * f  # n - 4f
        median = selected.median(dim=0).values
        dist_to_median = (selected - median).abs()
        _, closest = topk(dist_to_median, beta, dim=0, largest=False)

        return mean(selected.gather(0, closest), dim=0, out=out)
