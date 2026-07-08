"""AKSEL aggregation rule, median-pivot nearest-neighbor averaging.

Reference:
    Amine Boussetta, El-Mahdi El-Mhamdi, Rachid Guerraoui, Alexandre
    Maurer, and Sébastien Rouault. "AKSEL: Fast Byzantine SGD." In 24th
    International Conference on Principles of Distributed Systems
    (OPODIS 2020), Leibniz International Proceedings in Informatics,
    Volume 184, pp. 8:1--8:16. Schloss Dagstuhl - Leibniz-Zentrum
    für Informatik (2021).
"""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, mean, stack
from torch.linalg import vector_norm

from . import Aggregator


class Aksel(Aggregator):
    r"""AKSEL aggregation rule, median-pivot nearest-neighbor averaging.

    AKSEL computes the coordinate-wise median of the :math:`n` worker
    gradients as a robust pivot, then selects the :math:`n - f` gradients
    closest to this pivot (by Euclidean distance) and returns their mean.

    This achieves optimal time complexity :math:`\mathcal{O}(nd)`, an
    optimal breakdown point :math:`n > 2f`, and the lowest known upper
    bound on the expected angular error :math:`\mathcal{O}(\sqrt{d})`
    among full-gradient approaches.
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
            Mean of the :math:`n - f` gradients closest to the coordinate-wise
            median, of shape ``(d,)``.

        Raises:
            ValueError: If :math:`f` is negative or if there are not enough
                gradients (``len(gradients) <= 2f``).
        """
        if f < 0:
            raise ValueError(f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 0 ≤ f")

        if not isinstance(gradients, Tensor):
            gradients = stack(list(gradients))

        if gradients.size(0) <= 2 * f:
            raise ValueError(f"At least 2f+1 = {2 * f + 1} gradients required, got {gradients.size(0)}")

        pivot = gradients.median(dim=0).values
        distances = vector_norm(gradients - pivot, dim=1)
        num_to_keep = gradients.size(0) - f
        _, closest = distances.topk(num_to_keep, largest=False)
        return mean(gradients[closest], dim=0, out=out)
