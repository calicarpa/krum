"""Krum aggregation rule, single-gradient selection.

Reference:
    Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien
    Stainer. "Machine learning with adversaries: Byzantine tolerant
    gradient descent." In Advances in Neural Information Processing
    Systems 30 (NIPS 2017).
"""

from collections.abc import Sequence
from typing import Any

from torch import Tensor

from .multikrum import MultiKrum


class Krum(MultiKrum):
    r"""Krum aggregation rule, single-gradient selection.

    For each worker gradient, Krum scores it by the sum of its distances to
    its :math:`n - f - 2` closest neighbors, and returns the gradient with the
    smallest score — the one most consistent with the other honest workers.
    This is :class:`MultiKrum` with :math:`m = 1`.
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
            n: Total number of workers.
            f: Number of Byzantine workers to tolerate.
            **specialized: Additional keyword arguments.

        Returns:
            Aggregated gradient of shape ``(d,)``.

        Raises:
            ValueError: If :math:`n < 1`, :math:`f < 0`, :math:`f > n`, :math:`n < 2f + 3`,
                or ``len(gradients) != n``.
        """
        return MultiKrum.aggregate(gradients, out=out, n=n, f=f, m=1)
