"""MultiKrum aggregation rule, multi-gradient averaging.

Reference:
    Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien
    Stainer. "Machine learning with adversaries: Byzantine tolerant
    gradient descent." In Advances in Neural Information Processing
    Systems 30 (NIPS 2017).
"""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, cdist, mean, sort, stack, topk

from . import Aggregator


class MultiKrum(Aggregator):
    r"""MultiKrum aggregation rule, multi-gradient averaging.

    Scores every worker gradient by the sum of its distances to its
    :math:`n - f - 1` closest neighbors, picks the :math:`m` gradients with the
    smallest scores, and returns their mean. With :math:`m = 1` it reduces to
    :class:`~krum.primitives.aggregators.krum.Krum`.
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
        m: int,
        **specialized: Any,
    ) -> Tensor:
        r"""Aggregate the gradients.

        Args:
            gradients: Sequence of 1-D tensors containing gradients from workers.
            out: Optional pre-allocated tensor to write the result into.
            n: Total number of workers.
            f: Number of Byzantine workers to tolerate. Must satisfy
                ``1 <= f <= (n - 3) // 2``.
            m: Number of selected gradients to average. Must satisfy
                :math:`1 \le m \le n - f - 2`.
            **specialized: Additional keyword arguments.

        Returns:
            Aggregated gradient of shape ``(d,)``.

        Raises:
            ValueError: If :math:`n`, :math:`f`, :math:`m`, or the gradients count is invalid.
        """
        if n < 1:
            raise ValueError(f"Expected a list of at least one gradient to aggregate, got {n!r}")
        if f < 0:
            raise ValueError(f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 0 ≤ f")
        if f > n:
            raise ValueError(
                f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected f ≤ n = {n!r}"
            )
        if m < 1 or m > n - f - 2:
            raise ValueError(f"Invalid number of selected gradients, got m = {m!r}, expected 1 ≤ m ≤ {n - f - 2}")
        if n < 2 * f + 3:
            raise ValueError(
                f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 1 ≤ f ≤ {(n - 3) // 2}"
            )

        if not isinstance(gradients, Tensor):
            gradients = stack(list(gradients))

        if gradients.size(0) != n:
            raise ValueError(f"Expected {n} gradients, got {gradients.size(0)}")

        scores = cls._compute_scores(gradients, n=n, f=f)
        _, top_indices = topk(scores, m, largest=False)

        return mean(gradients[top_indices], dim=0, out=out)

    @staticmethod
    def _compute_scores(stacked: Tensor, *, n: int, f: int) -> Tensor:
        r"""Score every stacked gradient by its sum of distances to its :math:`n - f - 1` closest peers in the sorted distance table.

        After :func:`torch.sort` on each row, the first entry is the
        self-distance (zero), so the first :math:`n - f - 1` columns
        effectively include the self at distance 0 plus the
        :math:`n - f - 2` closest *other* workers — exactly the
        :math:`n - f - 2` non-self neighbors.
        The implementation sums Euclidean distances (not squared) but
        the ranking is preserved.

        The :math:`n - f - 1` closest-peers sum approximates how
        surrounded a gradient is by the (presumed honest) majority;
        lower scores are better.

        Args:
            stacked: Tensor of shape :math:`(n, d)` containing the stacked worker gradients.
            n: Total number of workers (rows of ``stacked``).
            f: Number of Byzantine workers to tolerate.

        Returns:
            Tensor of shape :math:`(n,)` containing the Krum score of each worker.
        """
        distances = cdist(stacked, stacked, p=2.0)
        sorted_distances, _ = sort(distances, dim=1)
        return sorted_distances[:, : n - f - 1].sum(dim=1)
