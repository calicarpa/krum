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

    Scores every worker gradient by the sum of squared Euclidean distances to
    its :math:`n - f - 2` closest peers, picks the :math:`m` gradients with the
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
        m: int | None = None,
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
                :math:`1 \le m \le n - f - 2`. If ``None``, defaults to :math:`n - f - 2`.
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
        if n < 1:
            raise ValueError(f"Expected a list of at least one gradient to aggregate, got {n=!r}")
        if f < 0:
            raise ValueError(f"Invalid number of Byzantine gradients to tolerate, got {f=!r}, expected 0 ≤ f")
        if f > n:
            raise ValueError(f"Invalid number of Byzantine gradients to tolerate, got {f=!r}, expected f ≤ n = {n!r}")
        if n < 2 * f + 3:
            raise ValueError(
                f"Invalid number of Byzantine gradients to tolerate, got {f=!r}, expected 1 ≤ f ≤ {(n - 3) // 2}"
            )
        if m is None:
            m = n - f - 2
        if m < 1 or m > n - f - 2:
            raise ValueError(f"Invalid number of selected gradients, got {m=!r}, expected 1 ≤ m ≤ {n - f - 2}")

        if not isinstance(gradients, Tensor):
            gradients = stack(list(gradients))

        if gradients.size(0) != n:
            raise ValueError(f"Expected {n} gradients, got {gradients.size(0)}")

        scores = cls.score(gradients, n=n, f=f, m=n - f - 2)
        _, top_indices = topk(scores, m, largest=False)

        return mean(gradients[top_indices], dim=0, out=out)

    @staticmethod
    def score(
        stacked: Tensor,
        *,
        n: int,
        f: int,
        m: int | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        r"""Score every stacked gradient by its sum of squared distances to its :math:`m` closest peers.

        After :func:`torch.sort` on each row, the self-distance is
        0 (set via :meth:`~torch.Tensor.fill_diagonal_`), so column 0
        is always the worker itself. Columns :math:`1` through
        :math:`m` give the :math:`m` closest *other* workers.

        When ``m`` is ``None`` it defaults to :math:`n - f - 2`,
        the standard Krum score from Blanchard et al.

        The :math:`m` closest-peers sum approximates how surrounded a
        gradient is by the (presumed honest) majority; lower scores
        are better.

        When ``valid_mask`` is provided, gradients with ``mask[i] = False``
        are treated as infinitely far from every other gradient (so they
        cannot win the top-``m`` selection).

        Args:
            stacked: Tensor of shape :math:`(n, d)` containing the stacked worker gradients.
            n: Total number of workers (rows of ``stacked``).
            f: Number of Byzantine workers to tolerate.
            m: Number of closest peers to consider. Defaults to :math:`n - f - 2`.
            valid_mask: Optional boolean tensor of shape :math:`(n,)``;
                ``False`` entries are excluded from selection.

        Returns:
            Tensor of shape :math:`(n,)` containing the Krum score of each worker.
        """
        if m is None:
            m = n - f - 2
        distances = cdist(stacked, stacked, p=2.0).square()
        if valid_mask is not None:
            distances[~valid_mask] = float("inf")
            distances[:, ~valid_mask] = float("inf")
        distances.fill_diagonal_(0.0)
        sorted_distances, _ = sort(distances, dim=1)
        return sorted_distances[:, 1 : m + 1].sum(dim=1)
