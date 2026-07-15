"""Bulyan aggregation rule, two-stage Krum + trimmed mean.

Reference:
    El Mahdi El Mhamdi, Rachid Guerraoui, and Sébastien Rouault. "The
    Hidden Vulnerability of Distributed Learning in Byzantium." In
    Proceedings of the 35th International Conference on Machine
    Learning (ICML 2018).
"""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, bool, cdist, mean, ones, sort, stack, topk

from . import Aggregator


class Bulyan(Aggregator):
    r"""Bulyan aggregation rule, two-stage Krum + trimmed mean.

    Bulyan first iteratively applies a Krum-style scoring rule to
    select a set S of θ = n − 2f gradients (one per iteration — the
    gradient with the lowest Krum score among the still-unselected
    gradients, then removed from the candidate pool). It then
    aggregates that set coordinate-wise by taking the median over S
    and averaging the β = θ − 2f = n − 4f closest values to the
    median per coordinate.

    The paper studies ``Bulyan(A)`` with ``A = Krum`` (``Bulyan(Krum)``)
    in all figures; this implementation follows that choice. A
    different base rule A could be plugged in by overriding
    :meth:`_select_one`.
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
            m: Number of gradients selected by MultiKrum at each iteration.
                Defaults to :math:`n - f - 2`.
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
        m = m if m is not None else n - f - 2
        if m < 1 or m > n - f - 2:
            raise ValueError(f"Invalid number of selected gradients, got m = {m!r}, expected 1 ≤ m ≤ {n - f - 2}")
        if f < 1 or n < 4 * f + 3:
            raise ValueError(
                f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 1 ≤ f ≤ {(n - 3) // 4}"
            )

        if not isinstance(gradients, Tensor):
            gradients = stack(list(gradients))

        if gradients.size(0) != n:
            raise ValueError(f"Expected {n} gradients, got {gradients.size(0)}")

        distances = cdist(gradients, gradients, p=2.0)
        valid_mask = ones(n, dtype=bool, device=gradients.device)
        selected = []

        theta = n - 2 * f

        for _ in range(theta):
            remaining = int(valid_mask.sum().item())
            k_nearest = min(n - f - 2, max(1, remaining - 1))
            m_cur = min(m, remaining)

            D = distances.clone()
            D[~valid_mask] = float("inf")
            D[:, ~valid_mask] = float("inf")
            D.fill_diagonal_(float("inf"))

            sorted_D, _ = sort(D, dim=1)
            scores = sorted_D[:, :k_nearest].sum(dim=1)
            scores[~valid_mask] = float("inf")

            _, top_nodes = topk(scores, m_cur, largest=False)
            selected.append(gradients[top_nodes].mean(dim=0))

            best_node = top_nodes[0]
            valid_mask[best_node] = False

        selected_tensor = stack(selected)

        bulyan_m = selected_tensor.size(0) - 2 * f
        median = selected_tensor.median(dim=0).values
        distances_to_median = (selected_tensor - median).abs()

        _, closests_indices = topk(distances_to_median, bulyan_m, dim=0, largest=False)

        return mean(selected_tensor.gather(0, closests_indices), dim=0, out=out)
