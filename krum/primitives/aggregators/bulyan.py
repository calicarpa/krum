"""Bulyan aggregator from The Hidden Vulnerability of Distributed Learning in Byzantium."""

import torch

from .aggregator import Aggregator


class Bulyan(Aggregator):
    """Bulyan aggregator from The Hidden Vulnerability of Distributed Learning in Byzantium.

    Args:
        n: Total number of workers.
        f: Number of Byzantine workers to tolerate.
        m: Number of selected gradients. Defaults to n - f - 2 if None.
    """

    def __init__(self, *, n: int, f: int, m: int | None = None):
        """Initialize the Bulyan aggregator.

        Args:
            n: Total number of workers.
            f: Number of Byzantine workers to tolerate.
            m: Number of selected gradients. Defaults to n - f - 2 if None.

        Raises:
            ValueError: If parameters are invalid.
        """
        if n < 1:
            raise ValueError(f"Expected a list of at least one gradient to aggregate, got {n!r}")
        if f < 0:
            raise ValueError(f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 0 ≤ f")
        if f > n:
            raise ValueError(
                f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected f ≤ n = {n!r}"
            )
        self.n = n
        self.f = f
        self.m = m if m is not None else n - f - 2
        if self.m < 1 or self.m > n - f - 2:
            raise ValueError(f"Invalid number of selected gradients, got m = {self.m!r}, expected 1 ≤ m ≤ {n - f - 2}")
        if f < 1 or n < 4 * f + 3:
            raise ValueError(
                f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 1 ≤ f ≤ {(n - 3) // 4}"
            )
        super().__init__()

    def aggregate(self, gradients: torch.Tensor) -> torch.Tensor:
        """Aggregate the gradients using the Bulyan algorithm.

        Args:
            gradients: Tensor of shape (n, d) containing gradients from workers.

        Returns:
            Aggregated gradient of shape (d,).
        """
        distances = torch.cdist(gradients, gradients, p=2.0)
        valid_mask = torch.ones(self.n, dtype=torch.bool, device=gradients.device)
        selected = []

        m_cur = self.m
        for i in range(self.n - 2 * self.f - 2):
            m_cur = min(self.m, self.n - self.f - 2 - i)

            D = distances.clone()
            D[~valid_mask] = float("inf")
            D[:, ~valid_mask] = float("inf")
            D.fill_diagonal_(float("inf"))

            sorted_D, _ = torch.sort(D, dim=1)
            scores = sorted_D[:, :m_cur].sum(dim=1)
            scores[~valid_mask] = float("inf")

            _, top_nodes = torch.topk(scores, m_cur, largest=False)
            selected.append(gradients[top_nodes].mean(dim=0))

            best_node = top_nodes[0]
            valid_mask[best_node] = False

        selected_tensor = torch.stack(selected)

        bulyan_m = selected_tensor.size(0) - 2 * self.f
        median = selected_tensor.median(dim=0).values
        distances_to_median = (selected_tensor - median).abs()

        _, closests_indices = torch.topk(distances_to_median, bulyan_m, dim=0, largest=False)

        return selected_tensor.gather(0, closests_indices).mean(dim=0)
