"""Bulyan aggregator from The Hidden Vulnerability of Distributed Learning in Byzantium."""

import numpy as np
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
        """
        self.m = m if m is not None else n - f - 2
        super().__init__(n=n, f=f)
        self.check()

    def check(self) -> None:
        """Check parameter validity for Bulyan rule.

        Raises:
            ValueError: If parameters are invalid for the Bulyan rule.
        """
        super().check()
        if self.f < 1 or self.n < 4 * self.f + 3:
            raise ValueError(
                f"Invalid number of Byzantine gradients to tolerate, got f = {self.f!r}, expected 1 ≤ f ≤ {(self.n - 3) // 4}"
            )
        if self.m < 1 or self.m > self.n - self.f - 2:
            raise ValueError(
                f"Invalid number of selected gradients, got m = {self.m!r}, expected 1 ≤ m ≤ {self.n - self.f - 2}"
            )

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

    def upper_bound(self) -> float:
        """Compute the theoretical upper bound on the ratio non-Byzantine standard deviation / norm.

        Returns:
            Theoretical upper bound.
        """
        return 1 / np.sqrt(
            2 * (self.n - self.f + self.f * (self.n + self.f * (self.n - self.f - 2) - 2) / (self.n - 2 * self.f - 2))
        )
