"""NaN attack."""

import torch

from .attack import Attack


class NaNAttack(Attack):
    """NaN attack.

    Generates Byzantine gradients filled with NaN values.
    """

    def generate(
        self,
        honest_gradients: torch.Tensor,
        num_byzantine: int,
    ) -> torch.Tensor:
        """Generate Byzantine gradients filled with NaN values.

        Args:
            honest_gradients: Tensor of shape (h, d) containing gradients from honest workers.
            num_byzantine: Number of Byzantine gradients to generate.

        Returns:
            Byzantine gradients of shape (num_byzantine, d) filled with NaN values.
        """
        self.check(honest_gradients, num_byzantine)

        return torch.full(
            (num_byzantine, honest_gradients.shape[1]),
            torch.nan,
            dtype=honest_gradients.dtype,
            device=honest_gradients.device,
        )
