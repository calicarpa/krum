"""Zero attack."""

import torch

from .attack import Attack


class ZeroAttack(Attack):
    """Zero attack.

    Generates Byzantine gradients filled with zero values.
    """

    def generate(
        self,
        honest_gradients: torch.Tensor,
        num_byzantine: int,
    ) -> torch.Tensor:
        """Generate Byzantine gradients filled with zero values.

        Args:
            honest_gradients: Tensor of shape (h, d) containing gradients from honest workers.
            num_byzantine: Number of Byzantine gradients to generate.

        Returns:
            Byzantine gradients of shape (num_byzantine, d) filled with zero values.
        """
        self.check(honest_gradients, num_byzantine)

        return honest_gradients.new_zeros((num_byzantine, honest_gradients.shape[1]))
