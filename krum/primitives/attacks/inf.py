"""Infinity attack."""

import torch

from .attack import Attack


class InfAttack(Attack):
    """Infinity attack.

    Generates Byzantine gradients filled with infinite values.

    Args:
        sign: Sign of the infinite values to generate. Must be "positive" or "negative".
    """

    def __init__(self, *, sign: str = "positive") -> None:
        """Initialize the attack.

        Args:
            sign: Sign of the infinite values to generate. Must be "positive" or "negative".
        """
        if sign not in {"positive", "negative"}:
            msg = f"Invalid infinity sign, got {sign!r}, expected 'positive' or 'negative'"
            raise ValueError(msg)
        self.sign = sign

    def generate(
        self,
        honest_gradients: torch.Tensor,
        num_byzantine: int,
    ) -> torch.Tensor:
        """Generate Byzantine gradients filled with infinite values.

        Args:
            honest_gradients: Tensor of shape (h, d) containing gradients from honest workers.
            num_byzantine: Number of Byzantine gradients to generate.

        Returns:
            Byzantine gradients of shape (num_byzantine, d) filled with infinite values.
        """
        self.check(honest_gradients, num_byzantine)

        return torch.full(
            (num_byzantine, honest_gradients.shape[1]),
            torch.inf if self.sign == "positive" else -torch.inf,
            dtype=honest_gradients.dtype,
            device=honest_gradients.device,
        )
