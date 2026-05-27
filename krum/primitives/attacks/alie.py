"""ALIE-style attack using exact honest gradient statistics."""

import warnings

import torch

from .attack import Attack
from .types import Direction


class ALIEAttack(Attack):
    """ALIE-style attack using exact honest gradient statistics.

    Generates Byzantine gradients from the exact coordinate-wise mean and
    standard deviation of the honest gradients passed to the attack.

    Args:
        z: Attack factor. Use "max" to compute the maximal valid attack factor.
        direction: Direction of the perturbation.
    """

    def __init__(self, *, z: float | str = "max", direction: Direction = Direction.NEGATIVE) -> None:
        """Initialize the attack.

        Args:
            z: Attack factor. Use "max" to compute the maximal valid attack factor.
            direction: Direction of the perturbation.
        """
        if z != "max":
            if not isinstance(z, int | float):
                msg = f"Invalid attack factor, got {z!r}, expected z >= 0 or 'max'"
                raise TypeError(msg)
            if z < 0:
                msg = f"Invalid attack factor, got {z!r}, expected z >= 0 or 'max'"
                raise ValueError(msg)
        if not isinstance(direction, Direction):
            msg = f"Invalid perturbation direction, got {direction!r}, expected a Direction"
            raise TypeError(msg)
        self.z = z
        self.direction = direction

    def generate(
        self,
        honest_gradients: torch.Tensor,
        num_byzantine: int,
    ) -> torch.Tensor:
        """Generate Byzantine gradients using ALIE-style statistics.

        Args:
            honest_gradients: Tensor of shape (h, d) containing gradients from honest workers.
            num_byzantine: Number of Byzantine gradients to generate.

        Returns:
            Byzantine gradients of shape (num_byzantine, d).
        """
        if honest_gradients.ndim != 2:
            raise ValueError("Expected a 2D tensor of honest gradients")
        if not torch.is_floating_point(honest_gradients):
            raise TypeError("Expected honest gradients to use a floating-point dtype")
        if num_byzantine < 0:
            msg = (
                f"Invalid number of Byzantine gradients to generate, got {num_byzantine!r}, expected 0 <= num_byzantine"
            )
            raise ValueError(msg)

        if num_byzantine == 0:
            return honest_gradients.new_empty((0, honest_gradients.shape[1]))

        z_max = self._max_z(honest_gradients, num_byzantine)
        z = z_max if self.z == "max" else honest_gradients.new_tensor(self.z)
        if z > z_max:
            warnings.warn(
                f"ALIE attack factor z = {float(z)!r} is greater than z_max = {float(z_max)!r}; "
                "the generated gradients may be easy to distinguish from honest gradients.",
                RuntimeWarning,
                stacklevel=2,
            )

        mean = honest_gradients.mean(dim=0)
        std = honest_gradients.std(dim=0, correction=0)
        perturbation = z * std
        malicious_gradient = mean + perturbation if self.direction is Direction.POSITIVE else mean - perturbation

        return malicious_gradient.repeat(num_byzantine, 1)

    def _max_z(self, honest_gradients: torch.Tensor, num_byzantine: int) -> torch.Tensor:
        """Compute the maximal valid ALIE attack factor.

        Args:
            honest_gradients: Tensor of shape (h, d) containing gradients from honest workers.
            num_byzantine: Number of Byzantine gradients to generate.

        Returns:
            Maximal attack factor.

        Raises:
            ValueError: If the worker configuration does not allow a non-negative ALIE factor.
        """
        num_honest = honest_gradients.shape[0]
        if num_honest == 0:
            msg = "Expected at least one honest gradient to compute ALIE statistics"
            raise ValueError(msg)
        num_workers = num_honest + num_byzantine
        num_supporters = num_workers // 2 + 1 - num_byzantine
        ratio = (num_honest - num_supporters) / num_honest
        if ratio >= 1:
            msg = f"Invalid worker configuration for ALIE, got normal CDF target = {ratio!r}, expected target < 1"
            raise ValueError(msg)
        # z_max is the largest z such that Phi(z) < (h - s) / h,
        # where h is the number of honest workers and s is the number
        # of honest supporters needed to keep the attack inside the majority.
        z_max = torch.distributions.Normal(
            honest_gradients.new_tensor(0.0),
            honest_gradients.new_tensor(1.0),
        ).icdf(honest_gradients.new_tensor(ratio))

        if z_max < 0:
            msg = f"Invalid worker configuration for ALIE, got z_max = {float(z_max)!r}, expected z_max >= 0"
            raise ValueError(msg)
        return z_max
