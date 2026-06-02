"""A Little Is Enough (ALIE) gradient attack.

Reference:
    Baruch, Gilad, Moriah Baruch, Yoav Goldberg, and Kfir Y. Levy. "A Little
    Is Enough: Circumventing Defenses For Distributed Learning." In
    Advances in Neural Information Processing Systems 32 (NeurIPS 2019).
"""

import warnings
from collections.abc import Sequence
from enum import Enum
from typing import Any

import torch

from . import Attack


class Direction(str, Enum):
    """Sign of the perturbation applied by ALIE relative to the honest mean.

    Chooses whether the perturbation ``z * std`` is added to (``POSITIVE``) or
    subtracted from (``NEGATIVE``) the honest coordinate-wise mean.
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"


class ALIEAttack(Attack):
    """ALIE-style attack using exact honest gradient statistics.

    Generates Byzantine gradients from the exact coordinate-wise mean and
    standard deviation of the honest gradients passed to the attack. The attack
    perturbs the honest mean by ``z * std`` along the chosen :class:`Direction`,
    where ``z`` is the attack factor.

    This corresponds to a statistics-oracle variant of ALIE rather than the
    original paper's more restricted information setting: the attacker is assumed
    to know the full honest gradient distribution, not just a subset.

    Args:
        honest_gradients: Sequence of 1-D tensors, one per honest worker.
        f: Number of Byzantine gradients to generate.
        z: Attack factor in standard-deviation units. Use the string ``"max"``
            to compute the largest factor ``z_max`` that keeps the Byzantine
            gradients within a majority of the honest gradient distribution. A
            :class:`RuntimeWarning` is emitted if a numeric ``z`` exceeds
            ``z_max``.
        direction: Direction of the perturbation relative to the honest mean.
            Defaults to :attr:`Direction.NEGATIVE`.

    Returns:
        Byzantine gradients of shape ``(f, d)``.

    Raises:
        TypeError: If ``z`` is not a number or the string ``"max"``, if
            ``direction`` is not a :class:`Direction`, or if the honest gradients
            do not use a floating-point dtype.
        ValueError: If ``z`` is negative, ``f`` is negative, there are no honest
            gradients, or the worker configuration admits no non-negative ALIE
            factor.
    """

    @classmethod
    def generate(
        cls,
        honest_gradients: Sequence[torch.Tensor],
        /,
        *,
        f: int,
        z: float | str = "max",
        direction: Direction = Direction.NEGATIVE,
        **specialized: Any,
    ) -> torch.Tensor:
        """Generate Byzantine gradients using ALIE-style statistics.

        Args:
            honest_gradients: Sequence of ``h`` gradient vectors, one per honest
                worker, each of shape ``(d,)``.
            f: Number of Byzantine gradients to generate.
            z: Attack factor in standard-deviation units, or ``"max"`` for the
                largest factor that keeps the Byzantine gradients within a
                majority of the honest gradient distribution.
            direction: Direction of the perturbation relative to the honest mean.
            **specialized: Additional keyword arguments.

        Returns:
            Byzantine gradients of shape ``(f, d)``. When ``f == 0``, returns an
            empty tensor of shape ``(0, d)``.

        Raises:
            TypeError: If ``z`` is not a number or ``"max"``, ``direction`` is
                not a :class:`Direction`, or the honest gradients are not
                floating-point.
            ValueError: If ``z`` is negative, ``f`` is negative, there are no
                honest gradients, or the worker configuration admits no
                non-negative ALIE factor.
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
        if f < 0:
            msg = f"Invalid number of Byzantine gradients to generate, got {f!r}, expected 0 <= f"
            raise ValueError(msg)
        if len(honest_gradients) == 0:
            raise ValueError("Expected at least one honest gradient to compute ALIE statistics")
        stacked = torch.stack(list(honest_gradients))
        if not torch.is_floating_point(stacked):
            raise TypeError("Expected honest gradients to use a floating-point dtype")

        if f == 0:
            return stacked.new_empty((0, stacked.shape[1]))

        z_max = cls._max_z(stacked, f)
        z_value = z_max if z == "max" else stacked.new_tensor(z)
        if z_value > z_max:
            warnings.warn(
                f"ALIE attack factor z = {float(z_value)!r} is greater than z_max = {float(z_max)!r}; "
                "the generated gradients may be easy to distinguish from honest gradients.",
                RuntimeWarning,
                stacklevel=2,
            )

        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0, correction=0)
        perturbation = z_value * std
        malicious_gradient = mean + perturbation if direction is Direction.POSITIVE else mean - perturbation

        return malicious_gradient.repeat(f, 1)

    @staticmethod
    def _max_z(honest_gradients: torch.Tensor, f: int) -> torch.Tensor:
        """Compute the maximal valid ALIE attack factor for the worker configuration.

        ``z_max`` is the largest ``z`` such that ``Phi(z) < (h - s) / h``,
        where ``h`` is the number of honest workers and ``s`` is the number of
        honest workers needed to form a majority among the ``n = h + f``
        workers: ``s = floor(n / 2) + 1 - f``.

        Args:
            honest_gradients: Tensor of shape ``(h, d)`` containing gradients
                from the ``h`` honest workers.
            f: Number of Byzantine gradients to generate.

        Returns:
            Maximal attack factor, as a 0-D tensor on the same device and
            dtype as ``honest_gradients``.

        Raises:
            ValueError: If there are no honest gradients, or if the worker
                configuration does not admit a non-negative ALIE factor.
        """
        num_honest = honest_gradients.shape[0]
        if num_honest == 0:
            msg = "Expected at least one honest gradient to compute ALIE statistics"
            raise ValueError(msg)
        num_workers = num_honest + f
        # s = floor(n / 2) + 1 - f: honest workers needed for a majority of n = h + f.
        num_supporters = num_workers // 2 + 1 - f
        ratio = (num_honest - num_supporters) / num_honest
        if ratio >= 1:
            msg = f"Invalid worker configuration for ALIE, got normal CDF target = {ratio!r}, expected target < 1"
            raise ValueError(msg)
        z_max = torch.distributions.Normal(
            honest_gradients.new_tensor(0.0),
            honest_gradients.new_tensor(1.0),
        ).icdf(honest_gradients.new_tensor(ratio))

        if z_max < 0:
            msg = f"Invalid worker configuration for ALIE, got z_max = {float(z_max)!r}, expected z_max >= 0"
            raise ValueError(msg)
        return z_max
