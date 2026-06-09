r"""A Little Is Enough (ALIE) gradient attack.

Reference:
    Gilad Baruch, Moriah Baruch, Yoav Goldberg, and Kfir Y. Levy. "A Little
    Is Enough: Circumventing Defenses For Distributed Learning." In
    Advances in Neural Information Processing Systems 32 (NeurIPS 2019).
"""

from collections.abc import Sequence
from enum import Enum
from typing import Any

from torch import Tensor, add, is_floating_point, mul, stack
from torch.distributions import Normal

from . import Attack


class Direction(str, Enum):
    r"""Sign of the perturbation applied by ALIE relative to the honest mean.

    Chooses whether the perturbation :math:`z \sigma` is added to (``POSITIVE``) or
    subtracted from (``NEGATIVE``) the honest coordinate-wise mean.
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"


class ALIEAttack(Attack):
    r"""ALIE-style attack using exact honest gradient statistics.

    Generates Byzantine gradients from the exact coordinate-wise mean and
    standard deviation of the honest gradients passed to the attack. The attack
    perturbs the honest mean by :math:`z \sigma` along the chosen :class:`Direction`,
    where :math:`z` is the attack factor.

    This corresponds to a statistics-oracle variant of ALIE rather than the
    original paper's more restricted information setting: the attacker is assumed
    to know the full honest gradient distribution, not just a subset.
    """

    @classmethod
    def generate(
        cls,
        honest_gradients: Sequence[Tensor] | Tensor,
        /,
        out: Tensor | None = None,
        *,
        f: int,
        z: float | str = "max",
        direction: Direction = Direction.NEGATIVE,
        **specialized: Any,
    ) -> Tensor:
        """Generate Byzantine gradients.

        Args:
            honest_gradients: Sequence of :math:`h` gradient vectors, one per honest
                worker, each of shape :math:`(d,)`.
            out: Optional pre-allocated tensor of shape :math:`(f, d)` to write the
                result into and return.
            f: Number of Byzantine gradients to generate.
            z: Attack factor in standard-deviation units, or ``"max"`` for the
                largest factor that keeps the Byzantine gradients within a
                majority of the honest gradient distribution.
            direction: Direction of the perturbation relative to the honest mean.
            **specialized: Additional keyword arguments.

        Returns:
            Byzantine gradients of shape :math:`(f, d)`. When :math:`f = 0`, returns an
            empty tensor of shape :math:`(0, d)`.

        Raises:
            TypeError: If :math:`z` is not a number or ``"max"``, ``direction`` is
                not a :class:`Direction`, or the honest gradients are not
                floating-point.
            ValueError: If :math:`z` is negative, :math:`f` is negative, there are no
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
        stacked = stack(list(honest_gradients))
        if not is_floating_point(stacked):
            raise TypeError("Expected honest gradients to use a floating-point dtype")

        if f == 0:
            return mul(stacked.new_empty((0, stacked.shape[1])), 1, out=out)

        z_value = cls.max_z(stacked, f) if z == "max" else stacked.new_tensor(z)

        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0, correction=0)
        perturbation = z_value * std
        # Tile ``mean ± perturbation`` to (f, d) through ``add``'s ``out=`` so the
        # buffer-reuse and wrong-shape (resize) behavior matches the aggregators.
        alpha = 1 if direction is Direction.POSITIVE else -1
        return add(mean.unsqueeze(0).expand(f, -1), perturbation, alpha=alpha, out=out)

    @staticmethod
    def max_z(honest_gradients: Tensor, f: int) -> Tensor:
        r"""Compute the maximal valid ALIE attack factor for the worker configuration.

        :math:`z_{\max}` is the largest :math:`z` such that
        :math:`\Phi(z) < (h - s) / h`, where :math:`h` is the number of honest
        workers and :math:`s` is the number of honest workers needed to form a
        majority among the :math:`n = h + f` workers:
        :math:`s = \lfloor n / 2 \rfloor + 1 - f`.

        Args:
            honest_gradients: Tensor of shape :math:`(h, d)` containing gradients
                from the :math:`h` honest workers.
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
        z_max = Normal(
            honest_gradients.new_tensor(0.0),
            honest_gradients.new_tensor(1.0),
        ).icdf(honest_gradients.new_tensor(ratio))

        if z_max < 0:
            msg = f"Invalid worker configuration for ALIE, got z_max = {float(z_max)!r}, expected z_max >= 0"
            raise ValueError(msg)
        return z_max
