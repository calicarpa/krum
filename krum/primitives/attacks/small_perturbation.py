r"""Small-perturbation attack (El Mhamdi, Guerraoui, Rouault — ICML 2018).

The attack exploits the curse of dimensionality: in ``d ≫ 1``, two honest
gradients naturally disagree by ``Θ(√d)`` in the ``ℓp`` norm even on a
"good" coordinate, so an attacker can shift one coordinate by ``Ω(√d)``
while still looking legitimate to a ``ℓp``-norm-based aggregator.

The attack builds a single malicious vector

.. math::

    B(γ) = \frac{1}{n - f} \sum_{i=1}^{n-f} V_i  +  γ \cdot E

from the ``n − f`` honest gradients :math:`V_1, \dots, V_{n-f}` and the
direction :math:`E \in \mathbb{R}^d`. It then performs a boundary search
for the largest ``γ = γ_m`` such that the target aggregator still
"selects" ``B(γ_m)`` — i.e. ``B(γ_m)`` is in the aggregator's input
subset. All ``f`` Byzantine workers send the same ``B(γ_m)`` vector.

Two directions ``E`` are supported, depending on the target norm
``p`` (Section 3.2 vs Section 3.3):

* **finite norm** ``p ≥ 1`` (Section 3.2): ``E = e_e`` is a one-hot
  vector at a single coordinate :math:`e \in \{1, \dots, d\}`. The
  default choice :math:`e = \arg\max_j \mathrm{std}(V_{\cdot,j})` is
  the coordinate with the largest honest variance, which maximises
  :math:`γ_m`.
* **infinite norm** ``p = ∞`` (Section 3.3): ``E = (1, \dots, 1)``, the
  all-ones vector. This is required because
  :math:`\lim_{p \to +\infty} \sqrt[p]{d} = 1`, so the finite-norm
  attack loses its bite as ``p`` grows. Modifying non-maximal coordinates
  does not substantially affect the infinite-norm distance to the honest
  gradients, so :math:`B(γ)` stays in the aggregator's selection set for
  :math:`γ = Θ(d)`.
"""

import math
from collections.abc import Sequence
from typing import Any

from torch import Tensor, allclose, argmax, cat, ones, stack, zeros

from krum.primitives.aggregators import Aggregator

from . import Attack


class SmallPerturbationAttack(Attack):
    r"""Small-perturbation attack.

    The attack accepts a target aggregator and a target norm, builds the
    direction :math:`E` accordingly, and performs a boundary search for the
    largest :math:`\gamma` such that the aggregator still "selects" :math:`B(\gamma)`.

    The aggregator is asked to "select" :math:`B(\gamma)` in the same way as the
    actual run: the test substitutes a placeholder for :math:`B(\gamma)` (the
    honest mean) and checks whether the aggregator's output changes. This
    works uniformly for any stateless aggregator class.
    """

    @classmethod
    def generate(
        cls,
        honest_gradients: Sequence[Tensor] | Tensor,
        /,
        out: Tensor | None = None,
        *,
        f: int,
        aggregator: type[Aggregator],
        n: int,
        p: int | float = 2,
        coordinate: int | str | None = None,
        aggregator_kwargs: dict[str, Any] | None = None,
        gamma_max: float = 1e6,
        gamma_init: float = 1.0,
        tol: float = 1e-3,
        **specialized: Any,
    ) -> Tensor:
        """Generate the small-perturbation Byzantine gradients.

        Args:
            honest_gradients: Sequence of ``h`` gradient vectors, one per honest
                worker, each of shape ``(d,)``. Must be a 2-D floating-point tensor.
            out: Optional pre-allocated tensor of shape ``(f, d)`` to write the
                result into and return.
            f: Number of Byzantine gradients to generate. Must equal the configured
                ``f`` (the attack is defined for a fixed worker configuration).
            aggregator: Target aggregator class.
            n: Total number of workers.
            p: Target norm.
            coordinate: Index of the poisoned coordinate.
            aggregator_kwargs: Extra keyword arguments for the aggregator.
            gamma_max: Upper bound on the search for :math:`γ_m`.
            gamma_init: Initial step used during the exponential search.
            tol: Tolerance of the binary refinement.
            **specialized: Additional keyword arguments.

        Returns:
            Byzantine gradients of shape ``(f, d)`` containing the same
            ``B(γ_m)`` vector repeated ``f`` times.

        Raises:
            ValueError: If the input shape is invalid or ``f`` does not match.
            TypeError: If the input is not floating-point.
        """
        if n < 1:
            raise ValueError(f"Invalid total workers, got n = {n!r}, expected n >= 1")
        if f < 1:
            raise ValueError(f"Invalid Byzantine workers, got f = {f!r}, expected f >= 1")
        if f > n:
            raise ValueError(f"Invalid Byzantine workers, got f = {f!r}, expected f <= n = {n!r}")
        if n < 2 * f + 1:
            raise ValueError(f"Invalid worker configuration, got (n={n}, f={f}); expected n >= 2f + 1")
        if p != 2 and p != math.inf:
            raise ValueError(f"Invalid target norm, got p = {p!r}, expected 2 or math.inf")
        if coordinate is not None and not isinstance(coordinate, int | str):
            raise TypeError(f"Invalid coordinate selector, got {coordinate!r}, expected int or str")
        if isinstance(coordinate, str) and coordinate not in {"max", "largest", "all"}:
            raise ValueError(f"Invalid coordinate selector string, got {coordinate!r}, expected 'max' or 'all'")
        if gamma_max <= 0 or gamma_init <= 0 or tol <= 0:
            raise ValueError("gamma_max, gamma_init, and tol must be positive")
        if coordinate == "all" and p != math.inf:
            raise ValueError("coordinate='all' is only valid with p=math.inf (Section 3.3)")
        if len(honest_gradients) == 0:
            raise ValueError("Expected at least one honest gradient")
        stacked = stack(list(honest_gradients))
        if stacked.shape[0] != n - f:
            raise ValueError(f"Expected {n - f} honest gradients, got {stacked.shape[0]}")

        aggregator_kwargs = aggregator_kwargs or {}
        device = stacked.device
        dtype = stacked.dtype
        d = stacked.shape[1]

        honest_mean = stacked.mean(dim=0)
        direction = cls._build_direction(stacked, d, device, dtype, p, coordinate)
        gamma_m = cls._find_gamma_max(stacked, honest_mean, direction, aggregator, n, f, aggregator_kwargs)
        b_gamma = honest_mean + gamma_m * direction
        result = b_gamma.unsqueeze(0).expand(f, -1).contiguous()
        if out is not None:
            return out.copy_(result)
        return result

    @staticmethod
    def _build_direction(
        honest_gradients: Tensor,
        d: int,
        device: Any,
        dtype: Any,
        p: int | float,
        coordinate: int | str | None,
    ) -> Tensor:
        r"""Build the attack direction :math:`E` for the configured norm.

        For ``p == math.inf`` (Section 3.3), :math:`E = (1, \dots, 1)`.
        Otherwise (Section 3.2, finite ``p ≥ 1``), :math:`E` is a one-hot
        vector at the configured coordinate; the default is the coordinate
        with the largest honest variance.

        Args:
            honest_gradients: Honest gradient stack of shape ``(n-f, d)``.
            d: Gradient dimension.
            device: Device of the output tensor.
            dtype: Dtype of the output tensor.
            p: Target norm.
            coordinate: Index of the poisoned coordinate.

        Returns:
            Direction tensor of shape ``(d,)``.
        """
        if p == math.inf or coordinate == "all":
            return ones(d, device=device, dtype=dtype)

        if coordinate is None or coordinate in {"max", "largest"}:
            std = honest_gradients.std(dim=0, correction=0)
            idx = int(argmax(std).item())
        else:
            idx = int(coordinate)

        if not 0 <= idx < d:
            raise ValueError(f"Coordinate {idx!r} out of range [0, {d})")

        direction = zeros(d, device=device, dtype=dtype)
        direction[idx] = 1.0
        return direction

    @staticmethod
    def _is_selected(
        honest_gradients: Tensor,
        b_gamma: Tensor,
        aggregator: type[Aggregator],
        n: int,
        f: int,
        aggregator_kwargs: dict[str, Any],
    ) -> bool:
        """Test whether the target aggregator "selects" ``B(γ)``.

        The test substitutes a placeholder for :math:`B(γ)` (the honest
        mean) and checks whether the aggregator's output changes. This is
        a uniform test that works for any stateless aggregator: it does
        not depend on the aggregator's internal selection rule.

        Args:
            honest_gradients: Honest gradient stack of shape ``(n-f, d)``.
            b_gamma: Candidate :math:`B(γ)` of shape ``(d,)``.
            aggregator: Target aggregator class.
            n: Total number of workers.
            f: Number of Byzantine workers.
            aggregator_kwargs: Extra keyword arguments for the aggregator.

        Returns:
            ``True`` if the aggregator's output is sensitive to :math:`B(γ)`.
        """
        honest_mean = honest_gradients.mean(dim=0)
        byz_with = b_gamma.unsqueeze(0).expand(f, -1)
        byz_without = honest_mean.unsqueeze(0).expand(f, -1)

        stacked_with = cat([honest_gradients, byz_with], dim=0)
        stacked_without = cat([honest_gradients, byz_without], dim=0)

        out_with = aggregator.aggregate(stacked_with, n=n, f=f, **aggregator_kwargs)
        out_without = aggregator.aggregate(stacked_without, n=n, f=f, **aggregator_kwargs)
        return not allclose(out_with, out_without, atol=1e-6, rtol=1e-5)

    @classmethod
    def _find_gamma_max(
        cls,
        honest_gradients: Tensor,
        honest_mean: Tensor,
        direction: Tensor,
        aggregator: type[Aggregator],
        n: int,
        f: int,
        aggregator_kwargs: dict[str, Any],
    ) -> float:
        """Find the largest :math:`γ` such that the aggregator selects :math:`B(γ)`.

        Uses an exponential search to bracket the boundary, then refines
        it with binary search up to ``tol``. Returns ``gamma_max``
        if the aggregator never rejects :math:`B(γ)` within the search
        range (i.e. :math:`B(γ)` remains in the selection set for
        arbitrarily large :math:`γ`).

        Args:
            honest_gradients: Honest gradient stack of shape ``(n-f, d)``.
            honest_mean: Mean of the honest gradients, shape ``(d,)``.
            direction: Attack direction :math:`E`, shape ``(d,)``.
            aggregator: Target aggregator class.
            n: Total number of workers.
            f: Number of Byzantine workers.
            aggregator_kwargs: Extra keyword arguments for the aggregator.

        Returns:
            The largest :math:`γ` for which the aggregator selects
            :math:`B(γ)`, as a Python float.
        """
        gamma_max = 1e6
        gamma_init = 1.0
        tol = 1e-3

        if not cls._is_selected(honest_gradients, honest_mean, aggregator, n, f, aggregator_kwargs):
            return 0.0

        low = 0.0
        high = gamma_init
        while high < gamma_max:
            b_gamma = honest_mean + high * direction
            if not cls._is_selected(honest_gradients, b_gamma, aggregator, n, f, aggregator_kwargs):
                break
            low = high
            high *= 2.0
        else:
            return gamma_max

        while high - low > tol:
            mid = 0.5 * (low + high)
            b_gamma = honest_mean + mid * direction
            if cls._is_selected(honest_gradients, b_gamma, aggregator, n, f, aggregator_kwargs):
                low = mid
            else:
                high = mid
        return low
