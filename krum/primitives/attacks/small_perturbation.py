r"""Small-perturbation attack (El Mhamdi, Guerraoui, Rouault — ICML 2018).

The attack exploits the curse of dimensionality: in ``d ≫ 1``, two honest
gradients naturally disagree by ``Θ(√d)`` in the ``ℓp`` norm even on a
"good" coordinate, so an attacker can shift one coordinate by ``Ω(√d)``
while still looking legitimate to a ``ℓp``-norm-based aggregator.

The attack builds a single malicious vector

.. math::

    B(γ) = \\frac{1}{n - f} \\sum_{i=1}^{n-f} V_i  +  γ \\cdot E

from the ``n − f`` honest gradients :math:`V_1, \\dots, V_{n-f}` and the
direction :math:`E \\in \\mathbb{R}^d`. It then performs a boundary search
for the largest ``γ = γ_m`` such that the target aggregator still
"selects" ``B(γ_m)`` — i.e. ``B(γ_m)`` is in the aggregator's input
subset. All ``f`` Byzantine workers send the same ``B(γ_m)`` vector.

Two directions ``E`` are supported, depending on the target norm
``p`` (Section 3.2 vs Section 3.3):

* **finite norm** ``p ≥ 1`` (Section 3.2): ``E = e_e`` is a one-hot
  vector at a single coordinate :math:`e \\in \\{1, \\dots, d\\}`. The
  default choice :math:`e = \\arg\\max_j \\mathrm{std}(V_{\\cdot,j})` is
  the coordinate with the largest honest variance, which maximises
  :math:`γ_m`.
* **infinite norm** ``p = ∞`` (Section 3.3): ``E = (1, \\dots, 1)`, the
  all-ones vector. This is required because
  :math:`\\lim_{p \\to +\\infty} \\sqrt[p]{d} = 1`, so the finite-norm
  attack loses its bite as ``p`` grows. Modifying non-maximal coordinates
  does not substantially affect the infinite-norm distance to the honest
  gradients, so :math:`B(γ)` stays in the aggregator's selection set for
  :math:`γ = Θ(d)`.

Reference:
    El Mahdi El Mhamdi, Rachid Guerraoui, and Sébastien Rouault. "The
    Hidden Vulnerability of Distributed Learning in Byzantium." In
    Proceedings of the 35th International Conference on Machine Learning
    (ICML 2018), Section 3.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ..aggregators import Aggregator
from . import Attack


class SmallPerturbationAttack(Attack):
    r"""Small-perturbation attack — Section 3 of El Mhamdi et al. (ICML 2018).

    The attack accepts a target aggregator and a target norm, builds the
    direction :math:`E` accordingly, and performs a boundary search for the
    largest :math:`γ` such that the aggregator still "selects" :math:`B(γ)`.

    The aggregator is asked to "select" :math:`B(γ)` in the same way as the
    actual run: the test substitutes a placeholder for :math:`B(γ)` (the
    honest mean) and checks whether the aggregator's output changes. This
    works uniformly for any stateless aggregator class.

    Args:
        aggregator: Target aggregator class (e.g. ``Krum``, ``GeoMed``,
            ``Brute``, ``Bulyan``). Used both as the actual aggregator in
            the simulation and as the reference rule during the boundary
            search.
        n: Total number of workers. Must satisfy :math:`n \\geq 2f + 1`.
        f: Number of Byzantine workers. Must satisfy :math:`f \\geq 1`.
        p: Target norm. ``2`` for the finite-norm attack of Section 3.2;
            ``math.inf`` for the Section 3.3 variant. The default is ``2``
            (the most common case, matching the paper's analysis).
        coordinate: Index of the poisoned coordinate (one-hot direction)
            for ``p < ∞``. Accepts an integer index, the string
            ``"max"`` to pick the coordinate with the largest honest
            variance (the default), or the string ``"all"`` to use the
            all-ones direction (forced when ``p == ∞``).
        aggregator_kwargs: Extra keyword arguments forwarded to
            ``aggregator.aggregate`` during the boundary search (e.g.
            ``{"m": 1}`` for ``Bulyan`` with Krum).
        gamma_max: Upper bound on the search for :math:`γ_m`. If the
            aggregator never rejects :math:`B(γ)` up to this value, the
            search returns ``gamma_max`` (a "no-break" outcome).
        gamma_init: Initial step used during the exponential search for
            the upper bound of the boundary.
        tol: Tolerance of the binary refinement of the boundary.

    Raises:
        ValueError: If ``n``, ``f``, ``p``, or ``coordinate`` is invalid.
    """

    def __init__(
        self,
        *,
        aggregator: type[Aggregator],
        n: int,
        f: int,
        p: int | float = 2,
        coordinate: int | str | None = None,
        aggregator_kwargs: dict[str, Any] | None = None,
        gamma_max: float = 1e6,
        gamma_init: float = 1.0,
        tol: float = 1e-3,
    ) -> None:
        """Initialize the small-perturbation attack."""
        if n < 1:
            raise ValueError(f"Invalid total workers, got n = {n!r}, expected n >= 1")
        if f < 1:
            raise ValueError(f"Invalid Byzantine workers, got f = {f!r}, expected f >= 1")
        if f > n:
            raise ValueError(f"Invalid Byzantine workers, got f = {f!r}, expected f <= n = {n!r}")
        if n < 2 * f + 1:
            raise ValueError(f"Invalid worker configuration, got (n={n}, f={f}); Brute requires n >= 2f + 1")
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

        self.aggregator = aggregator
        self.n = n
        self.f = f
        self.p = p
        self.coordinate = coordinate
        self.aggregator_kwargs = aggregator_kwargs or {}
        self.gamma_max = gamma_max
        self.gamma_init = gamma_init
        self.tol = tol

    def generate(
        self,
        honest_gradients: torch.Tensor,
        num_byzantine: int,
    ) -> torch.Tensor:
        """Generate the small-perturbation Byzantine gradients.

        Args:
            honest_gradients: Tensor of shape ``(h, d)`` containing gradients
                from the ``h = n - f`` honest workers. Must be a 2-D
                floating-point tensor.
            num_byzantine: Number of Byzantine gradients to generate. Must
                equal ``self.f`` (the attack is defined for a fixed worker
                configuration).

        Returns:
            Tensor of shape ``(f, d)`` containing the same ``B(γ_m)``
            vector repeated ``f`` times.

        Raises:
            ValueError: If the input shape is invalid or ``num_byzantine``
                does not match ``self.f``.
            TypeError: If the input is not floating-point.
        """
        if honest_gradients.ndim != 2:
            raise ValueError("Expected a 2D tensor of honest gradients")
        if not torch.is_floating_point(honest_gradients):
            raise TypeError("Expected honest gradients to use a floating-point dtype")
        if num_byzantine != self.f:
            raise ValueError(f"Mismatch between num_byzantine = {num_byzantine!r} and configured f = {self.f!r}")
        if honest_gradients.shape[0] != self.n - self.f:
            raise ValueError(f"Expected {self.n - self.f} honest gradients, got {honest_gradients.shape[0]}")

        device = honest_gradients.device
        dtype = honest_gradients.dtype
        d = honest_gradients.shape[1]

        honest_mean = honest_gradients.mean(dim=0)
        direction = self._build_direction(honest_gradients, d, device, dtype)
        gamma_m = self._find_gamma_max(honest_gradients, honest_mean, direction)
        b_gamma = honest_mean + gamma_m * direction
        return b_gamma.unsqueeze(0).expand(self.f, -1).contiguous()

    def _build_direction(
        self,
        honest_gradients: torch.Tensor,
        d: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        r"""Build the attack direction :math:`E` for the configured norm.

        For ``p == math.inf`` (Section 3.3), :math:`E = (1, \\dots, 1)`.
        Otherwise (Section 3.2, finite ``p ≥ 1``), :math:`E` is a one-hot
        vector at the configured coordinate; the default is the coordinate
        with the largest honest variance.

        Args:
            honest_gradients: Honest gradient stack of shape ``(n-f, d)``.
            d: Gradient dimension.
            device: Device of the output tensor.
            dtype: Dtype of the output tensor.

        Returns:
            Direction tensor of shape ``(d,)``.
        """
        if self.p == math.inf or self.coordinate == "all":
            return torch.ones(d, device=device, dtype=dtype)

        if self.coordinate is None or self.coordinate in {"max", "largest"}:
            std = honest_gradients.std(dim=0, correction=0)
            idx = int(torch.argmax(std).item())
        else:
            idx = int(self.coordinate)

        if not 0 <= idx < d:
            raise ValueError(f"Coordinate {idx!r} out of range [0, {d})")

        direction = torch.zeros(d, device=device, dtype=dtype)
        direction[idx] = 1.0
        return direction

    def _is_selected(
        self,
        honest_gradients: torch.Tensor,
        b_gamma: torch.Tensor,
    ) -> bool:
        """Test whether the target aggregator "selects" ``B(γ)``.

        The test substitutes a placeholder for :math:`B(γ)` (the honest
        mean) and checks whether the aggregator's output changes. This is
        a uniform test that works for any stateless aggregator: it does
        not depend on the aggregator's internal selection rule.

        Args:
            honest_gradients: Honest gradient stack of shape ``(n-f, d)``.
            b_gamma: Candidate :math:`B(γ)` of shape ``(d,)``.

        Returns:
            ``True`` if the aggregator's output is sensitive to :math:`B(γ)`.
        """
        honest_mean = honest_gradients.mean(dim=0)
        byz_with = b_gamma.unsqueeze(0).expand(self.f, -1)
        byz_without = honest_mean.unsqueeze(0).expand(self.f, -1)

        stacked_with = torch.cat([honest_gradients, byz_with], dim=0)
        stacked_without = torch.cat([honest_gradients, byz_without], dim=0)

        out_with = self.aggregator.aggregate(list(stacked_with), n=self.n, f=self.f, **self.aggregator_kwargs)
        out_without = self.aggregator.aggregate(list(stacked_without), n=self.n, f=self.f, **self.aggregator_kwargs)
        return not torch.allclose(out_with, out_without, atol=1e-6, rtol=1e-5)

    def _find_gamma_max(
        self,
        honest_gradients: torch.Tensor,
        honest_mean: torch.Tensor,
        direction: torch.Tensor,
    ) -> float:
        """Find the largest :math:`γ` such that the aggregator selects :math:`B(γ)`.

        Uses an exponential search to bracket the boundary, then refines
        it with binary search up to ``self.tol``. Returns ``self.gamma_max``
        if the aggregator never rejects :math:`B(γ)` within the search
        range (i.e. :math:`B(γ)` remains in the selection set for
        arbitrarily large :math:`γ`).

        Args:
            honest_gradients: Honest gradient stack of shape ``(n-f, d)``.
            honest_mean: Mean of the honest gradients, shape ``(d,)``.
            direction: Attack direction :math:`E`, shape ``(d,)`.

        Returns:
            The largest :math:`γ` for which the aggregator selects
            :math:`B(γ)`, as a Python float.
        """
        if not self._is_selected(honest_gradients, honest_mean):
            return 0.0

        low = 0.0
        high = self.gamma_init
        while high < self.gamma_max:
            b_gamma = honest_mean + high * direction
            if not self._is_selected(honest_gradients, b_gamma):
                break
            low = high
            high *= 2.0
        else:
            return self.gamma_max

        while high - low > self.tol:
            mid = 0.5 * (low + high)
            b_gamma = honest_mean + mid * direction
            if self._is_selected(honest_gradients, b_gamma):
                low = mid
            else:
                high = mid
        return low
