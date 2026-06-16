r"""Small-perturbation attack.

The attack exploits the curse of dimensionality: in :math:`d \gg 1`, two honest
gradients naturally disagree by :math:`\Theta(\sqrt{d})` in the :math:`\ell_p`
norm even on a "good" coordinate, so an attacker can shift one coordinate by
:math:`\Omega(\sqrt{d})` while still looking legitimate to a
:math:`\ell_p`-norm-based aggregator.

The attack builds a single malicious vector

.. math::

    B(\gamma) = \frac{1}{n - f} \sum_{i=1}^{n-f} V_i  +  \gamma \cdot E

from the :math:`n - f` honest gradients :math:`V_1, \dots, V_{n-f}` and the
direction :math:`E \in \mathbb{R}^d`. It then performs a boundary search
for the largest :math:`\gamma = \gamma_m` such that the target aggregator still
"selects" :math:`B(\gamma_m)` — i.e. :math:`B(\gamma_m)` is in the aggregator's
input subset. All :math:`f` Byzantine workers send the same
:math:`B(\gamma_m)` vector.

Two directions :math:`E` are supported, depending on the target norm
:math:`p`:

* **finite norm** :math:`p \ge 1`: :math:`E = e_e` is a one-hot
  vector at a single coordinate :math:`e \in \{1, \dots, d\}`. The
  default choice :math:`e = \arg\max_j \operatorname{std}(V_{\cdot,j})` is
  the coordinate with the largest honest variance, which maximises
  :math:`\gamma_m`.
* **infinite norm** :math:`p = \infty`: :math:`E = (1, \dots, 1)`, the
  all-ones vector. This is required because
  :math:`\lim_{p \to +\infty} \sqrt[p]{d} = 1`, so the finite-norm
  attack loses its bite as :math:`p` grows. Modifying non-maximal coordinates
  does not substantially affect the infinite-norm distance to the honest
  gradients, so :math:`B(\gamma)` stays in the aggregator's selection set for
  :math:`\gamma = \Theta(d)`.
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
    largest :math:`\gamma` such that the aggregator still "selects"
    :math:`B(\gamma)`.

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
        gamma: float | None = None,
        gamma_max: float = 1e6,
        gamma_init: float = 1.0,
        tol: float = 1e-3,
        atol: float = 1e-6,
        rtol: float = 1e-5,
        **specialized: Any,
    ) -> Tensor:
        r"""Generate Byzantine gradients.

        Args:
            honest_gradients: Sequence of :math:`h` gradient vectors, one per honest
                worker, each of shape :math:`(d,)`. May also be a 2-D tensor of shape
                :math:`(n-f, d)`.
            out: Optional pre-allocated tensor of shape :math:`(f, d)` to write the
                result into and return.
            f: Number of Byzantine gradients to generate. Must equal the configured
                :math:`f` (the attack is defined for a fixed worker configuration).
            aggregator: Target aggregator class.
            n: Total number of workers.
            p: Target norm.
            coordinate: Index of the poisoned coordinate.
            aggregator_kwargs: Extra keyword arguments for the aggregator.
            gamma: If provided, use this exact value for :math:`\gamma` instead of
                running the boundary search. This is useful when the caller knows
                the desired perturbation magnitude or wants to compare aggregators
                at a fixed attack strength.
            gamma_max: Upper bound on the search for :math:`\gamma_m`.
            gamma_init: Initial step used during the exponential search.
            tol: Tolerance of the binary refinement.
            atol: Absolute tolerance for the aggregator selection test
                (see :meth:`_is_selected`).
            rtol: Relative tolerance for the aggregator selection test.
            **specialized: Additional keyword arguments.

        Returns:
            Byzantine gradients of shape ``(f, d)`` containing the same
            ``B(\gamma_m)`` vector repeated ``f`` times.

        Raises:
            ValueError: If the input shape is invalid or :math:`f` does not match.
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
            raise ValueError("coordinate='all' is only valid with p=math.inf")
        if len(honest_gradients) == 0:
            raise ValueError("Expected at least one honest gradient")

        if isinstance(honest_gradients, Tensor):
            stacked = honest_gradients
        else:
            stacked = stack(list(honest_gradients))
        if stacked.shape[0] != n - f:
            raise ValueError(f"Expected {n - f} honest gradients, got {stacked.shape[0]}")

        aggregator_kwargs = aggregator_kwargs or {}
        device = stacked.device
        dtype = stacked.dtype
        d = stacked.shape[1]

        honest_mean = stacked.mean(dim=0)
        direction = cls._build_direction(stacked, d, device, dtype, p, coordinate)
        if gamma is not None:
            gamma_m = gamma
        else:
            gamma_m = cls._find_gamma_max(
                stacked,
                honest_mean,
                direction,
                aggregator,
                n,
                f,
                aggregator_kwargs,
                gamma_max=gamma_max,
                gamma_init=gamma_init,
                tol=tol,
                atol=atol,
                rtol=rtol,
            )
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

        For :math:`p = \infty`, :math:`E = (1, \dots, 1)`.
        Otherwise (finite :math:`p \ge 1`), :math:`E` is a
        one-hot vector at the configured coordinate; the default is the
        coordinate with the largest honest variance.

        Args:
            honest_gradients: Honest gradient stack of shape :math:`(n-f, d)`.
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
            # correction=0 → population std (matches the paper's reference
            # implementation from the LPD-EPFL lab).
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
        *,
        out_without: Tensor | None = None,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> bool:
        r"""Test whether the target aggregator "selects" :math:`B(\gamma)`.

        This is a heuristic: it substitutes a placeholder for :math:`B(\gamma)`
        (the honest mean) and checks whether the aggregator's output changes
        within the given tolerances. It works uniformly for any stateless
        aggregator, but the absolute/relative tolerances may need tuning
        depending on the gradient scale.

        Args:
            honest_gradients: Honest gradient stack of shape :math:`(n-f, d)`.
            b_gamma: Candidate :math:`B(\gamma)` of shape :math:`(d,)`.
            aggregator: Target aggregator class.
            n: Total number of workers.
            f: Number of Byzantine workers.
            aggregator_kwargs: Extra keyword arguments for the aggregator.
            out_without: Pre-computed aggregator output when the Byzantine
                gradients are replaced by the honest mean. If ``None``, it is
                computed on the fly.
            atol: Absolute tolerance for the output comparison.
            rtol: Relative tolerance for the output comparison.

        Returns:
            ``True`` if the aggregator's output is sensitive to
            ``B(\gamma)``.
        """
        if out_without is None:
            honest_mean = honest_gradients.mean(dim=0)
            byz_without = honest_mean.unsqueeze(0).expand(f, -1)
            stacked_without = cat([honest_gradients, byz_without], dim=0)
            out_without = aggregator.aggregate(stacked_without, n=n, f=f, **aggregator_kwargs)

        byz_with = b_gamma.unsqueeze(0).expand(f, -1)
        stacked_with = cat([honest_gradients, byz_with], dim=0)
        out_with = aggregator.aggregate(stacked_with, n=n, f=f, **aggregator_kwargs)
        return not allclose(out_with, out_without, atol=atol, rtol=rtol)

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
        *,
        gamma_max: float = 1e6,
        gamma_init: float = 1.0,
        tol: float = 1e-3,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> float:
        r"""Find the largest :math:`\gamma` such that the aggregator selects :math:`B(\gamma)`.

        Uses an exponential search to bracket the boundary, then refines
        it with binary search up to :math:`\varepsilon`. Returns
        :math:`\gamma_{\max}` if the aggregator never rejects
        :math:`B(\gamma)` within the search range (i.e. :math:`B(\gamma)`
        remains in the selection set for arbitrarily large
        :math:`\gamma`).

        Args:
            honest_gradients: Honest gradient stack of shape :math:`(n-f, d)`.
            honest_mean: Mean of the honest gradients, shape :math:`(d,)`.
            direction: Attack direction :math:`E`, shape :math:`(d,)`.
            aggregator: Target aggregator class.
            n: Total number of workers.
            f: Number of Byzantine workers.
            aggregator_kwargs: Extra keyword arguments for the aggregator.
            gamma_max: Upper bound on the search.
            gamma_init: Initial step used during the exponential search.
            tol: Tolerance of the binary refinement.
            atol: Absolute tolerance for the aggregator selection test.
            rtol: Relative tolerance for the aggregator selection test.

        Returns:
            The largest ``\gamma`` for which the aggregator selects
            ``B(\gamma)``, as a Python float.
        """
        # Pre-compute the aggregator output with the honest mean as placeholder.
        # This avoids calling aggregate() twice per selection test.
        byz_placeholder = honest_mean.unsqueeze(0).expand(f, -1)
        stacked_without = cat([honest_gradients, byz_placeholder], dim=0)
        out_without = aggregator.aggregate(stacked_without, n=n, f=f, **aggregator_kwargs)

        low = 0.0
        high = gamma_init
        while high < gamma_max:
            b_gamma = honest_mean + high * direction
            if not cls._is_selected(
                honest_gradients,
                b_gamma,
                aggregator,
                n,
                f,
                aggregator_kwargs,
                out_without=out_without,
                atol=atol,
                rtol=rtol,
            ):
                break
            low = high
            high *= 2.0
        else:
            return gamma_max

        while high - low > tol:
            mid = 0.5 * (low + high)
            b_gamma = honest_mean + mid * direction
            if cls._is_selected(
                honest_gradients,
                b_gamma,
                aggregator,
                n,
                f,
                aggregator_kwargs,
                out_without=out_without,
                atol=atol,
                rtol=rtol,
            ):
                low = mid
            else:
                high = mid
        return low
