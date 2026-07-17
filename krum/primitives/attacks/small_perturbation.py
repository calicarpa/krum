r"""Small-perturbation gradient attack.

Reference:
    El Mahdi El Mhamdi, Rachid Guerraoui, and Sébastien Rouault. "The
    Hidden Vulnerability of Distributed Learning in Byzantium." In
    Proceedings of the 35th International Conference on Machine
    Learning (ICML 2018).

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

from torch import Tensor, arange, argmax, cat, isin, ones, stack, topk, zeros

from ..aggregators import Aggregator
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
        threshold: float = 0.5,
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
            threshold: Relative difference threshold for the aggregator selection
                test (default 0.5 = 50%). See :meth:`_is_selected`.
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
                threshold=threshold,
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
        threshold: float = 0.5,
    ) -> bool:
        r"""Test whether the target aggregator "selects" :math:`B(\gamma)`.

        For Krum-based aggregators (those with a ``score`` static method),
        the test directly checks whether at least one :math:`B(\gamma)` copy
        lands in the top-``m`` Krum scores — a subset-membership test.
        For all other aggregators, the test falls back to a relative-output-
        change heuristic: the Byzantine gradient is considered "selected" if
        swapping the honest mean for :math:`B(\gamma)` changes the aggregator
        output by more than ``threshold``.

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
            threshold: Relative difference threshold (default 0.5 = 50%).

        Returns:
            ``True`` if the aggregator selects :math:`B(\gamma)`.
        """
        byz_with = b_gamma.unsqueeze(0).expand(f, -1)
        stacked_with = cat([honest_gradients, byz_with], dim=0)

        if hasattr(aggregator, "score"):
            m = aggregator_kwargs.get("m", n - f - 2)
            if m < 1 or m > n - f - 2:
                m = min(max(m, 1), n - f - 2)
            scores = aggregator.score(stacked_with, n=n, f=f, num_peers=m)  # ty:ignore[call-non-callable]
            _, top_indices = topk(scores, m, largest=False)
            byz_indices = arange(n - f, n, device=stacked_with.device, dtype=top_indices.dtype)
            return bool(isin(top_indices, byz_indices).any().item())

        if out_without is None:
            honest_mean = honest_gradients.mean(dim=0)
            byz_without = honest_mean.unsqueeze(0).expand(f, -1)
            stacked_without = cat([honest_gradients, byz_without], dim=0)
            out_without = aggregator.aggregate(stacked_without, n=n, f=f, **aggregator_kwargs)

        out_with = aggregator.aggregate(stacked_with, n=n, f=f, **aggregator_kwargs)

        diff_norm = (out_with - out_without).norm()
        ref_norm = out_without.norm()
        if ref_norm < 1e-10:
            return diff_norm.item() > threshold
        relative_diff = (diff_norm / ref_norm).item()
        return relative_diff > threshold

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
        threshold: float = 0.5,
    ) -> float:
        r"""Find the largest :math:`\gamma` such that the aggregator selects :math:`B(\gamma)`.

        The selection test is *non-monotonic* in :math:`\gamma`: at
        :math:`\gamma = 0` the perturbation has no effect, at moderate
        :math:`\gamma` the byzantine vector shifts the aggregator's output
        beyond ``threshold`` (selected), and at large :math:`\gamma` the
        aggregator rejects :math:`B(\gamma)` and the output collapses back
        onto the honest majority (not selected). The "selected" region is
        therefore a bounded window :math:`[\gamma_{\text{low}}, \gamma_{\text{up}}]`.

        To handle this, the search first scans exponentially upward from
        ``gamma_init`` (skipping non-selected probes that sit below the
        window) until a selected :math:`\gamma` is found, then continues
        doubling until the window's upper edge is bracketed by a
        non-selected probe, and finally refines that edge with a binary
        search up to :math:`\varepsilon`.

        Returns :math:`0` if no :math:`\gamma` within the search range ever
        influences the aggregator (the attack has no effect), and
        :math:`\gamma_{\max}` if the aggregator never *rejects*
        :math:`B(\gamma)` within the search range (i.e. :math:`B(\gamma)`
        stays selected for arbitrarily large :math:`\gamma`).

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
            threshold: Relative difference threshold for the aggregator selection
                test (default 0.5 = 50%). See :meth:`_is_selected`.

        Returns:
            The largest ``\gamma`` for which the aggregator selects
            ``B(\gamma)``, as a Python float.
        """
        byz_placeholder = honest_mean.unsqueeze(0).expand(f, -1)
        stacked_without = cat([honest_gradients, byz_placeholder], dim=0)
        out_without = aggregator.aggregate(stacked_without, n=n, f=f, **aggregator_kwargs)

        def _selected(gamma: float) -> bool:
            return cls._is_selected(
                honest_gradients,
                honest_mean + gamma * direction,
                aggregator,
                n,
                f,
                aggregator_kwargs,
                out_without=out_without,
                threshold=threshold,
            )

        selected_probe: float | None = None
        rejection_probe: float | None = None
        probe = gamma_init
        while probe < gamma_max:
            if _selected(probe):
                selected_probe = probe
            elif selected_probe is not None:
                rejection_probe = probe
                break
            probe *= 2.0

        if selected_probe is None:
            return 0.0
        if rejection_probe is None:
            return gamma_max

        low = selected_probe
        high = rejection_probe
        while high - low > tol:
            mid = 0.5 * (low + high)
            if _selected(mid):
                low = mid
            else:
                high = mid
        return low
