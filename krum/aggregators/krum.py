###
# @file   krum.py
# @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
#
# @section LICENSE
#
# Copyright © 2018-2020 École Polytechnique Fédérale de Lausanne (EPFL).
# See LICENSE file.
#
# @section DESCRIPTION
#
# Multi-Krum GAR.
###

r"""Krum and Multi-Krum are distance-based Byzantine-resilient aggregation rules.

For each candidate gradient, the rule computes a score by summing the distances
to its :math:`n - f - 1` nearest neighbours. It then selects the :math:`m`
lowest-scoring gradients and returns their average. Honest gradients are
expected to cluster together, while Byzantine gradients should receive larger
scores when they are far from the honest majority.

Use Case
--------

General Byzantine-resilient aggregation when Byzantine gradients are expected to
be geometrically separated from honest gradients.

Properties
----------

- Distance-based: Relies on pairwise gradient distances.
- Selects a subset: Not all gradients contribute to the final average.
- Multi-Krum: Averages the best :math:`m` candidates instead of selecting only
  one candidate.
- Theoretical bound available through :func:`upper_bound`.

Theoretical Bound
-----------------

The Multi-Krum rule provides guarantees when:

.. math::

    \\frac{\\sigma}{\\|g\\|} \\leq \\frac{1}{\\sqrt{2 (n - f + \\frac{f(n + f(n - f - 2) - 2)}{n - 2f - 2})}}

where:

- :math:`n` is the total number of workers.
- :math:`f` is the number of Byzantine workers.
- :math:`\\sigma` is the standard deviation of honest gradients.
- :math:`\\|g\\|` is the norm of the honest gradient.

Complexity
----------
- Time: :math:`O(n^2 \\cdot d)` where :math:`n` is the number of gradients and
  :math:`d` is the gradient dimension.
- Space: :math:`O(n^2)` for storing pairwise distances.

Parameters
----------
m : int, optional
    Number of gradients to select for averaging. Defaults to ``n - f - 2``.
    Must satisfy ``1 <= m <= n - f - 2``.

Example:
-------
>>> import torch
>>> from aggregators import krum
>>> gradients = [
...     torch.tensor([1., 2., 3.]),
...     torch.tensor([1.1, 2.1, 3.1]),
...     torch.tensor([0.9, 1.9, 2.9]),
...     torch.tensor([1.2, 2.2, 3.2]),
...     torch.tensor([100., 200., 300.]),
... ]
>>> result = krum(gradients=gradients, f=1, m=2)
>>> result
tensor([1.0500, 2.0500, 3.0500])
"""

import math

import torch

from .. import tools
from . import register

# Optional 'native' module
try:
    from krum import native
except ImportError:
    native = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------- #
# Multi-Krum GAR


def _compute_scores(gradients: list[torch.Tensor], f: int, m: int, **kwargs) -> list[tuple[float, torch.Tensor]]:
    """Compute Multi-Krum scores for all candidate gradients.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients.
    f : int
        Number of Byzantine gradients to tolerate.
    m : int
        Number of gradients to select.
    **kwargs : object
        Additional keyword arguments, ignored by this implementation.

    Returns:
    -------
    list of tuple[float, torch.Tensor]
        Candidate gradients paired with their scores, sorted by increasing
        score.
    """
    n = len(gradients)
    # Compute all pairwise distances
    distances = [0] * (n * (n - 1) // 2)
    for i, (x, y) in enumerate(tools.pairwise(tuple(range(n)))):
        dist = gradients[x].sub(gradients[y]).norm().item()
        if not math.isfinite(dist):
            dist = math.inf
        distances[i] = dist
    # Compute the scores
    scores = []
    for i in range(n):
        # Collect the distances
        grad_dists = []
        for j in range(i):
            grad_dists.append(distances[(2 * n - j - 3) * j // 2 + i - 1])
        for j in range(i + 1, n):
            grad_dists.append(distances[(2 * n - i - 3) * i // 2 + j - 1])
        # Select the n - f - 1 smallest distances
        grad_dists.sort()
        scores.append((sum(grad_dists[: n - f - 1]), gradients[i]))
    # Sort the gradients by increasing scores
    scores.sort(key=lambda x: x[0])
    return scores


def aggregate(gradients: list[torch.Tensor], f: int, m: int | None = None, **kwargs) -> torch.Tensor:
    """Aggregate gradients with Multi-Krum.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients to aggregate.
    f : int
        Number of Byzantine gradients to tolerate. Must satisfy
        ``1 <= f <= (n - 3) // 2`` where ``n = len(gradients)``.
    m : int, optional
        Number of gradients to select for averaging. Defaults to
        ``n - f - 2``. Must satisfy ``1 <= m <= n - f - 2``.
    **kwargs : object
        Additional keyword arguments, ignored by this implementation.

    Returns:
    -------
    torch.Tensor
        Average of the selected ``m`` gradients with the smallest Krum scores.

    Notes:
    -----
    The output tensor is newly created and does not alias any input tensor.
    """
    # Defaults
    if m is None:
        m = len(gradients) - f - 2
    # Compute aggregated gradient
    scores = _compute_scores(gradients, f, m, **kwargs)
    return sum(grad for _, grad in scores[:m]).div_(m)


def aggregate_native(gradients: list[torch.Tensor], f: int, m: int | None = None, **kwargs) -> torch.Tensor:
    """Aggregate gradients with the native Multi-Krum implementation.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients to aggregate.
    f : int
        Number of Byzantine gradients to tolerate.
    m : int, optional
        Number of gradients to select. Defaults to ``n - f - 2``.
    **kwargs : object
        Additional keyword arguments, ignored by this implementation.

    Returns:
    -------
    torch.Tensor
        Average of the selected gradients.
    """
    # Defaults
    if m is None:
        m = len(gradients) - f - 2
    # Computation
    return native.krum.aggregate(gradients, f, m)  # type: ignore[attr-defined]


def check(gradients: list[torch.Tensor], f: int, m: int | None = None, **kwargs) -> str | None:
    """Check whether Multi-Krum can be used with the given parameters.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients to aggregate.
    f : int
        Number of Byzantine gradients to tolerate.
    m : int, optional
        Number of gradients to select.
    **kwargs : object
        Additional keyword arguments, ignored by this implementation.

    Returns:
    -------
    str or None
        ``None`` when the parameters are valid, otherwise a user-facing error
        message.
    """
    if not isinstance(gradients, list) or len(gradients) < 1:
        return f"Expected a list of at least one gradient to aggregate, got {gradients!r}"
    if not isinstance(f, int) or f < 1 or len(gradients) < 2 * f + 3:
        return "Invalid number of Byzantine gradients to tolerate, got f = %r, expected 1 ≤ f ≤ %d" % (
            f,
            (len(gradients) - 3) // 2,
        )
    if m is not None and (not isinstance(m, int) or m < 1 or m > len(gradients) - f - 2):
        return "Invalid number of selected gradients, got m = %r, expected 1 ≤ m ≤ %d" % (m, len(gradients) - f - 2)
    return None


def upper_bound(n: int, f: int, d: int) -> float:
    """Compute the theoretical Multi-Krum robustness bound.

    Parameters
    ----------
    n : int
        Number of workers, including honest and Byzantine workers.
    f : int
        Expected number of Byzantine workers.
    d : int
        Dimension of the gradient space. This parameter is accepted for the
        standard GAR metadata contract and is not used by this formula.

    Returns:
    -------
    float
        Upper bound on the ratio between non-Byzantine standard deviation and
        gradient norm.
    """
    return 1 / math.sqrt(2 * (n - f + f * (n + f * (n - f - 2) - 2) / (n - 2 * f - 2)))


def influence(
    honests: list[torch.Tensor],
    attacks: list[torch.Tensor],
    f: int,
    m: int | None = None,
    **kwargs,
) -> float:
    """Compute the ratio of Byzantine gradients selected by Multi-Krum.

    Parameters
    ----------
    honests : list of torch.Tensor
        Non-empty list of honest gradients.
    attacks : list of torch.Tensor
        List of attack, or Byzantine, gradients.
    f : int
        Number of Byzantine gradients to tolerate.
    m : int, optional
        Number of gradients to select. Defaults to ``n - f - 2``.
    **kwargs : object
        Additional keyword arguments forwarded to score computation.

    Returns:
    -------
    float
        Ratio of selected gradients that come from ``attacks``.
    """
    gradients = honests + attacks
    # Defaults
    if m is None:
        m = len(gradients) - f - 2
    # Compute the sorted scores
    scores = _compute_scores(gradients, f, m, **kwargs)
    # Compute the influence ratio
    count = 0
    for _, gradient in scores[:m]:
        for attack in attacks:
            if gradient is attack:
                count += 1
                break
    return count / m


# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (pytorch version)
method_name = "krum"
register(method_name, aggregate, check, upper_bound, influence)

# Register aggregation rule (native version, if available)
if native is not None:
    native_name = method_name
    method_name = "native-" + method_name
    if native_name in dir(native):
        register(method_name, aggregate_native, check, upper_bound)
    else:
        tools.warning(
            f"GAR {method_name!r} could not be registered since the associated native module {native_name!r} is unavailable"
        )
