###
# @file   brute.py
# @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
#
# @section LICENSE
#
# Copyright © 2018-2021 École Polytechnique Fédérale de Lausanne (EPFL).
# See LICENSE file.
#
# @section DESCRIPTION
#
# Brute GAR.
###

"""
The Brute aggregation rule exhaustively searches all subsets of
:math:`n - f` gradients and selects the subset with the smallest finite
diameter. The diameter of a subset is the maximum pairwise distance between
any two gradients in that subset.

Use Case
--------
Theoretical baseline for evaluating other aggregation rules. Brute provides
strong Byzantine resilience guarantees, but its combinatorial search makes it
practical only for small worker counts or controlled experiments.

Properties
----------
- Exhaustive search: evaluates every :math:`\\binom{n}{n-f}` candidate subset.
- Optimal selection: returns a smallest-diameter valid subset under the explored
  objective.
- Limited scalability: intended for small :math:`n` or research baselines.

Theoretical Bound
-----------------
The Brute rule provides the best theoretical guarantees:

.. math::

    \\frac{\\sigma}{\\|g\\|} \\leq \\frac{n - f}{\\sqrt{8} f}

where :math:`\\sigma` is the standard deviation of honest gradients.

Complexity
----------
- Time: :math:`O(\\binom{n}{n-f} \\cdot d \\cdot n^2)` where :math:`d` is the
  gradient dimension.
- Space: :math:`O(n^2)` for storing pairwise distances.

Example
-------
>>> import torch
>>> from aggregators import brute
>>> gradients = [
...     torch.tensor([1., 2., 3.]),
...     torch.tensor([1.1, 2.1, 3.1]),
...     torch.tensor([0.9, 1.9, 2.9]),
...     torch.tensor([100., 200., 300.]),    # Byzantine
...     torch.tensor([-100., -200., -300.])  # Byzantine
... ]
>>> result = brute(gradients=gradients, f=2)
tensor([1., 2., 3.])
"""

import itertools
import math

import tools
import torch

from . import register

# Optional 'native' module
try:
    import native
except ImportError:
    native = None

# ---------------------------------------------------------------------------- #
# Brute GAR


def _compute_selection(
    gradients: list[torch.Tensor], f: int, **kwargs
) -> tuple[int, ...]:
    """
    Select the gradient indices forming the smallest-diameter subset.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of candidate gradients.
    f : int
        Number of Byzantine gradients to tolerate.
    **kwargs : object
        Additional keyword arguments, ignored by this helper.

    Returns
    -------
    tuple of int
        Indices of the selected :math:`n - f` gradients.

    Notes
    -----
    Candidate subsets containing non-finite pairwise distances are ignored.
    """
    n = len(gradients)
    # Compute all pairwise distances
    distances = [0] * (n * (n - 1) // 2)
    for i, (x, y) in enumerate(tools.pairwise(tuple(range(n)))):
        distances[i] = gradients[x].sub(gradients[y]).norm().item()
    # Select the set of smallest diameter
    sel_iset = None
    sel_diam = None
    for cur_iset in itertools.combinations(range(n), n - f):
        # Compute the current diameter (max of pairwise distances)
        cur_diam = 0.0
        for x, y in tools.pairwise(cur_iset):
            # Get distance between these two gradients ("magic" formula valid since x < y)
            cur_dist = distances[(2 * n - x - 3) * x // 2 + y - 1]
            # Check finite distance (non-Byzantine gradient must only contain finite coordinates), drop set if non-finite
            if not math.isfinite(cur_dist):
                break
            # Check if new maximum
            cur_diam = max(cur_diam, cur_dist)
        else:
            # Check if new selected diameter
            if sel_iset is None or cur_diam < sel_diam:
                sel_iset = cur_iset
                sel_diam = cur_diam
    # Return the selected gradients
    assert sel_iset is not None, (
        "Too many non-finite gradients: a non-Byzantine gradient must only contain finite coordinates"
    )
    return sel_iset


def aggregate(gradients: list[torch.Tensor], f: int, **kwargs) -> torch.Tensor | float:
    """
    Compute the Brute aggregation (mean of smallest-diameter subset).

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients to aggregate.
    f : int
        Number of Byzantine gradients to tolerate. Must satisfy
        ``1 <= f <= (n - 1) // 2`` where ``n = len(gradients)``.
    **kwargs : object
        Additional keyword arguments, ignored by this implementation.

    Returns
    -------
    torch.Tensor
        Mean of the selected :math:`n - f` gradients with smallest finite
        diameter.

    Notes
    -----
    The returned tensor is newly computed and does not alias any input tensor.
    """
    sel_iset = _compute_selection(gradients, f, **kwargs)
    return sum(gradients[i] for i in sel_iset).div_(len(gradients) - f)


def aggregate_native(
    gradients: list[torch.Tensor], f: int, **kwargs
) -> torch.Tensor | float:
    """
    Compute the Brute aggregation using native C++/CUDA acceleration.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients to aggregate.
    f : int
        Number of Byzantine gradients to tolerate.
    **kwargs : object
        Additional keyword arguments, ignored by this implementation.

    Returns
    -------
    torch.Tensor | float
        Mean of the subset selected by the native Brute implementation.
    """
    return native.brute.aggregate(gradients, f)


def check(gradients: list[torch.Tensor], f: int, **kwargs) -> str | None:
    """
    Check parameter validity for the Brute aggregation rule.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients to aggregate.
    f : int
        Number of Byzantine gradients to tolerate.
    **kwargs : object
        Additional keyword arguments, ignored by this check.

    Returns
    -------
    str or None
        ``None`` when parameters are valid, otherwise a user-facing error
        message.
    """
    if not isinstance(gradients, list) or len(gradients) < 1:
        return (
            f"Expected a list of at least one gradient to aggregate, got {gradients!r}"
        )
    if not isinstance(f, int) or f < 1 or len(gradients) < 2 * f + 1:
        return (
            "Invalid number of Byzantine gradients to tolerate, got f = %r, expected 1 ≤ f ≤ %d"
            % (f, (len(gradients) - 1) // 2)
        )
    return None


def upper_bound(n: int, f: int, d: int) -> float:
    """
    Compute the theoretical Brute resilience bound.

    Parameters
    ----------
    n : int
        Total number of workers, including Byzantine workers.
    f : int
        Expected number of Byzantine workers.
    d : int
        Dimension of the gradient space.

    Returns
    -------
    float
        Upper bound on the ratio between non-Byzantine standard deviation and
        gradient norm under which the rule is expected to apply.
    """
    return (n - f) / (math.sqrt(8) * f)


def influence(
    honests: list[torch.Tensor], attacks: list[torch.Tensor], f: int, **kwargs
) -> float:
    """
    Compute the ratio of Byzantine gradients selected by Brute.

    Parameters
    ----------
    honests : list of torch.Tensor
        Non-empty list of honest gradients.
    attacks : list of torch.Tensor
        List of attack, or Byzantine, gradients.
    f : int
        Number of Byzantine gradients to tolerate.
    **kwargs : object
        Additional keyword arguments forwarded to the selection helper.

    Returns
    -------
    float
        Fraction of selected gradients that come from ``attacks``.
    """
    gradients = honests + attacks
    # Compute the selection set
    sel_iset = _compute_selection(gradients, f, **kwargs)
    # Compute the influence ratio
    count = 0
    for i in sel_iset:
        gradient = gradients[i]
        for attack in attacks:
            if gradient is attack:
                count += 1
                break
    return count / (len(gradients) - f)


# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (pytorch version)
method_name = "brute"
register(method_name, aggregate, check, upper_bound=upper_bound, influence=influence)

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
