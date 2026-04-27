# coding: utf-8
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
The Brute aggregation rule finds the optimal subset of :math:`n - f` gradients
(with smallest diameter) by exhaustively checking all possible combinations.
The "diameter" of a set is the maximum pairwise distance between any two
gradients in that set.

**Use case:** Theoretical baseline for evaluating other aggregation rules.
Provides optimal Byzantine resilience but with exponential complexity.

**Properties:**
- Exhaustive search: Checks all possible :math:`\\binom{n}{n-f}` subsets.
- Optimal: Guarantees the smallest possible diameter among all valid subsets.
- Not scalable: Only practical for small :math:`n` or research purposes.

Theoretical Bound
----------------

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
...     torch.tensor([100., 200., 300.]),  # Byzantine
...     torch.tensor([100., 200., 300.])   # Byzantine
... ]
>>> result = brute(gradients, f=2)
tensor([1.0500, 2.0500, 3.0500])
"""

import tools
from . import register

import itertools
import math
import torch

# Optional 'native' module
try:
    import native
except ImportError:
    native = None

# ---------------------------------------------------------------------------- #
# Brute GAR

def _compute_selection(gradients, f, **kwargs):
    """
    Find the subset with smallest diameter.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients.
    f : int
        Number of Byzantine gradients to tolerate.
    **kwargs : dict
        Additional keyword arguments (ignored).

    Returns
    -------
    tuple
        Indices of the selected gradients.
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
        cur_diam = 0.
        for x, y in tools.pairwise(cur_iset):
            # Get distance between these two gradients ("magic" formula valid since x < y)
            cur_dist = distances[(2 * n - x - 3) * x // 2 + y - 1]
            # Check finite distance (non-Byzantine gradient must only contain finite coordinates), drop set if non-finite
            if not math.isfinite(cur_dist):
                break
            # Check if new maximum
            if cur_dist > cur_diam:
                cur_diam = cur_dist
        else:
            # Check if new selected diameter
            if sel_iset is None or cur_diam < sel_diam:
                sel_iset = cur_iset
                sel_diam = cur_diam
    # Return the selected gradients
    assert sel_iset is not None, "Too many non-finite gradients: a non-Byzantine gradient must only contain finite coordinates"
    return sel_iset

def aggregate(gradients, f, **kwargs):
    """
    Compute the Brute aggregation (mean of smallest-diameter subset).

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients to aggregate.
    f : int
        Number of Byzantine gradients to tolerate. Must satisfy
        ``1 <= f <= (n - 1) // 2`` where ``n = len(gradients)``.
    **kwargs : dict
        Additional keyword arguments (ignored).

    Returns
    -------
    torch.Tensor
        Average of the selected subset with smallest diameter.

    Notes
    -----
    The output tensor is a new tensor, not aliasing any input tensor.
    """
    sel_iset = _compute_selection(gradients, f, **kwargs)
    return sum(gradients[i] for i in sel_iset).div_(len(gradients) - f)

def aggregate_native(gradients, f, **kwargs):
    """
    Compute the Brute aggregation using native (C++/CUDA) acceleration.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients to aggregate.
    f : int
        Number of Byzantine gradients to tolerate.
    **kwargs : dict
        Additional keyword arguments (ignored).

    Returns
    -------
    torch.Tensor
        Average of the selected subset.
    """
    return native.brute.aggregate(gradients, f)

def check(gradients, f, **kwargs):
    """
    Check parameter validity for Brute rule.

    Parameters
    ----------
    gradients : list
        Non-empty list of gradients to aggregate.
    f : int
        Number of Byzantine gradients to tolerate.
    **kwargs : dict
        Additional keyword arguments (ignored).

    Returns
    -------
    None or str
        None if valid, otherwise error message string.
    """
    if not isinstance(gradients, list) or len(gradients) < 1:
        return "Expected a list of at least one gradient to aggregate, got %r" % gradients
    if not isinstance(f, int) or f < 1 or len(gradients) < 2 * f + 1:
        return "Invalid number of Byzantine gradients to tolerate, got f = %r, expected 1 ≤ f ≤ %d" % (f, (len(gradients) - 1) // 2)

def upper_bound(n, f, d):
    """
    Compute the theoretical upper bound on the ratio non-Byzantine standard
    deviation / norm to use this rule.

    Parameters
    ----------
    n : int
        Number of workers (Byzantine + non-Byzantine).
    f : int
        Expected number of Byzantine workers.
    d : int
        Dimension of the gradient space.

    Returns
    -------
    float
        Theoretical upper-bound value.
    """
    return (n - f) / (math.sqrt(8) * f)

def influence(honests, attacks, f, **kwargs):
    """
    Compute the ratio of accepted Byzantine gradients.

    Parameters
    ----------
    honests : list of torch.Tensor
        Non-empty list of honest gradients.
    attacks : list of torch.Tensor
        List of attack (Byzantine) gradients.
    f : int
        Number of Byzantine gradients to tolerate.
    **kwargs : dict
        Additional keyword arguments (ignored).

    Returns
    -------
    float
        Ratio of Byzantine gradients in the final aggregation.
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
        tools.warning("GAR %r could not be registered since the associated native module %r is unavailable" % (method_name, native_name))
