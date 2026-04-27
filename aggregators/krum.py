# coding: utf-8
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

"""
Krum (and its variant Multi-Krum) is a distance-based Byzantine-resilient
aggregation rule. It works by:

1. Computing pairwise distances between all gradients
2. For each gradient, summing the distances to its :math:`n - f - 1` nearest
   neighbors (where :math:`f` is the number of Byzantine workers)
3. Selecting the :math:`m` gradients with the smallest total distances
4. Averaging the selected gradients

The intuition is that honest gradients will be close to each other, while
Byzantine gradients will be far from the honest majority.

**Use case:** General Byzantine-resilient aggregation when the attack model
assumes that Byzantine gradients are significantly different from honest ones.

**Properties:**
- Distance-based: Relies on geometric properties of gradients.
- Selects subset: Not all gradients are used in the final average.
- Theoretical guarantee: Provides bounds under certain assumptions on the
  Byzantine attack.

Theoretical Bound
-----------------

The Multi-Krum rule provides guarantees when:

.. math::

    \\frac{\\sigma}{\\|g\\|} \\leq \\frac{1}{\\sqrt{2 (n - f + \\frac{f(n + f(n - f - 2) - 2)}{n - 2f - 2})}}

where:

- :math:`n` is the total number of workers
- :math:`f` is the number of Byzantine workers
- :math:`\\sigma` is the standard deviation of honest gradients
- :math:`\\|g\\|` is the norm of the honest gradient

Parameters
----------

m : int, optional
    Number of gradients to select for averaging. Default is ``n - f - 2``.
    Must satisfy ``1 <= m <= n - f - 2``.

Example
-------

>>> import torch
>>> from aggregators import krum
>>> gradients = [
...     torch.tensor([1., 2., 3.]),
...     torch.tensor([1.1, 2.1, 3.1]),  # close to first
...     torch.tensor([100., 200., 300.])  # far (Byzantine)
... ]
>>> result = krum(gradients, f=1, m=2)
tensor([1.0500, 2.0500, 3.0500])
"""

import tools
from . import register

import math
import torch

# Optional 'native' module
try:
    import native
except ImportError:
    native = None

# ---------------------------------------------------------------------------- #
# Multi-Krum GAR

def _compute_scores(gradients, f, m, **kwargs):
    """
    Compute Multi-Krum scores for all gradients.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients.
    f : int
        Number of Byzantine gradients to tolerate.
    m : int
        Number of gradients to select.
    **kwargs : dict
        Additional keyword arguments (ignored).

    Returns
    -------
    list of tuple
        List of (score, gradient) sorted by increasing scores.
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
    scores = list()
    for i in range(n):
        # Collect the distances
        grad_dists = list()
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

def aggregate(gradients, f, m=None, **kwargs):
    """
    Compute the Multi-Krum aggregation.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients to aggregate.
    f : int
        Number of Byzantine gradients to tolerate. Must satisfy
        ``1 <= f <= (n - 3) // 2`` where ``n = len(gradients)``.
    m : int, optional
        Number of gradients to select for averaging. Default is
        ``n - f - 2``. Must satisfy ``1 <= m <= n - f - 2``.
    **kwargs : dict
        Additional keyword arguments (ignored).

    Returns
    -------
    torch.Tensor
        Average of the selected ``m`` gradients with smallest Krum scores.

    Notes
    -----
    The output tensor is a new tensor, not aliasing any input tensor.
    """
    # Defaults
    if m is None:
        m = len(gradients) - f - 2
    # Compute aggregated gradient
    scores = _compute_scores(gradients, f, m, **kwargs)
    return sum(grad for _, grad in scores[:m]).div_(m)

def aggregate_native(gradients, f, m=None, **kwargs):
    """
    Compute the Multi-Krum aggregation using native (C++/CUDA) acceleration.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients to aggregate.
    f : int
        Number of Byzantine gradients to tolerate.
    m : int, optional
        Number of gradients to select.
    **kwargs : dict
        Additional keyword arguments (ignored).

    Returns
    -------
    torch.Tensor
        Average of the selected gradients.
    """
    # Defaults
    if m is None:
        m = len(gradients) - f - 2
    # Computation
    return native.krum.aggregate(gradients, f, m)

def check(gradients, f, m=None, **kwargs):
    """
    Check parameter validity for Multi-Krum rule.

    Parameters
    ----------
    gradients : list
        Non-empty list of gradients to aggregate.
    f : int
        Number of Byzantine gradients to tolerate.
    m : int, optional
        Number of gradients to select.
    **kwargs : dict
        Additional keyword arguments (ignored).

    Returns
    -------
    None or str
        None if valid, otherwise error message string.
    """
    if not isinstance(gradients, list) or len(gradients) < 1:
        return "Expected a list of at least one gradient to aggregate, got %r" % gradients
    if not isinstance(f, int) or f < 1 or len(gradients) < 2 * f + 3:
        return "Invalid number of Byzantine gradients to tolerate, got f = %r, expected 1 ≤ f ≤ %d" % (f, (len(gradients) - 3) // 2)
    if m is not None and (not isinstance(m, int) or m < 1 or m > len(gradients) - f - 2):
        return "Invalid number of selected gradients, got m = %r, expected 1 ≤ m ≤ %d" % (m, len(gradients) - f - 2)

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
    return 1 / math.sqrt(2 * (n - f + f * (n + f * (n - f - 2) - 2) / (n - 2 * f - 2)))

def influence(honests, attacks, f, m=None, **kwargs):
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
    m : int, optional
        Number of gradients to select.
    **kwargs : dict
        Additional keyword arguments (ignored).

    Returns
    -------
    float
        Ratio of Byzantine gradients in the final aggregation.
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

# Register aggregation rule (native version, if available)
if native is not None:
  native_name = method_name
  method_name = "native-" + method_name
  if native_name in dir(native):
    register(method_name, aggregate_native, check, upper_bound)
  else:
    tools.warning("GAR %r could not be registered since the associated native module %r is unavailable" % (method_name, native_name))
