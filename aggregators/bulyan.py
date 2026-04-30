# coding: utf-8
###
# @file   bulyan.py
# @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
#
# @section LICENSE
#
# Copyright © 2018-2020 École Polytechnique Fédérale de Lausanne (EPFL).
# See LICENSE file.
#
# @section DESCRIPTION
#
# Bulyan over Multi-Krum GAR.
###

"""
Bulyan aggregation rule built on top of Multi-Krum.

Bulyan combines distance-based gradient selection with coordinate-wise
robust averaging. It first selects a candidate set using a Multi-Krum-like
criterion, then aggregates each coordinate from the values closest to the
coordinate-wise median.

Use Case
--------

Use Bulyan when stronger Byzantine resilience is needed than plain Multi-Krum
can provide, and when the worker count is high enough to satisfy the stricter
``n >= 4f + 3`` requirement.

Properties
----------

- Two-stage aggregation: geometric selection then coordinate-wise averaging.
- Requires at least :math:`4f + 3` submitted gradients.
- Uses only newly allocated output tensors and does not return aliases of input
  gradients.
- Theoretical bound available through :func:`upper_bound`.

Algorithm
---------

1. Select candidate gradients with the smallest Multi-Krum scores.
2. For each coordinate, compute the median over the selected candidates.
3. Average the values closest to that median.

Complexity
----------

- Time: :math:`O(n^2 \\cdot d)` where :math:`n` is the number of gradients and
  :math:`d` is the gradient dimension.
- Space: :math:`O(n^2)` for storing pairwise distances.

Parameters
----------
m : int, optional
    Number of gradients to consider in each Multi-Krum selection step. Defaults
    to ``n - f - 2``. Must satisfy ``1 <= m <= n - f - 2``.

Example
-------

>>> import torch
>>> from aggregators import bulyan
>>> gradients = [
...     torch.tensor([1., 2., 3.]),
...     torch.tensor([1.1, 2.1, 3.1]),
...     torch.tensor([0.9, 1.9, 2.9]),
...     torch.tensor([1.2, 2.2, 3.2]),
...     torch.tensor([0.8, 1.8, 2.8]),
...     torch.tensor([1.05, 2.05, 3.05]),
...     torch.tensor([100., 200., 300.]),  # Byzantine
... ]
>>> result = bulyan(gradients=gradients, f=1)
tensor([1., 2., 3.])
"""

import math

import torch

import tools

from . import register

# Optional 'native' module
try:
    import native
except ImportError:
    native = None

# ---------------------------------------------------------------------------- #
# Bulyan GAR class


def aggregate(gradients: list[torch.Tensor], f: int, m=None, **kwargs) -> torch.Tensor:
    """
    Compute the Bulyan aggregate.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of flattened gradients to aggregate. All tensors must
        have the same shape, dtype, and device.
    f : int
        Number of Byzantine gradients to tolerate. Must satisfy
        ``1 <= f <= (n - 3) // 4`` where ``n = len(gradients)``.
    m : int, optional
        Number of nearest gradients considered in each Multi-Krum selection
        step. Defaults to ``n - f - 2``.
    **kwargs : object
        Additional keyword arguments. They are accepted for compatibility with
        the GAR interface and ignored by this implementation.

    Returns
    -------
    torch.Tensor
        Bulyan-aggregated gradient.

    Notes
    -----
    The returned tensor is newly allocated and does not alias any input tensor.
    """
    n = len(gradients)
    d = gradients[0].shape[0]
    # Defaults
    m_max = n - f - 2
    if m is None:
        m = m_max
    # Compute all pairwise distances
    distances = list([(math.inf, None)] * n for _ in range(n))
    for gid_x, gid_y in tools.pairwise(tuple(range(n))):
        dist = gradients[gid_x].sub(gradients[gid_y]).norm().item()
        if not math.isfinite(dist):
            dist = math.inf
        distances[gid_x][gid_y] = (dist, gid_y)
        distances[gid_y][gid_x] = (dist, gid_x)
    # Compute the scores
    scores = [None] * n
    for gid in range(n):
        dists = distances[gid]
        dists.sort(key=lambda x: x[0])
        dists = dists[:m]
        scores[gid] = (sum(dist for dist, _ in dists), gid)
        distances[gid] = dict(dists)
    # Selection loop
    selected = torch.empty(
        n - 2 * f - 2, d, dtype=gradients[0].dtype, device=gradients[0].device
    )
    for i in range(selected.shape[0]):
        # Update 'm'
        m = min(m, m_max - i)
        # Compute the average of the selected gradients
        scores.sort(key=lambda x: x[0])
        selected[i] = sum(gradients[gid] for _, gid in scores[:m]).div_(m)
        # Remove the gradient from the distances and scores
        gid_prune = scores[0][1]
        scores[0] = (math.inf, None)
        for score, gid in scores[1:]:
            if gid == gid_prune:
                scores[gid] = (score - distance[gid][gid_prune], gid)
    # Coordinate-wise averaged median
    m = selected.shape[0] - 2 * f
    median = selected.median(dim=0).values
    closests = (
        selected.clone()
        .sub_(median)
        .abs_()
        .topk(m, dim=0, largest=False, sorted=False)
        .indices
    )
    closests.mul_(d).add_(
        torch.arange(0, d, dtype=closests.dtype, device=closests.device)
    )
    avgmed = selected.take(closests).mean(dim=0)
    # Return resulting gradient
    return avgmed


def aggregate_native(
    gradients: list[torch.Tensor], f: int, m=None, **kwargs
) -> torch.Tensor:
    """
    Compute the Bulyan aggregate using native C++/CUDA acceleration.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of flattened gradients to aggregate.
    f : int
        Number of Byzantine gradients to tolerate.
    m : int, optional
        Number of nearest gradients considered in each Multi-Krum selection
        step. Defaults to ``n - f - 2``.
    **kwargs : object
        Additional keyword arguments. They are accepted for compatibility with
        the GAR interface and ignored by this implementation.

    Returns
    -------
    torch.Tensor
        Bulyan-aggregated gradient.
    """
    # Defaults
    if m is None:
        m = len(gradients) - f - 2
    # Computation
    return native.bulyan.aggregate(gradients, f, m)


def check(gradients: list[torch.Tensor], f: int, m=None, **kwargs) -> str | None:
    """
    Check whether the Bulyan parameters satisfy the GAR contract.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients to aggregate.
    f : int
        Number of Byzantine gradients to tolerate.
    m : int, optional
        Number of nearest gradients considered in each Multi-Krum selection
        step. If provided, must satisfy ``1 <= m <= n - f - 2``.
    **kwargs : object
        Additional keyword arguments. They are accepted for compatibility with
        the GAR interface and ignored by this check.

    Returns
    -------
    str or None
        ``None`` when parameters are valid, otherwise a user-facing error
        message.
    """
    if not isinstance(gradients, list) or len(gradients) < 1:
        return (
            "Expected a list of at least one gradient to aggregate, got %r" % gradients
        )
    if not isinstance(f, int) or f < 1 or len(gradients) < 4 * f + 3:
        return (
            "Invalid number of Byzantine gradients to tolerate, got f = %r, expected 1 ≤ f ≤ %d"
            % (f, (len(gradients) - 3) // 4)
        )
    if m is not None and (
        not isinstance(m, int) or m < 1 or m > len(gradients) - f - 2
    ):
        return (
            "Invalid number of selected gradients, got m = %r, expected 1 ≤ m ≤ %d"
            % (f, len(gradients) - f - 2)
        )


def upper_bound(n: int, f: int, d: int) -> float:
    """
    Compute Bulyan's theoretical resilience upper bound.

    Parameters
    ----------
    n : int
        Total number of workers, including Byzantine workers.
    f : int
        Expected number of Byzantine workers.
    d : int
        Gradient dimension. Accepted for compatibility with the GAR metadata
        interface; the current formula does not depend on it.

    Returns
    -------
    float
        Upper bound on the ratio between non-Byzantine standard deviation and
        gradient norm under the Bulyan assumptions.
    """
    return 1 / math.sqrt(2 * (n - f + f * (n + f * (n - f - 2) - 2) / (n - 2 * f - 2)))


# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (pytorch version)
method_name = "bulyan"
register(method_name, aggregate, check, upper_bound=upper_bound)

# Register aggregation rule (native version, if available)
if native is not None:
    native_name = method_name
    method_name = "native-" + method_name
    if native_name in dir(native):
        register(method_name, aggregate_native, check, upper_bound=upper_bound)
    else:
        tools.warning(
            "GAR %r could not be registered since the associated native module %r is unavailable"
            % (method_name, native_name)
        )
