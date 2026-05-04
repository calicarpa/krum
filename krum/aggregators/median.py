###
# @file   median.py
# @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
#
# @section LICENSE
#
# Copyright © 2018-2020 École Polytechnique Fédérale de Lausanne (EPFL).
# See LICENSE file.
#
# @section DESCRIPTION
#
# Coordinate-wise median GAR.
###

"""
Coordinate-wise median aggregation rule.

This rule computes the median of each coordinate independently across all
submitted gradients. It delegates to ``torch.median`` and does not filter
non-finite values before aggregation. NaN values may propagate, while Inf values
participate in the coordinate ordering.

Use Case
--------
Baseline coordinate-wise robust aggregation when input gradients are expected to
be finite.

Properties
----------
- Coordinate-wise: each dimension is treated independently.
- Non-finite values are not filtered before aggregation.
- Theoretical bound available through :func:`upper_bound`.

Theoretical Bound
-----------------
The coordinate-wise median provides guarantees when the ratio of non-Byzantine
standard deviation to gradient norm is below:

.. math::

    \\frac{1}{\\sqrt{n - f}}

where:

- :math:`n` is the total number of workers.
- :math:`f` is the number of Byzantine workers.

Example
-------
>>> import torch
>>> from aggregators import median
>>> gradients = [
...     torch.tensor([1., 100., 3.]),
...     torch.tensor([2., 200., 4.]),
...     torch.tensor([3., 300., 5.]),
... ]
>>> result = median(gradients=gradients)
>>> result
tensor([2., 200., 4.])
"""

import math

import torch

from .. import tools
from . import register

# Optional 'native' module
try:
    import native
except ImportError:
    native = None

# ---------------------------------------------------------------------------- #
# Coordinate-wise median GAR


def aggregate(gradients: list[torch.Tensor], **kwargs) -> torch.Tensor:
    """
    Compute the coordinate-wise median of all submitted gradients.

    This method delegates to ``torch.median`` and does not filter non-finite
    values before aggregation. NaN values may propagate, while Inf values
    participate in the coordinate ordering.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients to aggregate. Each gradient should be a
        1-D tensor with the same shape, dtype, and device as the others.
    **kwargs : object
        Additional keyword arguments, accepted for compatibility with the GAR
        interface and ignored by this implementation.

    Returns
    -------
    torch.Tensor
        Coordinate-wise median of all input gradients.

    Notes
    -----
    The returned tensor is newly computed and does not alias any input tensor.
    """
    return torch.stack(gradients).median(dim=0)[0]


def aggregate_native(gradients: list[torch.Tensor], **kwargs) -> torch.Tensor:
    """
    Compute the coordinate-wise median using native C++/CUDA acceleration.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients to aggregate.
    **kwargs : object
        Additional keyword arguments, accepted for compatibility with the GAR
        interface and ignored by this implementation.

    Returns
    -------
    torch.Tensor
        Coordinate-wise median of all input gradients.
    """
    return native.median.aggregate(gradients)


def check(gradients: list[torch.Tensor], **kwargs) -> str | None:
    """
    Check whether the median rule can be used with the given parameters.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients to aggregate.
    **kwargs : object
        Additional keyword arguments, accepted for compatibility with the GAR
        interface and ignored by this check.

    Returns
    -------
    str or None
        ``None`` when parameters are valid, otherwise a user-facing error
        message.
    """
    if not isinstance(gradients, list) or len(gradients) < 1:
        return f"Expected a list of at least one gradient to aggregate, got {gradients!r}"
    return None


def upper_bound(n: int, f: int, d: int) -> float:
    """
    Compute the theoretical coordinate-wise median robustness bound.

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
        gradient norm.

    Notes
    -----
    The bound formula is:

    .. math::

        \\frac{1}{\\sqrt{n - f}}
    """
    return 1 / math.sqrt(n - f)


# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (pytorch version)
method_name = "median"
register(method_name, aggregate, check, upper_bound)

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
