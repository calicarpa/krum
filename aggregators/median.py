# coding: utf-8
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
 # NaN-resilient, coordinate-wise median GAR.
###

"""
This aggregation rule computes the median of each coordinate independently across
all submitted gradients. It is robust against Byzantine gradients containing
NaN or Inf values (they are simply ignored in the median computation).

**Use case:** Simple defense against attacks that inject NaN/Inf values, or
as a baseline for coordinate-wise robust aggregation.

**Properties:**
- NaN-resilient: Gradients containing NaN values are naturally ignored.
- Coordinate-wise: Each dimension is treated independently.
- No theoretical guarantee: Unlike other methods, this rule has no proven
  Byzantine-resilience guarantees for general attacks.

Theoretical Bound
-----------------

The coordinate-wise median provides guarantees when the ratio of non-Byzantine
standard deviation to gradient norm is below:

.. math::

    \\frac{1}{\\sqrt{n - f}}

where :math:`n` is the total number of workers and :math:`f` is the number
of Byzantine workers.


Example
-------

>>> import torch
>>> from aggregators import median
>>> gradients = [
...     torch.tensor([1., 100., 3.]),
...     torch.tensor([2., 200., 4.]),
...     torch.tensor([3., 300., 5.])
... ]
>>> result = median(gradients)
tensor([2., 200., 4.])
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
# NaN-resilient, coordinate-wise median GAR

def aggregate(gradients, **kwargs):
    """
    Compute the coordinate-wise median of all gradients.

    This method is resilient to NaN and Inf values in the input gradients,
    as the median operation naturally ignores such values.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients to aggregate. Each gradient should be
        a 1-D tensor of the same shape.
    **kwargs : dict
        Additional keyword arguments (ignored).

    Returns
    -------
    torch.Tensor
        Coordinate-wise median of all input gradients. NaN and Inf values
        are naturally excluded from the median computation.

    Notes
    -----
    The output tensor is a new tensor, not aliasing any input tensor.
    """
    return torch.stack(gradients).median(dim=0)[0]

def aggregate_native(gradients, **kwargs):
    """
    Compute the coordinate-wise median using native (C++/CUDA) acceleration.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients to aggregate.
    **kwargs : dict
        Additional keyword arguments (ignored).

    Returns
    -------
    torch.Tensor
        Coordinate-wise median of all input gradients.
    """
    return native.median.aggregate(gradients)

def check(gradients, **kwargs):
    """
    Check parameter validity for the median rule.

    Parameters
    ----------
    gradients : list
        Non-empty list of gradients to aggregate.
    **kwargs : dict
        Additional keyword arguments (ignored).

    Returns
    -------
    None or str
        None if valid, otherwise error message string.
    """
    if not isinstance(gradients, list) or len(gradients) < 1:
        return "Expected a list of at least one gradient to aggregate, got %r" % gradients

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

    Notes
    -----
    The bound formula is:

    .. math::

        \\frac{1}{\\sqrt{n - f}}

    This bound applies under the assumption that Byzantine gradients cannot
    influence the median more than this ratio.
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
        tools.warning("GAR %r could not be registered since the associated native module %r is unavailable" % (method_name, native_name))
