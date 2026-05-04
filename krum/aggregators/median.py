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

r"""Coordinate-wise median aggregation rule.

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

Example:
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
from . import Aggregator, AggregatorSpec, register_class

# Optional 'native' module
try:
    from krum import native
except ImportError:
    native = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------- #
# Coordinate-wise median GAR


def _upper_bound(n: int, f: int, d: int) -> float:
    return 1 / math.sqrt(n - f)


@register_class
class Median(Aggregator):
    """Coordinate-wise median aggregation rule."""

    spec = AggregatorSpec(
        name="median",
        aliases=("Median",),
        description="Coordinate-wise median of all submitted gradients.",
        upper_bound=_upper_bound,
    )

    def aggregate(self, gradients: list[torch.Tensor], **kwargs) -> torch.Tensor:
        """Compute the coordinate-wise median of all submitted gradients.

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

        Returns:
        -------
        torch.Tensor
            Coordinate-wise median of all input gradients.

        Notes:
        -----
        The returned tensor is newly computed and does not alias any input tensor.
        """
        return torch.stack(gradients).median(dim=0)[0]

    def check(self, gradients: list[torch.Tensor], **kwargs) -> str | None:
        """Check whether the median rule can be used with the given parameters.

        Parameters
        ----------
        gradients : list of torch.Tensor
            Non-empty list of gradients to aggregate.
        **kwargs : object
            Additional keyword arguments, accepted for compatibility with the GAR
            interface and ignored by this check.

        Returns:
        -------
        str or None
            ``None`` when parameters are valid, otherwise a user-facing error
            message.
        """
        if not isinstance(gradients, list) or len(gradients) < 1:
            return f"Expected a list of at least one gradient to aggregate, got {gradients!r}"
        return None

    @staticmethod
    def upper_bound(n: int, f: int, d: int) -> float:
        """Compute the theoretical coordinate-wise median robustness bound."""
        return _upper_bound(n, f, d)


def aggregate(gradients: list[torch.Tensor], **kwargs) -> torch.Tensor:
    """Compute the coordinate-wise median of all submitted gradients.

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

    Returns:
    -------
    torch.Tensor
        Coordinate-wise median of all input gradients.

    Notes:
    -----
    The returned tensor is newly computed and does not alias any input tensor.
    """
    return Median().aggregate(gradients=gradients, **kwargs)


def aggregate_native(gradients: list[torch.Tensor], **kwargs) -> torch.Tensor:
    """Compute the coordinate-wise median using native C++/CUDA acceleration.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients to aggregate.
    **kwargs : object
        Additional keyword arguments, accepted for compatibility with the GAR
        interface and ignored by this implementation.

    Returns:
    -------
    torch.Tensor
        Coordinate-wise median of all input gradients.
    """
    return native.median.aggregate(gradients)  # type: ignore[attr-defined]


def check(gradients: list[torch.Tensor], **kwargs) -> str | None:
    """Check whether the median rule can be used with the given parameters.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients to aggregate.
    **kwargs : object
        Additional keyword arguments, accepted for compatibility with the GAR
        interface and ignored by this check.

    Returns:
    -------
    str or None
        ``None`` when parameters are valid, otherwise a user-facing error
        message.
    """
    return Median().check(gradients=gradients, **kwargs)


def upper_bound(n: int, f: int, d: int) -> float:
    r"""Compute the theoretical coordinate-wise median robustness bound.

    Parameters
    ----------
    n : int
        Total number of workers, including Byzantine workers.
    f : int
        Expected number of Byzantine workers.
    d : int
        Gradient dimension. Accepted for compatibility with the GAR metadata
        interface; the current formula does not depend on it.

    Returns:
    -------
    float
        Upper bound on the ratio between non-Byzantine standard deviation and
        gradient norm.

    Notes:
    -----
    The bound formula is:

    .. math::

        \\frac{1}{\\sqrt{n - f}}
    """
    return Median.upper_bound(n, f, d)


# Register aggregation rule (native version, if available)
if native is not None:
    native_name = "median"
    method_name = "native-median"
    if native_name in dir(native):

        @register_class
        class NativeMedian(Median):
            """Native coordinate-wise median aggregation rule."""

            spec = AggregatorSpec(
                name=method_name,
                description="Native coordinate-wise median of all submitted gradients.",
                supports_native=True,
                upper_bound=_upper_bound,
            )

            def aggregate(self, gradients: list[torch.Tensor], **kwargs) -> torch.Tensor:
                """Compute the coordinate-wise median using native acceleration."""
                return aggregate_native(gradients=gradients, **kwargs)

    else:
        tools.warning(
            f"GAR {method_name!r} could not be registered since the associated native module {native_name!r} is unavailable"
        )
