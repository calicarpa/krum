# coding: utf-8
###
 # @file   average.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Simple arithmetic mean aggregation rule.
###

"""
This is the simplest aggregation rule, computing the arithmetic mean of all
submitted gradients. It serves as a baseline for comparison with Byzantine-
resilient methods.

Use Case
--------

Baseline for non-adversarial settings or when no Byzantine
behavior is expected.

Limitations
-----------

Vulnerable to any Byzantine attack. A single malicious gradient can completely
skew the result.

Example
-------

>>> import aggregators
>>> import torch
>>> from aggregators import average
>>> gradients = [torch.tensor([1., 2., 3.]), torch.tensor([4., 5., 6.])]
>>> result = average(gradients=gradients)
tensor([2.5000, 3.5000, 4.5000])
"""

from . import register

# ---------------------------------------------------------------------------- #
# Average GAR


def aggregate(gradients, **kwargs):
    """
    Compute the arithmetic mean of all submitted gradients.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Non-empty list of gradients to aggregate. Each gradient should be
        a 1-D tensor representing the flattened model parameters.
    **kwargs : object
        Additional keyword arguments, ignored by this rule.

    Returns
    -------
    torch.Tensor
        The arithmetic mean of all input gradients.

    Notes
    -----
    The output tensor is a new tensor, not aliasing any input tensor.
    """
    return sum(gradients) / len(gradients)


def check(gradients, **kwargs):
    """
    Check parameter validity for the averaging rule.

    Parameters
    ----------
    gradients : list
        Non-empty list of gradients to aggregate.
    **kwargs : object
        Additional keyword arguments, ignored by this rule.

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


def influence(honests, attacks, **kwargs):
    """
    Compute the ratio of accepted Byzantine gradients.

    For arithmetic mean, all submitted gradients are used in the aggregation,
    so the influence ratio is simply the fraction of Byzantine gradients
    in the total.

    Parameters
    ----------
    honests : list of torch.Tensor
        Non-empty list of honest gradients.
    attacks : list of torch.Tensor
        List of attack (Byzantine) gradients.
    **kwargs : object
        Additional keyword arguments, ignored by this rule.

    Returns
    -------
    float
        Ratio of Byzantine gradients in the aggregation (attackers / total).
    """
    return len(attacks) / (len(honests) + len(attacks))


# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule
register("average", aggregate, check, influence=influence)
