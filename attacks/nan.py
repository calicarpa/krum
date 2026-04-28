# coding: utf-8
###
 # @file   nan.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Attack that generates NaN gradient(s), hence the name.
###

"""
This attack generates gradients containing NaN (Not a Number) values. The goal
is to disrupt training by introducing non-finite values into the aggregation
process.

Use Case
--------

Simple baseline attack to test the robustness of aggregation
rules against non-finite values.

Properties
----------

- Simple: Generates gradients filled with NaN values
- Effective against naive aggregators
- Mitigated by coordinate-wise median and other robust methods

Example
-------

>>> import torch
>>> from attacks import nan
>>> honest_grads = [torch.tensor([1., 2., 3.]), torch.tensor([4., 5., 6.])]
>>> byzantine_grads = nan(honest_grads=honest_grads, f_decl=1, f_real=1, model=None)
>>> print(byzantine_grads[0])
tensor([nan, nan, nan])
"""

import math
import torch

from . import register

# ---------------------------------------------------------------------------- #
# Non-finite gradient attack

def attack(grad_honests, f_real, **kwargs):
    """
    Generate non-finite gradients.

    Parameters
    ----------
    grad_honests : list of torch.Tensor
        Non-empty list of honest gradients.
    f_real : int
        Number of Byzantine gradients to generate.
    **kwargs : dict
        Additional keyword arguments (ignored).

    Returns
    -------
    list of torch.Tensor
        List of Byzantine gradients containing NaN values.
    """
    # Fast path
    if f_real == 0:
        return list()
    # Generate the non-finite Byzantine gradient
    byz_grad = torch.empty_like(grad_honests[0])
    byz_grad.copy_(torch.tensor((math.nan,), dtype=byz_grad.dtype))
    # Return this Byzantine gradient 'f_real' times
    return [byz_grad] * f_real

def check(grad_honests, f_real, **kwargs):
    """
    Check parameter validity for this attack.

    Parameters
    ----------
    grad_honests : list
        Non-empty list of honest gradients.
    f_real : int
        Number of Byzantine gradients to generate.
    **kwargs : dict
        Additional keyword arguments (ignored).

    Returns
    -------
    None or str
        None if valid, otherwise error message string.
    """
    if not isinstance(grad_honests, list) or len(grad_honests) == 0:
        return "Expected a non-empty list of honest gradients, got %r" % (grad_honests,)
    if not isinstance(f_real, int) or f_real < 0:
        return "Expected a non-negative number of Byzantine gradients to generate, got %r" % (f_real,)

# ---------------------------------------------------------------------------- #
# Attack registering

# Register the attack
register("nan", attack, check)
