# coding: utf-8
###
 # @file   __init__.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Loading of the local modules.
###

"""
This module provides a collection of Byzantine attack strategies for testing
the robustness of distributed learning algorithms.

Contract
--------

Each attack MUST:

1. Accept keyword-only arguments
2. Accept the reserved parameter ``grad_honests`` (non-empty list of honest gradients)
3. Accept the reserved parameter ``f_decl`` (number of declared Byzantine gradients)
4. Accept the reserved parameter ``f_real`` (number of Byzantine gradients to generate)
5. Accept the reserved parameter ``model`` (model with configured defaults)
6. Accept the reserved parameter ``defense`` (aggregation rule to defeat)
7. Return exactly ``f_real`` tensors (list of Byzantine gradients)
8. NOT return tensors that alias any input tensor

Each attack MUST provide a ``check`` function that validates parameters and
returns ``None`` if valid, or an error message if invalid.

The module exposes three variants for each attack:

- ``attack``: The default version (checked in debug mode, unchecked in release)
- ``attack.checked``: Always validates parameters
- ``attack.unchecked``: Skips validation (faster in production)

Example
-------

.. code-block:: python

    import attacks
    import torch

    honest_grads = [torch.randn(1000) for _ in range(10)]

    # Using NaN attack
    byzantine = attacks.nan(
        grad_honests=honest_grads,
        f_decl=2,
        f_real=2,
        model=model
    )

    # Using Little attack with custom factor
    byzantine = attacks.little(
        grad_honests=honest_grads,
        f_decl=2,
        f_real=2,
        defense=aggregator,
        model=model,
        factor=1.5
    )

    # Validate parameters first
    if attacks.nan.check(grad_honests=honest_grads, f_real=2) is None:
        byzantine = attacks.nan(grad_honests=honest_grads, f_decl=2, f_real=2, model=model)
"""

import pathlib
import torch

import tools

# ---------------------------------------------------------------------------- #
# Automated attack loader

def register(name, unchecked, check):
    """
    Simple registration-wrapper helper.

    Parameters
    ----------
    name : str
        Attack name.
    unchecked : callable
        Associated function (see module description).
    check : callable
        Parameter validity check function.
    """
    global attacks
    # Check if name already in use
    if name in attacks:
        tools.warning(f"Unable to register {name!r} attack: name already in use")
        return
    # Closure wrapping the call with checks
    def checked(f_real, **kwargs):
        # Check parameter validity
        message = check(f_real=f_real, **kwargs)
        if message is not None:
            raise tools.UserException(f"Attack {name!r} cannot be used with the given parameters: {message}")
        # Attack
        res = unchecked(f_real=f_real, **kwargs)
        # Forward asserted return value
        assert isinstance(res, list) and len(res) == f_real, f"Expected attack {name!r} to return a list of {f_real} Byzantine gradients, got {res!r}"
        return res
    # Select which function to call by default
    func = checked if __debug__ else unchecked
    # Bind all the (sub) functions to the selected function
    setattr(func, "check", check)
    setattr(func, "checked", checked)
    setattr(func, "unchecked", unchecked)
    # Export the selected function with the associated name
    attacks[name] = func

# Registered attacks (mapping name -> attack)
attacks = dict()

# Load native and all local modules
with tools.Context("attacks", None):
    tools.import_directory(pathlib.Path(__file__).parent, globals())

# Bind/overwrite the attack names with the associated attacks in globals()
for name, attack in attacks.items():
    globals()[name] = attack
