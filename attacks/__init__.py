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
Byzantine attack registry used to evaluate aggregation-rule robustness.

Each attack combines a keyword-only generation function with a validation
function. Registered attacks are loaded dynamically and exposed as module-level
callables.

Contract
--------

Each attack MUST:

1. Accept keyword-only arguments.
2. Accept the reserved parameter ``grad_honests`` (non-empty list of honest gradients).
3. Accept the reserved parameter ``f_decl`` (number of declared Byzantine gradients).
4. Accept the reserved parameter ``f_real`` (number of Byzantine gradients to generate).
5. Accept the reserved parameter ``model`` (model with configured defaults).
6. Accept the reserved parameter ``defense`` (aggregation rule to defeat).
7. Return exactly ``f_real`` tensors (list of Byzantine gradients).
8. NOT return tensors that alias any honest input tensor.
9. MAY reuse the same Byzantine tensor object when all generated gradients are identical.

Each attack MUST provide a ``check`` function that validates parameters and
returns ``None`` when valid, or a user-facing error message otherwise.

The module exposes three variants for each attack:

- ``attack``: The default version (checked in debug mode, unchecked in release)
- ``attack.checked``: Always validates parameters
- ``attack.unchecked``: Skips validation (faster in production)
"""

import pathlib

import torch

import tools

from typing import Callable

# ---------------------------------------------------------------------------- #
# Automated attack loader


def register(name: str, unchecked: Callable, check: Callable) -> None:
    """
    Register a Byzantine attack.

    Parameters
    ----------
    name : str
        User-visible attack name.
    unchecked : callable
        Attack implementation without parameter checks. It must return exactly
        ``f_real`` Byzantine gradients.
    check : callable
        Validation function associated with ``unchecked``. It must return
        ``None`` when parameters are valid, or an error message otherwise.

    Returns
    -------
    None
        The attack is registered as a module-level callable.
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
            raise tools.UserException(
                f"Attack {name!r} cannot be used with the given parameters: {message}"
            )
        # Attack
        res = unchecked(f_real=f_real, **kwargs)
        # Forward asserted return value
        assert isinstance(res, list) and len(res) == f_real, (
            f"Expected attack {name!r} to return a list of {f_real} Byzantine gradients, got {res!r}"
        )
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
