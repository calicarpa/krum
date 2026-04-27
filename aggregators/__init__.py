# coding: utf-8
###
 # @file   __init__.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Loading of the local modules.
###

"""
This module provides a collection of robust aggregation rules for distributed
learning. Each rule can tolerate a certain number of Byzantine (malicious)
workers while still converging correctly.

Contract
--------

Each aggregation rule MUST:

1. Accept keyword-only arguments
2. Accept the reserved parameter ``gradients`` (non-empty list of gradients)
3. Accept the reserved parameter ``f`` (number of Byzantine gradients to tolerate)
4. Accept the reserved parameter ``model`` (model with configured defaults)
5. NOT return a tensor that aliases any input tensor

Each rule MUST provide a ``check`` function that validates parameters and
returns ``None`` if valid, or an error message if invalid.

The module exposes three variants for each rule:

- ``rule``: The default version (checked in debug mode, unchecked in release)
- ``rule.checked``: Always validates parameters
- ``rule.unchecked``: Skips validation (faster in production)

Additional metadata available on each rule:

- ``rule.check``: The validation function
- ``rule.upper_bound``: Theoretical bound on stddev/norm ratio (if available)
- ``rule.influence``: Attack acceptance ratio (if available)

Available Rules
---------------

- **Average**: Simple arithmetic mean (baseline, no Byzantine resilience)
- **Median**: Coordinate-wise median (basic resilience against NaN)
- **Krum / Multi-Krum**: Distance-based selection (moderate resilience)
- **Bulyan**: Two-stage selection + trimmed mean (strong resilience)
- **Brute**: Exhaustive search for optimal subset (best theoretical)

Example
-------

.. code-block:: python

    import aggregators
    import torch

    gradients = [torch.randn(1000) for _ in range(10)]

    # Using Krum with f=2 Byzantine workers
    result = aggregators.krum(gradients=gradients, f=2, model=model)

    # Validate parameters first
    if aggregators.krum.check(gradients=gradients, f=2) is None:
        result = aggregators.krum(gradients=gradients, f=2)

    # Access theoretical bound
    bound = aggregators.krum.upper_bound(n=10, f=2, d=1000)
"""

import pathlib
import torch

import tools

# ---------------------------------------------------------------------------- #
# Automated GAR loader

def make_gar(unchecked, check, upper_bound=None, influence=None):
    """
    GAR wrapper helper.

    Parameters
    ----------
    unchecked : callable
        Associated function (see module description).
    check : callable
        Parameter validity check function.
    upper_bound : callable, optional
        Compute the theoretical upper bound on the ratio non-Byzantine standard
        deviation / norm to use this aggregation rule: (n, f, d) -> float.
    influence : callable, optional
        Attack acceptance ratio function.

    Returns
    -------
    callable
        Wrapped GAR.
    """
    # Closure wrapping the call with checks
    def checked(**kwargs):
        # Check parameter validity
        message = check(**kwargs)
        if message is not None:
            raise tools.UserException("Aggregation rule %r cannot be used with the given parameters: %s" % (name, message))
        # Aggregation (hard to assert return value, duck-typing is allowed...)
        return unchecked(**kwargs)
    # Select which function to call by default
    func = checked if __debug__ else unchecked
    # Bind all the (sub) functions to the selected function
    setattr(func, "check", check)
    setattr(func, "checked", checked)
    setattr(func, "unchecked", unchecked)
    setattr(func, "upper_bound", upper_bound)
    setattr(func, "influence", influence)
    # Return the selected function with the associated name
    return func

def register(name, unchecked, check, upper_bound=None, influence=None):
    """
    Simple registration-wrapper helper.

    Parameters
    ----------
    name : str
        GAR name.
    unchecked : callable
        Associated function (see module description).
    check : callable
        Parameter validity check function.
    upper_bound : callable, optional
        Compute the theoretical upper bound.
    influence : callable, optional
        Attack acceptance ratio function.
    """
    global gars
    # Check if name already in use
    if name in gars:
        tools.warning("Unable to register %r GAR: name already in use" % name)
        return
    # Export the selected function with the associated name
    gars[name] = make_gar(unchecked, check, upper_bound=upper_bound, influence=influence)

# Registered rules (mapping name -> aggregation rule)
gars = dict()

# Load all local modules
with tools.Context("aggregators", None):
    tools.import_directory(pathlib.Path(__file__).parent, globals())

# Bind/overwrite the GAR name with the associated rules in globals()
for name, rule in gars.items():
    globals()[name] = rule
