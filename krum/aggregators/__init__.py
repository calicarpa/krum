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
Gradient aggregation rules (GARs) for Byzantine-resilient distributed learning.

Each rule combines a keyword-only aggregation function with a validation
function and optional metadata used by the training and experiment scripts.

Contract
--------

Each aggregation rule MUST:

1. Accept keyword-only arguments
2. Accept the reserved parameter ``gradients`` (non-empty list of gradients)
3. Accept the reserved parameter ``f`` (number of Byzantine gradients to tolerate)
4. Accept the reserved parameter ``model`` (model with configured defaults)
5. NOT return a tensor that aliases any input tensor

Each rule MUST provide a ``check`` function that validates parameters and
returns ``None`` when valid, or a user-facing error message otherwise.

The module exposes three variants for each rule:

- ``rule``: The default version (checked in debug mode, unchecked in release)
- ``rule.checked``: Always validates parameters
- ``rule.unchecked``: Skips validation (faster in production)

Additional metadata available on each rule:

- ``rule.check``: The validation function
- ``rule.upper_bound``: Theoretical bound on stddev/norm ratio (if available)
- ``rule.influence``: Attack acceptance ratio (if available)
"""

import pathlib
from collections.abc import Callable
from typing import Any, cast

import torch

from .. import tools

# ---------------------------------------------------------------------------- #
# Automated GAR loader


def make_gar(
    unchecked: Callable,
    check: Callable,
    upper_bound: Callable | None = None,
    influence: Callable | None = None,
) -> Callable:
    """
    Wrap an unchecked GAR with validation and metadata.

    Parameters
    ----------
    unchecked : callable
        Aggregation function implementing the rule without parameter checks.
    check : callable
        Validation function. It must return ``None`` when parameters are valid,
        or an error message otherwise.
    upper_bound : callable, optional
        Function computing the theoretical upper bound on the ratio between
        non-Byzantine standard deviation and gradient norm. The expected
        signature is ``(n, f, d) -> float``.
    influence : callable, optional
        Function computing the accepted Byzantine-gradient ratio for a given
        set of honest and attack gradients.

    Returns
    -------
    callable
        Checked or unchecked GAR selected according to ``__debug__``. The
        returned callable is annotated with ``check``, ``checked``,
        ``unchecked``, ``upper_bound``, and ``influence`` attributes.
    """

    # Closure wrapping the call with checks
    def checked(**kwargs):
        # Check parameter validity
        message = check(**kwargs)
        if message is not None:
            raise tools.UserException(f"Aggregation rule {name!r} cannot be used with the given parameters: {message}")
        # Aggregation (hard to assert return value, duck-typing is allowed...)
        return unchecked(**kwargs)

    # Select which function to call by default
    func = cast(Any, checked if __debug__ else unchecked)
    # Bind all the (sub) functions to the selected function
    func.check = check
    func.checked = checked
    func.unchecked = unchecked
    func.upper_bound = upper_bound
    func.influence = influence
    # Return the selected function with the associated name
    return func


def register(
    name: str,
    unchecked: Callable,
    check: Callable,
    upper_bound: Callable | None = None,
    influence: Callable | None = None,
) -> None:
    """
    Register a gradient aggregation rule.

    Parameters
    ----------
    name : str
        User-visible GAR name.
    unchecked : callable
        Aggregation function implementing the rule without parameter checks.
    check : callable
        Validation function associated with ``unchecked``.
    upper_bound : callable, optional
        Function computing the rule's theoretical upper bound.
    influence : callable, optional
        Function computing the accepted Byzantine-gradient ratio.
    """
    global gars
    # Check if name already in use
    if name in gars:
        tools.warning(f"Unable to register {name!r} GAR: name already in use")
        return
    # Export the selected function with the associated name
    gars[name] = make_gar(unchecked, check, upper_bound=upper_bound, influence=influence)


# Registered rules (mapping name -> aggregation rule)
gars = {}

# Load all local modules
with tools.Context("aggregators", None):
    tools.import_directory(pathlib.Path(__file__).parent, globals())

# Bind/overwrite the GAR name with the associated rules in globals()
for name, rule in gars.items():
    globals()[name] = rule
