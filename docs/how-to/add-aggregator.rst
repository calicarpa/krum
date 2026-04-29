How to add a new aggregator
===========================

This guide walks you through creating and registering a new
Byzantine-resilient gradient aggregation rule (GAR).

Step 1 — Create the module
--------------------------

Create a new Python file in ``aggregators/``. The file is auto-discovered at
import time — no ``__init__.py`` edit is needed.

.. code-block:: text

   aggregators/
   ├── krum.py
   ├── median.py
   └── my_aggregator.py      ← your new file

Step 2 — Implement the aggregation function
-------------------------------------------

The aggregation function receives a list of gradient tensors and must return a
**single** tensor of the same shape. It must **never** return a tensor that
aliases an input.

.. code-block:: python

   # aggregators/my_aggregator.py

   import torch
   from . import register


   def aggregate(gradients, f, **kwargs):
       """Trimmed-mean aggregation: drop the *f* largest and smallest,
       then average the remaining gradients."""
       stacked = torch.stack(gradients)
       n = len(gradients)
       # Sort by L2 norm, drop the *f* extremes on each side
       norms = stacked.flatten(1).norm(dim=1)
       keep = norms.argsort()[f : n - f]
       return stacked[keep].mean(dim=0)

Step 3 — Write the ``check`` function
--------------------------------------

The ``check`` function validates parameters **before** the aggregation runs.
Return ``None`` when everything is valid, or a human-readable error string
otherwise.

.. code-block:: python

   def check(gradients, f, **kwargs):
       if not isinstance(gradients, list) or len(gradients) < 1:
           return "Expected a non-empty list of gradients, got %r" % gradients
       if not isinstance(f, int) or f < 1 or len(gradients) < 2 * f + 1:
           return (
               "Invalid f = %r; need 1 <= f <= %d"
               % (f, (len(gradients) - 1) // 2)
           )

Step 4 — Register the rule
--------------------------

Call :func:`aggregators.register` at module level. The function accepts:

- ``name`` — unique identifier used on the CLI (e.g. ``--gar my_agg``)
- ``unchecked`` — the aggregation function
- ``check`` — the validation function
- ``upper_bound`` *(optional)* — theoretical robustness bound
- ``influence`` *(optional)* — attack acceptance ratio

.. code-block:: python

   def upper_bound(n, f, d):
       """Return the theoretical upper bound on stddev/norm ratio."""
       return 1 / (n - 2 * f)


   register("my_agg", aggregate, check, upper_bound)

Step 5 — Use it
---------------

.. code-block:: bash

   python -m experiments --gar my_agg --gar-args f:2 ...

Contract summary
----------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``gradients``
     - ``list[Tensor]``
     - Non-empty list of honest + Byzantine gradients
   * - ``f``
     - ``int``
     - Number of Byzantine gradients to tolerate
   * - ``model``
     - ``nn.Module``
     - Model with configured defaults (reserved)
   * - ``**kwargs``
     -
     - Forward-compatibility; always accept it

The rule exposes three variants after registration:

- ``my_agg`` — checked in debug mode, unchecked in release
- ``my_agg.checked`` — always validates parameters
- ``my_agg.unchecked`` — skips validation (faster in production)

.. seealso::

   For the full contract and anti-aliasing guarantee, see
   :doc:`/explanation/key-concepts`.
   For existing rules, see :doc:`/reference/aggregators/index`.
