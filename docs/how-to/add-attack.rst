How to add a new attack
=======================

This guide walks you through creating and registering a new Byzantine attack
strategy.

Step 1 — Create the module
--------------------------

Create a new Python file in ``attacks/``. The file is auto-discovered at
import time — no ``__init__.py`` edit is needed.

.. code-block:: text

   attacks/
   ├── identical.py
   ├── nan.py
   └── my_attack.py      ← your new file

Step 2 — Implement the attack function
---------------------------------------

The attack function must return a list of exactly ``f_real`` Byzantine
gradient tensors. Use the honest gradients to craft adversarial directions.

.. code-block:: python

   # attacks/my_attack.py

   import torch
   from . import register


   def attack(
       grad_honests,
       f_real,
       f_decl,
       defense,
       model,
       scale=1.0,
       **kwargs,
   ):
       """Scale the mean gradient by a configurable factor."""
       if f_real == 0:
           return []
       grad_avg = torch.stack(grad_honests).mean(dim=0)
       byz_grad = grad_avg * scale
       return [byz_grad] * f_real

Reserved parameters
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``grad_honests``
     - ``list[Tensor]``
     - Non-empty list of honest gradients
   * - ``f_real``
     - ``int``
     - Number of Byzantine gradients to generate
   * - ``f_decl``
     - ``int``
     - Number of declared Byzantine workers
   * - ``defense``
     - ``Callable``
     - The aggregation rule to evaluate against
   * - ``model``
     - ``nn.Module``
     - Model with configured defaults
   * - ``**kwargs``
     -
     - Forward-compatibility; always accept it

Step 3 — Write the ``check`` function
--------------------------------------

Return ``None`` when valid, or an error string otherwise.

.. code-block:: python

   def check(grad_honests, f_real, **kwargs):
       if not isinstance(grad_honests, list) or len(grad_honests) == 0:
           return "Expected a non-empty list of honest gradients"
       if not isinstance(f_real, int) or f_real < 0:
           return "Expected a non-negative f_real, got %r" % f_real

Step 4 — Register the attack
----------------------------

Call :func:`attacks.register` at module level with ``(name, unchecked, check)``.

.. code-block:: python

   register("my_attack", attack, check)

Step 5 — Use it
---------------

.. code-block:: bash

   python -m experiments --attack my_attack --attack-args scale:2.0 ...

Contract summary
----------------

The attack function **must**:

1. Accept keyword-only arguments (reserved parameters above).
2. Return exactly ``f_real`` tensors — no more, no less.
3. **Never** return tensors that alias any honest input tensor.
4. May reuse the same tensor object when all generated gradients are identical
   (e.g. ``[byz_grad] * f_real``).

The attack exposes three variants after registration:

- ``my_attack`` — checked in debug mode, unchecked in release
- ``my_attack.checked`` — always validates parameters
- ``my_attack.unchecked`` — skips validation (faster in production)

.. seealso::

   For the full contract, see :doc:`/explanation/key-concepts`.
   For existing attacks, see :doc:`/reference/attacks/index`.
