Implement a custom aggregator
=============================

**Problem:** The built-in aggregation rules don't cover your use case.
How do you write your own aggregation rule?

All aggregators in Krum follow the same protocol:

* Subclass :class:`~krum.primitives.aggregators.Aggregator`
* Implement :meth:`~krum.primitives.aggregators.Aggregator.aggregate` as a ``@classmethod``
* Accept ``gradients`` (first positional arg), an optional ``out`` tensor, and
  rule-specific keyword arguments like ``f``, ``n``, ``m``
* Return a single tensor of shape ``(d,)``

What we'll build
----------------

We'll implement **FirstGrad**, the simplest possible aggregator. It
discards all gradients except the first one. This is not a Byzantine-resilient
rule, but it illustrates the protocol in its purest form.

We'll start with just ``gradients``, then add ``out`` for in-place output,
then ``**specialized`` for extra parameters.

Step 1: subclass Aggregator
----------------------------

Create a file ``first_grad.py``:

.. code-block:: python

   from collections.abc import Sequence
   from typing import Any

   from torch import Tensor

   from krum.primitives.aggregators import Aggregator

   class FirstGrad(Aggregator):
       ...

Step 2: implement aggregate (gradients only)
----------------------------------------------

Start with only the ``gradients`` argument, the minimal contract:

.. code-block:: python

       @classmethod
       def aggregate(
           cls,
           gradients: Sequence[Tensor] | Tensor,
       ) -> Tensor:
           ...

Normalise the input, then return the first gradient:

.. code-block:: python

       @classmethod
       def aggregate(
           cls,
           gradients: Sequence[Tensor] | Tensor,
       ) -> Tensor:
           if not isinstance(gradients, Tensor):
               gradients = stack(list(gradients))

           return gradients[0]

That is the full algorithm. Feed it ``n`` worker gradients and get back
the gradient of worker 0.

At this point the aggregator works for direct calls and the simulation
will accept it, but it ignores the ``out`` and ``**specialized``
parameters that more advanced callers may pass.

Step 3: add the out parameter
-------------------------------

The optional ``out`` argument lets the caller pass a pre-allocated tensor.
When provided, write into it instead of returning a new tensor:

.. code-block:: python

       @classmethod
       def aggregate(
           cls,
           gradients: Sequence[Tensor] | Tensor,
           out: Tensor | None = None,
       ) -> Tensor:
           if not isinstance(gradients, Tensor):
               gradients = stack(list(gradients))

           if out is not None:
               out.copy_(gradients[0])
               return out
           return gradients[0]

With ``out``, the simulation can reuse the same output buffer every round
and avoid an allocation:

.. code-block:: python

   buffer = torch.empty(100)
   result = FirstGrad.aggregate(grads, out=buffer)
   assert result is buffer  # same tensor, no allocation

.. note::

   The ``out`` parameter is part of every aggregator's signature. It
   enables the caller to control memory allocation. See the
   :doc:`/reference/primitives/aggregators/index` for the full API
   reference.

Step 4: add specialized keyword arguments
------------------------------------------

Aggregators receive extra parameters like ``n`` and ``f`` from the
simulation. Absorb them with ``**specialized`` so the simulation
interface stays uniform:

.. code-block:: python

       @classmethod
       def aggregate(
           cls,
           gradients: Sequence[Tensor] | Tensor,
           /,
           out: Tensor | None = None,
           **specialized: Any,
       ) -> Tensor:
           ...

The ``/`` marks ``gradients`` as positional-only (prevents accidental
keyword usage). ``**specialized`` collects everything else (``n``,
``f``, ``m``, etc.) that the simulation passes automatically. You can
inspect them inside your algorithm:

.. code-block:: python

   class FirstGrad(Aggregator):
       @classmethod
       def aggregate(cls, gradients, /, out=None, **specialized):
           f = specialized.get("f", 0)
           print(f"Running with f={f} Byzantine workers")
           ...

To pass your own custom keyword arguments, use ``aggregator_kwargs`` on
the simulation:

.. code-block:: python

   sim = KrumSimulation(
       ...,
       aggregator=FirstGrad,
       aggregator_kwargs={"my_param": 42},
   )

Full code
---------

.. code-block:: python

   from collections.abc import Sequence
   from typing import Any

   import torch
   from torch import Tensor, stack

   from krum.primitives.aggregators import Aggregator


   class FirstGrad(Aggregator):
       """Aggregation rule that keeps only the first gradient.

       This is a pedagogical rule, not a robust one. Every worker
       after the first is ignored entirely.
       """

       @classmethod
       def aggregate(
           cls,
           gradients: Sequence[Tensor] | Tensor,
           /,
           out: Tensor | None = None,
           **specialized: Any,
       ) -> Tensor:
           if not isinstance(gradients, Tensor):
               gradients = stack(list(gradients))

           if out is not None:
               return out.copy_(gradients[0])
           return gradients[0]

Using your aggregator
---------------------

Import it and call it like any built-in aggregator. Aggregators are
**stateless**, so you pass the class itself, never an instance:

.. code-block:: python

   import torch
   from first_grad import FirstGrad

   grads = torch.randn(10, 100)
   result = FirstGrad.aggregate(grads)
   print(result.shape)  # (100,)
   print(result is grads[0])  # True, same tensor

In a simulation
---------------

Pass the **class** (not an instance) to a simulation:

.. code-block:: python

   from krum.primitives.attacks.sign_flip import SignFlipAttack
   from krum.primitives.models.mlp import Krum2017MLPMnist
   from krum.simulations.centralised.krum_nips_2017 import KrumSimulation

   sim = KrumSimulation(
       model_cls=Krum2017MLPMnist,
       train_set=train_set,
       test_set=test_set,
       aggregator=FirstGrad,
       attack=SignFlipAttack,
       attack_kwargs={"scale": 1.5},
       n=10, f=2, rounds=50, batch_size=64, lr=0.01, seed=42,
   )
   sim.setup()
   for _ in range(50):
       sim.step()
   loss, accuracy = sim.evaluate()

Testing
-------

A minimal test suite, run with pytest:

.. code-block:: python

   import pytest
   import torch

   from first_grad import FirstGrad

   def test_first_grad_shape():
       grads = torch.randn(10, 100)
       result = FirstGrad.aggregate(grads)
       assert result.shape == (100,)

   def test_first_grad_returns_first():
       grads = torch.randn(10, 100)
       result = FirstGrad.aggregate(grads)
       assert torch.allclose(result, grads[0])

   def test_first_grad_uses_out():
       grads = torch.randn(10, 100)
       buffer = torch.empty(100)
       result = FirstGrad.aggregate(grads, out=buffer)
       assert result is buffer
       assert torch.allclose(result, grads[0])

Next steps
----------

* Browse the :doc:`/reference/primitives/aggregators/index` for all built-in rules
* See :doc:`implement_attack` for creating custom Byzantine attacks
* :doc:`centralised_simulation_walkthrough` to test your aggregator in a full training loop
