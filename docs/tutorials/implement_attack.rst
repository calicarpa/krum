Implement a custom attack
=========================

**Problem:** You want to stress-test an aggregator against a novel attack
not yet in the library. How do you implement a custom Byzantine attack
strategy from scratch?

All attacks in Krum follow the same protocol:

* Subclass :class:`~krum.primitives.attacks.Attack`
* Implement :meth:`~krum.primitives.attacks.Attack.generate` as a ``@classmethod``
* Accept ``honest_gradients`` (first positional arg), an optional ``out`` tensor,
  a keyword-only ``f`` (number of Byzantine gradients to produce), and any
  attack-specific keyword arguments
* Return a tensor of shape ``(f, d)``

What we'll build
----------------

We'll implement **MomentumMismatch**, an attack that estimates the honest
update direction as the mean of the honest gradients and sends Byzantine
gradients pointing in the *opposite* direction, scaled by a configurable
factor. A small random perturbation is added per worker so the Byzantine
gradients aren't identical — a crowd of identical outliers is easier for
robust aggregators to isolate. The intuition: if the honest workers are
converging toward a good minimum, the Byzantine workers try to pull the
model elsewhere.

Step 1: subclass Attack
-------------------------

Create a file ``momentum_mismatch.py``:

.. code-block:: python

   from collections.abc import Sequence
   from typing import Any

   from torch import Tensor

   from krum.primitives.attacks import Attack

   class MomentumMismatch(Attack):
       ...

Step 2: implement generate
----------------------------

Add the ``@classmethod``:

.. code-block:: python

       @classmethod
       def generate(
           cls,
           honest_gradients: Sequence[Tensor] | Tensor,
           /,
           out: Tensor | None = None,
           *,
           f: int,
           **specialized: Any,
       ) -> Tensor:
           ...

Step 3: validate inputs
-------------------------

.. code-block:: python

           if f < 0:
               msg = f"Invalid f, got {f!r}, expected 0 <= f"
               raise ValueError(msg)
           if len(honest_gradients) == 0:
               msg = "Expected at least one honest gradient"
               raise ValueError(msg)

           stacked = stack(list(honest_gradients))
           if not is_floating_point(stacked):
               raise TypeError("Expected honest gradients to use a floating-point dtype")

Step 4: compute the Byzantine gradients
-----------------------------------------

The attack estimates the honest direction as the mean of the honest
gradients, then sends the opposite vector with ``scale`` times the
magnitude. When ``f == 0``, return an empty ``(0, d)`` tensor like the
built-in attacks do:

.. code-block:: python

           from torch import mean, randn

           _, d = stacked.shape
           honest_mean = mean(stacked, dim=0)             # (d,)
           magnitude = honest_mean.norm() * scale

           if f == 0:
               empty = stacked.new_empty((0, d))
               if out is not None:
                   return out.copy_(empty)
               return empty

           # Each Byzantine worker sends roughly the same wrong direction
           # plus a small random perturbation so they aren't identical
           noise = randn(f, d, device=stacked.device, dtype=stacked.dtype)
           byzantine = (
               -honest_mean.unsqueeze(0).expand(f, -1) * scale
               + noise * magnitude * 0.1
           )

           if out is not None:
               return out.copy_(byzantine)
           return byzantine

Full code
---------

.. code-block:: python

   from collections.abc import Sequence
   from typing import Any

   from torch import Tensor, is_floating_point, mean, randn, stack

   from krum.primitives.attacks import Attack


   class MomentumMismatch(Attack):
       """Byzantine attack that pushes against the estimated honest direction.

       Computes the mean of honest gradients, then sends Byzantine
       gradients that point in the opposite direction with a configurable
       scale factor. Small random noise is added to each Byzantine worker
       so the gradients aren't identical.
       """

       @classmethod
       def generate(
           cls,
           honest_gradients: Sequence[Tensor] | Tensor,
           /,
           out: Tensor | None = None,
           *,
           f: int,
           scale: float = 2.0,
           **specialized: Any,
       ) -> Tensor:
           if f < 0:
               msg = f"Invalid f, got {f!r}, expected 0 <= f"
               raise ValueError(msg)
           if scale < 0:
               msg = f"Invalid scale, got {scale!r}, expected scale >= 0"
               raise ValueError(msg)
           if len(honest_gradients) == 0:
               msg = "Expected at least one honest gradient"
               raise ValueError(msg)

           stacked = stack(list(honest_gradients))
           if not is_floating_point(stacked):
               raise TypeError("Expected honest gradients to use a floating-point dtype")

           _, d = stacked.shape
           honest_mean = mean(stacked, dim=0)
           magnitude = honest_mean.norm() * scale

           if f == 0:
               empty = stacked.new_empty((0, d))
               if out is not None:
                   return out.copy_(empty)
               return empty

           noise = randn(f, d, device=stacked.device, dtype=stacked.dtype)
           byzantine = (
               -honest_mean.unsqueeze(0).expand(f, -1) * scale + noise * magnitude * 0.1
           )

           if out is not None:
               return out.copy_(byzantine)
           return byzantine

Using your attack
-----------------

Import it and call it like any built-in attack — attacks are **stateless**,
so you pass the class itself, never an instance:

.. code-block:: python

   import torch
   from momentum_mismatch import MomentumMismatch

   honest = torch.randn(8, 100)
   byzantine = MomentumMismatch.generate(honest, f=2, scale=2.0)
   print(byzantine.shape)  # (2, 100)

In a simulation
---------------

Pass the **class** (not an instance) to a simulation, with any extra
hyperparameters forwarded via ``attack_kwargs``:

.. code-block:: python

   from krum.primitives.aggregators.multikrum import MultiKrum
   from krum.primitives.models.mlp import Krum2017MLPMnist
   from krum.simulations.centralised.krum_nips_2017 import KrumSimulation

   sim = KrumSimulation(
       model_cls=Krum2017MLPMnist,
       train_set=train_set,   # see :doc:`centralised_simulation_walkthrough` for dataset setup
       test_set=test_set,
       aggregator=MultiKrum,
       attack=MomentumMismatch,
       attack_kwargs={"scale": 2.0},
       n=10, f=2, rounds=50, batch_size=64, lr=0.01, seed=42,
   )
   sim.setup()
   for _ in range(50):
       sim.step()
   loss, accuracy = sim.evaluate()

To sweep configurations and collect structured results, wrap the run in an
experiment function driven by the :class:`~krum.orchestration.orchestrator.Orchestrator` —
see :doc:`working_with_orchestrator`.

Testing
-------

A minimal test suite, run with pytest:

.. code-block:: python

   import pytest
   import torch

   from momentum_mismatch import MomentumMismatch

   def test_generates_correct_shape():
       honest = torch.randn(8, 100)
       byzantine = MomentumMismatch.generate(honest, f=2, scale=2.0)
       assert byzantine.shape == (2, 100)

   def test_generates_no_byzantine_when_f_is_zero():
       honest = torch.randn(8, 100)
       byzantine = MomentumMismatch.generate(honest, f=0)
       assert byzantine.shape == (0, 100)

   def test_rejects_negative_f():
       honest = torch.randn(8, 100)
       with pytest.raises(ValueError, match="Invalid f"):
           MomentumMismatch.generate(honest, f=-1)

   def test_direction_is_opposite():
       honest = torch.ones(4, 10)
       byzantine = MomentumMismatch.generate(honest, f=1, scale=1.0)
       # Byzantine gradient should have negative correlation with honest mean
       assert torch.dot(byzantine[0], honest.mean(dim=0)) < 0

Next steps
----------

* Browse the :doc:`/reference/primitives/attacks/index` for all built-in attacks
* See :doc:`implement_aggregator` for creating custom aggregation rules
* :doc:`centralised_simulation_walkthrough` to test your attack in a full training loop
