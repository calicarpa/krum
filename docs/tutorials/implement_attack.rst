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

We'll implement **RepeatAttack**, the simplest possible Byzantine attack.
It takes the first honest gradient and repeats it ``f`` times. Every
Byzantine worker sends the exact same gradient. The intuition: if honest
workers are converging, sending a stale or misleading gradient repeated
many times can shift the aggregate away from the true direction.

Step 1: subclass Attack
-------------------------

Create a file ``repeat_attack.py``:

.. code-block:: python

   from collections.abc import Sequence
   from typing import Any

   from torch import Tensor

   from krum.primitives.attacks import Attack

   class RepeatAttack(Attack):
       ...

Step 2: implement generate (gradients only)
----------------------------------------------

Start without ``out`` or ``**specialized`` to see the core logic:

.. code-block:: python

       @classmethod
       def generate(
           cls,
           honest_gradients: Sequence[Tensor] | Tensor,
           *,
           f: int,
       ) -> Tensor:
           ...

Stack the honest gradients, take the first one, and repeat it:

.. code-block:: python

       @classmethod
       def generate(
           cls,
           honest_gradients: Sequence[Tensor] | Tensor,
           *,
           f: int,
       ) -> Tensor:
           if not isinstance(honest_gradients, Tensor):
               honest_gradients = stack(list(honest_gradients))

           _, d = honest_gradients.shape
           first = honest_gradients[0:1]  # (1, d)

           if f == 0:
               return honest_gradients.new_empty((0, d))

           return first.expand(f, d)  # broadcast to (f, d)

If ``f`` is zero the attack returns an empty tensor. The simulation handles
this correctly.

Step 3: add the out parameter
-------------------------------

Add support for the optional output buffer:

.. code-block:: python

       @classmethod
       def generate(
           cls,
           honest_gradients: Sequence[Tensor] | Tensor,
           /,
           out: Tensor | None = None,
           *,
           f: int,
       ) -> Tensor:
           if not isinstance(honest_gradients, Tensor):
               honest_gradients = stack(list(honest_gradients))

           _, d = honest_gradients.shape
           first = honest_gradients[0:1]

           if f == 0:
               empty = honest_gradients.new_empty((0, d))
               if out is not None:
                   return out.copy_(empty)
               return empty

            result = first.expand(f, d)
            if out is not None:
                return out.copy_(result)
            return result

.. note::

   The ``out`` parameter is part of every attack's signature. It
   enables the caller to control memory allocation. See the
   :doc:`/reference/primitives/attacks/index` for the full API
   reference.

Step 4: add specialized keyword arguments
------------------------------------------

Add ``**specialized`` to absorb whatever the simulation passes:

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

To pass your own custom keyword arguments, use ``attack_kwargs`` on the
simulation:

.. code-block:: python

   sim = KrumSimulation(
       ...,
       attack=RepeatAttack,
       attack_kwargs={"my_param": 42},
   )

Inside the attack, read them from ``specialized``:

.. code-block:: python

   class RepeatAttack(Attack):
       @classmethod
       def generate(cls, honest_gradients, /, out=None, *, f, **specialized):
           threshold = specialized.get("my_param", 0)
           ...

Full code
---------

.. code-block:: python

   from collections.abc import Sequence
   from typing import Any

   from torch import Tensor, stack

   from krum.primitives.attacks import Attack


   class RepeatAttack(Attack):
       """Byzantine attack that repeats the first honest gradient f times.

       Every Byzantine worker sends the same gradient. Simple to
       implement, useful as a baseline for testing aggregators.
       """

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
           if not isinstance(honest_gradients, Tensor):
               honest_gradients = stack(list(honest_gradients))

           _, d = honest_gradients.shape
           first = honest_gradients[0:1]

           if f == 0:
               empty = honest_gradients.new_empty((0, d))
               if out is not None:
                   return out.copy_(empty)
               return empty

           result = first.expand(f, d)
           if out is not None:
               return out.copy_(result)
           return result

Using your attack
-----------------

Import it and call it like any built-in attack. Attacks are **stateless**,
so you pass the class itself, never an instance:

.. code-block:: python

   import torch
   from repeat_attack import RepeatAttack

   honest = torch.randn(8, 100)
   byzantine = RepeatAttack.generate(honest, f=2)
   print(byzantine.shape)  # (2, 100)
   print(torch.allclose(byzantine[0], byzantine[1]))  # True, all identical

In a simulation
---------------

Pass the **class** (not an instance) to a simulation:

.. code-block:: python

   from krum.primitives.aggregators.multikrum import MultiKrum
   from krum.primitives.models.mlp import Krum2017MLPMnist
   from krum.simulations.centralised.krum_nips_2017 import KrumSimulation

   sim = KrumSimulation(
       model_cls=Krum2017MLPMnist,
       train_set=train_set,
       test_set=test_set,
       aggregator=MultiKrum,
       attack=RepeatAttack,
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

   from repeat_attack import RepeatAttack

   def test_generates_correct_shape():
       honest = torch.randn(8, 100)
       byzantine = RepeatAttack.generate(honest, f=2)
       assert byzantine.shape == (2, 100)

   def test_generates_no_byzantine_when_f_is_zero():
       honest = torch.randn(8, 100)
       byzantine = RepeatAttack.generate(honest, f=0)
       assert byzantine.shape == (0, 100)

   def test_all_byzantine_are_identical():
       honest = torch.randn(8, 100)
       byzantine = RepeatAttack.generate(honest, f=3)
       assert torch.allclose(byzantine[0], byzantine[1])
       assert torch.allclose(byzantine[0], byzantine[2])

   def test_byzantine_matches_first_honest():
       honest = torch.ones(4, 10)
       byzantine = RepeatAttack.generate(honest, f=1)
       assert torch.allclose(byzantine[0], honest[0])

Next steps
---------

* :doc:`implement_aggregator`: write an aggregation rule that defends
  against your attack.
* :doc:`centralised_simulation_walkthrough`: test your attack in a
  full training loop.
* :doc:`decentralised_simulation_walkthrough`: use the same attack in a
  peer-to-peer setting.
* :doc:`structured_experiments`: benchmark your attack across seeds and
  aggregators with ``Orchestrator``.
* Browse the :doc:`/reference/primitives/attacks/index` for all built-in
  attacks.
