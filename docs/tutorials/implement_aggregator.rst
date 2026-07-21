Implement a custom aggregator
=============================

This tutorial walks you through implementing a new gradient aggregation rule
from scratch.

All aggregators in Krum follow the same protocol:

* Subclass :class:`~krum.primitives.aggregators.Aggregator`
* Implement :meth:`~krum.primitives.aggregators.Aggregator.aggregate` as a ``@classmethod``
* Accept ``gradients`` (first positional arg), an optional ``out`` tensor, and
  rule-specific keyword arguments like ``f``, ``n``, ``m``
* Return a single tensor of shape ``(d,)``

What we'll build
----------------

We'll implement **CenteredMean**, a simple Byzantine-resilient rule that
discards the single gradient farthest from the mean and averages the rest.
It's not a published algorithm, but it's simple enough to illustrate the
pattern clearly.

Step 1: subclass Aggregator
-----------------------------

Create a file ``centered_mean.py``:

.. code-block:: python

   from collections.abc import Sequence
   from typing import Any

   from torch import Tensor

   from krum.primitives.aggregators import Aggregator

   class CenteredMean(Aggregator):
       ...

Step 2: implement aggregate
-----------------------------

Add the ``@classmethod``:

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

Step 3: unpack and validate
-----------------------------

Normalise the input to a stacked 2-D tensor and add guardrails:

.. code-block:: python

           if not isinstance(gradients, Tensor):
               gradients = stack(list(gradients))

           n = gradients.shape[0]
           if n < 3:
               msg = f"CenteredMean requires n >= 3, got n={n}"
               raise ValueError(msg)

Step 4: compute the result
-----------------------------

Find the gradient farthest from the mean, remove it, and average the rest:

.. code-block:: python

           from torch import cdist, mean, stack, topk

           # Mean across all workers
           mu = mean(gradients, dim=0, keepdim=True)          # (1, d)

           # Distances from the mean
           dists = cdist(gradients, mu).squeeze(-1)            # (n,)

           # Index of the farthest gradient
           _, worst = topk(dists, k=1, largest=True)           # (1,)

           # Mask it out and average the rest
           mask = torch.ones(n, dtype=torch.bool)
           mask[worst] = False
           clean = gradients[mask]                              # (n-1, d)

           return mean(clean, dim=0, out=out)

Full code
---------

.. code-block:: python

   from collections.abc import Sequence
   from typing import Any

   import torch
   from torch import Tensor, cdist, mean, stack, topk

   from krum.primitives.aggregators import Aggregator


   class CenteredMean(Aggregator):
       """Byzantine-resilient aggregation that discards the gradient
       farthest from the mean and averages the rest.

       This is a pedagogical rule, not a published algorithm,
       that demonstrates the Aggregator protocol.
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

           n = gradients.shape[0]
           if n < 3:
               msg = f"CenteredMean requires n >= 3, got n={n}"
               raise ValueError(msg)

           mu = mean(gradients, dim=0, keepdim=True)
           dists = cdist(gradients, mu).squeeze(-1)
           _, worst = topk(dists, k=1, largest=True)

           mask = torch.ones(n, dtype=torch.bool)
           mask[worst] = False
           clean = gradients[mask]

           return mean(clean, dim=0, out=out)

Using your aggregator
---------------------

Import it and call it like any built-in aggregator — aggregators are
**stateless**, so you pass the class itself, never an instance:

.. code-block:: python

   import torch
   from centered_mean import CenteredMean

   grads = torch.randn(10, 100)
   result = CenteredMean.aggregate(grads)
   print(result.shape)  # (100,)

In a simulation
---------------

Pass the **class** (not an instance) to a simulation. The ``n`` and ``f``
injected by the simulation are absorbed by ``**specialized``:

.. code-block:: python

   from krum.primitives.attacks.sign_flip import SignFlipAttack
   from krum.primitives.models.mlp import Krum2017MLPMnist
   from krum.simulations.centralised.krum_nips_2017 import KrumSimulation

   sim = KrumSimulation(
       model_cls=Krum2017MLPMnist,
       train_set=train_set,   # see :doc:`centralised_simulation_walkthrough` for dataset setup
       test_set=test_set,
       aggregator=CenteredMean,
       attack=SignFlipAttack,
       attack_kwargs={"scale": 1.5},
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

   from centered_mean import CenteredMean

   def test_centered_mean_shape():
       grads = torch.randn(10, 100)
       result = CenteredMean.aggregate(grads)
       assert result.shape == (100,)

   def test_centered_mean_removes_outlier():
       # 9 identical gradients + 1 far away
       honest = torch.ones(9, 10)
       outlier = -100 * torch.ones(1, 10)
       grads = torch.cat([honest, outlier], dim=0)

       result = CenteredMean.aggregate(grads)
       assert torch.allclose(result, torch.ones(10))

   def test_centered_mean_too_few_workers():
       grads = torch.randn(2, 100)
       with pytest.raises(ValueError, match="n >= 3"):
           CenteredMean.aggregate(grads)

Next steps
----------

* Browse the :doc:`/reference/primitives/aggregators/index` for all built-in rules
* See :doc:`implement_attack` for creating custom Byzantine attacks
* :doc:`centralised_simulation_walkthrough` to test your aggregator in a full training loop
