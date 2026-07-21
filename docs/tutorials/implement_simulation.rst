Implement a custom simulation
=============================

The built-in simulations cover common protocols. When you need a different
evaluation metric, a custom learning-rate schedule, or a new peer-to-peer
communication topology, subclass one of the two base classes:

* :class:`~krum.simulations.centralised.CentralisedSimulation` — parameter-server
  setting (one shared model, synchronous SGD).
* :class:`~krum.simulations.decentralised.DecentralisedSimulation` — peer-to-peer
  setting (each worker holds its own model, local optimisation + model mixing).

Centralised simulation
----------------------

A centralised simulation subclasses
:class:`~krum.simulations.centralised.CentralisedSimulation` and at minimum
defines an ``evaluate`` method — the base class deliberately leaves evaluation
to subclasses, since each protocol reports its own metrics.

What you can customise
----------------------

* **Evaluation** — define ``evaluate`` to return any tuple of metrics.
* **Training loss** — define ``evaluate_train`` if your protocol reports it
  (both built-in simulations do).
* **Training step** — override ``step`` to inject logic before or after each
  round.
* **Initialisation** — override ``setup`` to add custom weight initialisation
  or data transformations.

Minimal example
^^^^^^^^^^^^^^^

Here is a simulation that reports top-5 accuracy alongside the standard loss:

.. code-block:: python

   import torch

   from krum.simulations.centralised import CentralisedSimulation


   class Top5Simulation(CentralisedSimulation):
       def evaluate(self) -> tuple[float, float, float]:
           assert self._model is not None and self._test_loader is not None
           self._model.module.eval()
           with torch.no_grad():
               x, y = next(iter(self._test_loader))
               x, y = x.to(self.device), y.to(self.device)
               logits = self._model.module(x)
               loss = self.loss_fn(logits, y).item()
               preds = logits.argsort(dim=1, descending=True)
               top1 = (preds[:, 0] == y).float().mean().item()
               top5 = (preds[:, :5] == y.unsqueeze(1)).any(dim=1).float().mean().item()
           return loss, top1, top5

Use it exactly like the built-in simulations:

.. code-block:: python

   from torchvision import datasets, transforms
   from krum.primitives.aggregators.multikrum import MultiKrum
   from krum.primitives.attacks.sign_flip import SignFlipAttack
   from krum.primitives.models.mlp import Krum2017MLPMnist

   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,)),
   ])
   train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
   test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

   sim = Top5Simulation(
       model_cls=Krum2017MLPMnist,
       train_set=train_set, test_set=test_set,
       aggregator=MultiKrum, attack=SignFlipAttack,
       attack_kwargs={"scale": 1.5},
       n=10, f=2, rounds=50, batch_size=64, lr=0.01, seed=42,
   )
   sim.setup()
   for round_idx in range(50):
       sim.step()
       if round_idx % 10 == 0:
           loss, top1, top5 = sim.evaluate()
           print(f"round {round_idx}: loss={loss:.4f}  top-1={top1:.2%}  top-5={top5:.2%}")

Decentralised simulation
------------------------

A decentralised simulation subclasses
:class:`~krum.simulations.decentralised.DecentralisedSimulation` and must
implement three abstract methods:

* :meth:`~krum.simulations.decentralised.DecentralisedSimulation.local_update`
  — how each worker turns its gradient into a post-local-update model.
* :meth:`~krum.simulations.decentralised.DecentralisedSimulation.gather_received_models`
  — which models each worker receives (the communication topology).
* :meth:`~krum.simulations.decentralised.DecentralisedSimulation.build_step_result`
  — the snapshot structure returned by each round.

The constructor takes a :class:`~krum.primitives.models.Model` **instance**
(not a class), one data stream per honest worker, and a loss function:

.. code-block:: python

   import torch
   import torch.nn as nn

   from krum.primitives.models import Model
   from krum.simulations.decentralised import DecentralisedSimulation, StepResult


   class MyStepResult(StepResult):
       """Extra fields my protocol produces each round."""
       my_state: torch.Tensor


   class MySimulation(DecentralisedSimulation[MyStepResult]):
       def local_update(self, gradients: torch.Tensor) -> torch.Tensor:
           # Simple SGD: theta_{t+1/2} = theta_t - lr * gradient
           return self.parameters - 0.01 * gradients

       def gather_received_models(
           self,
           honest_vectors: torch.Tensor,
           byzantine_parameters: torch.Tensor,
           *,
           worker_index: int,
       ) -> torch.Tensor:
           # Receive every model (worst-case topology)
           own = honest_vectors[worker_index].unsqueeze(0)
           others = torch.cat([
               honest_vectors[:worker_index],
               honest_vectors[worker_index + 1:],
               byzantine_parameters,
           ], dim=0)
           return torch.cat([own, others], dim=0)

       def build_step_result(
           self,
           *,
           honest_gradients: torch.Tensor,
           local_parameters: torch.Tensor,
           byzantine_parameters: torch.Tensor,
           mixed_parameters: torch.Tensor,
           losses: torch.Tensor,
       ) -> MyStepResult:
           return {
               "step": self.step_index,
               "parameters": self.parameters.detach().clone(),
               "my_state": torch.tensor(0.0),
               "honest_gradients": honest_gradients.detach().clone(),
               "local_parameters": local_parameters.detach().clone(),
               "byzantine_parameters": byzantine_parameters.detach().clone(),
               "mixed_parameters": mixed_parameters.detach().clone(),
               "losses": losses.detach().clone(),
           }

The data argument is one iterable of batches per honest worker. Wrap a
:class:`~torch.utils.data.DataLoader` or use a generator. Streams are consumed
one batch per round, so make them infinite (or long enough) for the number of
rounds you plan to run — here with a simple cycling generator:

.. code-block:: python

   from torch.utils.data import DataLoader, TensorDataset
   from krum.primitives.aggregators.krum import Krum
   from krum.primitives.attacks.gaussian import GaussianAttack
   from krum.primitives.models import Model

   def cycle(loader):
       while True:
           yield from loader

   dummy_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
   streams = [cycle(DataLoader(dummy_data, batch_size=4)) for _ in range(8)]

   model = Model(nn.Linear(10, 2))

   sim = MySimulation(
       model=model,
       data=streams,
       loss_fn=nn.functional.cross_entropy,
       n=10,
       f=2,
       attack=GaussianAttack,
       aggregator=Krum,
       aggregator_kwargs={"n": 10, "f": 2},
       seed=42,
   )

.. note::

   The decentralised base injects only the per-worker ``pivot`` into each
   ``aggregate`` call — unlike the centralised simulations, it does **not**
   inject ``n`` and ``f``. Aggregators like ``Krum`` that require them must
   receive them through ``aggregator_kwargs``, as above. Pivot-anchored
   rules like ``NearestNeighborAverage`` need no extra arguments.

Run it with :meth:`~krum.simulations.decentralised.DecentralisedSimulation.step`
for one round or :meth:`~krum.simulations.decentralised.DecentralisedSimulation.run`
for several:

.. code-block:: python

   snapshots = sim.run(50)
   print(f"final loss: {snapshots[-1]['losses'].mean():.4f}")

Choosing a base class
---------------------

.. list-table::
   :header-rows: 1

   * - Use
     - Base class
   * - Parameter-server SGD, shared model, dataset splits, built-in evaluation
     - :class:`~krum.simulations.centralised.CentralisedSimulation`
   * - Peer-to-peer, per-worker model, custom topology, snapshot-based results
     - :class:`~krum.simulations.decentralised.DecentralisedSimulation`

Next steps
----------

* :doc:`using_simulations` — using the built-in simulations.
* See :doc:`/reference/simulations/centralised/index` for the full
  :class:`~krum.simulations.centralised.CentralisedSimulation` API.
* See :doc:`/reference/simulations/decentralised/index` for the full
  :class:`~krum.simulations.decentralised.DecentralisedSimulation` API.
* :doc:`working_with_orchestrator` — collect structured results with
  ``Metric`` and ``Orchestrator``.
