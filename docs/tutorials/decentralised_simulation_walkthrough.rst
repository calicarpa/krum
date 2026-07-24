Decentralised simulation walkthrough
====================================

**Problem:** Your scenario needs peer-to-peer communication with
per-worker models instead of a central parameter server. How do you
set up a simulation where each worker trains its own model and
exchanges parameters with neighbours?

Krum ships with one built-in decentralised (peer-to-peer) simulation.
This tutorial covers the peer-to-peer framework where each worker holds
its **own** model and workers exchange models through a communication
topology.

If you have not yet used the centralised simulations, start with
:doc:`centralised_simulation_walkthrough` first. This tutorial builds on
that foundation.

Centralised vs decentralised
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Centralised
     - Decentralised
   * - One shared model, broadcast to all workers each round.
     - Each worker owns its **own** model. No broadcast.
   * - Parameter-server topology: one aggregator combines all ``n``
       gradients.
     - Peer-to-peer topology: workers mix their model with models
       *received* from neighbours.
   * - Lifecycle: ``setup() -> step() -> evaluate()``.
      - Lifecycle: **instantiate → step** (or ``run(rounds)``); no setup,
        no evaluate.
   * - One loss / accuracy for the shared model.
     - Per-worker losses, per-worker parameter vectors.

Available simulations
---------------------

**Decentralised** (peer-to-peer):

* :class:`~krum.simulations.decentralised.monna_icml_2023.MonnaSimulation` from Farhadkhani et al. (ICML 2023): momentum-SGD, nearest-neighbour averaging, per-worker losses and model snapshots.

All decentralised simulations share the lifecycle:
**instantiate → step** (or ``run(rounds)``), with a per-round snapshot.

Each round runs two phases:

1. **Local optimisation**: each honest worker computes a gradient on its own
   batch and updates its own model.
2. **Model mixing**: each worker gathers ``n - f`` models from other nodes
   (*received set*) and replaces its model with an aggregate of that set.

Minimal example
---------------

.. code-block:: python

   import random
   import torch
   import torch.nn as nn
   from itertools import cycle
   from torch.utils.data import DataLoader, Subset
   from torchvision import datasets, transforms

   from krum.primitives.models.mlp import Monna2023SmallMnist
   from krum.primitives.models import Model
   from krum.simulations.decentralised.monna_icml_2023 import MonnaSimulation

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   seed = 42
   torch.manual_seed(seed)
   random.seed(seed)
   n, f = 6, 0

   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,)),
   ])
   train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

   # One infinite data stream per honest worker
   def cycle_loader(loader):
       while True:
           yield from loader

   workers_data = [
       cycle_loader(DataLoader(
           Subset(train_set, range(i * 5000, (i + 1) * 5000)),
           batch_size=64, shuffle=True,
       ))
       for i in range(n - f)
   ]

   model = Model(Monna2023SmallMnist().to(device))

   sim = MonnaSimulation(
       model=model,
       data=workers_data,
       loss_fn=nn.CrossEntropyLoss(),
       n=n,
       f=f,
       learning_rate=0.1,
       beta=0.99,
       seed=seed,
   )

   results = sim.run(50)

   # results is a list of MonnaStepResult dicts, one per round
   print(f"Ran {len(results)} rounds")
   print(f"Final per-worker losses: {results[-1]['losses']}")

The ``data`` argument is a **sequence of iterables**, one per honest worker.
Each iterable yields ``(inputs, targets)`` batches. Using :class:`~torch.utils.data.DataLoader`
wrapped in an infinite iterator (``cycle_loader``) is the standard pattern.
Without the cycle, a stream that runs out raises ``StopIteration``.

Inspecting the round snapshot
-----------------------------

:meth:`~krum.simulations.decentralised.DecentralisedSimulation.step` returns a
:class:`~krum.simulations.decentralised.StepResult` dict (or a subclass like
:class:`~krum.simulations.decentralised.monna_icml_2023.MonnaStepResult`):

.. code-block:: python

   result = sim.step()  # a single round

   print(f"Step:         {result['step']}")
   print(f"Parameters:   {result['parameters'].shape}")      # (n-f, d)
   print(f"Momentum:     {result['momentum'].shape}")         # (n-f, d)
   print(f"Losses:       {result['losses']}")                 # (n-f,)
   print(f"Local params: {result['local_parameters'].shape}")  # (n-f, d)
   print(f"Mixed params: {result['mixed_parameters'].shape}")  # (n-f, d)

   # access specific workers
   worker_0_loss = result["losses"][0]
   worker_2_params = result["parameters"][2]

The keys are:

* ``step``: round counter (1-indexed)
* ``parameters``: the committed parameters after mixing, shape ``(n - f, d)``
* ``momentum``: the momentum buffer (``MonnaStepResult`` only)
* ``honest_gradients``: computed gradients before local update
* ``local_parameters``: parameters after local update, before mixing
* ``byzantine_parameters``: the Byzantine models injected this round
* ``mixed_parameters``: same as ``parameters`` (already committed)
* ``losses``: per-worker scalar losses

Byzantine workers
-----------------

Add Byzantine workers by setting ``f > 0`` and providing an attack. Each round,
the attack generates ``f`` Byzantine parameter vectors from the honest ones.
The :attr:`~krum.simulations.decentralised.monna_icml_2023.MonnaSimulation.byzantine_reach`
mode controls which workers receive them:

* ``"all"``: every Byzantine model reaches every worker; only the honest
  responders are sampled. Worst-case adversary.
* ``"sampled"``: responders drawn uniformly from all other nodes; a worker
  receives ``0`` to ``f`` Byzantine models. Models gossip where Byzantine
  reach is random.

.. code-block:: python

   from krum.primitives.attacks.sign_flip import SignFlipAttack

   sim_all = MonnaSimulation(
       model=model, data=workers_data, loss_fn=nn.CrossEntropyLoss(),
       n=8, f=2, learning_rate=0.1,
       attack=SignFlipAttack, attack_kwargs={"scale": 1.5},
       byzantine_reach="all", seed=42,
   )

   sim_sampled = MonnaSimulation(
       model=model, data=workers_data, loss_fn=nn.CrossEntropyLoss(),
       n=8, f=2, learning_rate=0.1,
       attack=SignFlipAttack, attack_kwargs={"scale": 1.5},
       byzantine_reach="sampled", seed=42,
   )

   result_all = sim_all.run(50)
   result_sampled = sim_sampled.run(50)
   print(f"'all'    mean final loss: {result_all[-1]['losses'].mean():.4f}")
   print(f"'sampled' mean final loss: {result_sampled[-1]['losses'].mean():.4f}")

Switching the mixing aggregator
-------------------------------

By default ``MonnaSimulation`` uses
:class:`~krum.primitives.aggregators.nearest_neighbor_average.NearestNeighborAverage`
with ``num_closest = n - 2f``.  You can override this with any
:class:`~krum.primitives.aggregators.Aggregator` subclass:

.. code-block:: python

   from krum.primitives.aggregators.median import Median

   sim = MonnaSimulation(
       model=model,
       data=workers_data,
       loss_fn=nn.CrossEntropyLoss(),
       n=8,
       f=2,
       learning_rate=0.1,
       attack=SignFlipAttack,
       aggregator=Median,
       seed=42,
   )

   results = sim.run(50)

When specifying a custom aggregator, you must also pass
``aggregator_kwargs`` if the aggregator requires extra parameters.

Custom data streams
-------------------

Each honest worker gets its own data iterator. The standard pattern wraps a
:class:`~torch.utils.data.DataLoader` in an infinite cycle:

.. code-block:: python

   from itertools import cycle
   from torch.utils.data import DataLoader

   def cycle_loader(loader):
       while True:
           yield from loader

   workers_data = [
       cycle_loader(DataLoader(shard, batch_size=64, shuffle=True))
       for shard in worker_shards
   ]

You are not limited to :class:`~torch.utils.data.DataLoader`; any iterable of
``(inputs, targets)`` tuples works:

.. code-block:: python

   workers_data = [
       [(torch.randn(4, 784), torch.randint(0, 10, (4,))) for _ in range(100)]
       for _ in range(4)
   ]

Without a cycle, a stream that runs out raises ``StopIteration``.

Data partitioning
-----------------

The minimal example splits the dataset into equal IID shards with
``Subset``. The experiment scripts also support a **Dirichlet non-IID**
split, where each worker receives a different class distribution:

.. code-block:: python

   from experiments.decentralised.datasets import split_dataset

   worker_shards = split_dataset(
       dataset=train_set,
       partition="dirichlet",
       num_parts=n - f,
       dirichlet_alpha=1.0,
       seed=seed,
   )

A lower ``alpha`` produces more skewed distributions (less IID). This is
useful for testing robustness under data heterogeneity.

Evaluating worker models
-------------------------

In the centralised simulation, ``evaluate()`` reports loss and accuracy
for the single shared model. In the decentralised setting, each worker
has its own parameters. The standard approach evaluates every honest
worker on the same test set and averages the results:

.. code-block:: python

   @torch.no_grad()
   def evaluate_workers(model, parameters, test_loader, loss_fn):
       losses, accuracies = [], []
       for worker_params in parameters:
           model.parameters.copy_(worker_params)
           model.module.eval()
           total_loss = total_correct = total = 0
           for inputs, targets in test_loader:
               inputs, targets = inputs.to(device), targets.to(device)
               logits = model.module(inputs)
               loss = loss_fn(logits, targets)
               total_loss += loss.item() * targets.numel()
               total_correct += (logits.argmax(1) == targets).sum().item()
               total += targets.numel()
           losses.append(total_loss / total)
           accuracies.append(total_correct / total)
       return sum(losses) / len(losses), sum(accuracies) / len(accuracies)

   test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
   avg_loss, avg_acc = evaluate_workers(
       model, sim.parameters, test_loader, nn.CrossEntropyLoss()
   )
   print(f"Average test loss: {avg_loss:.4f}, accuracy: {avg_acc:.2%}")

The function iterates over all honest worker parameter vectors
(``sim.parameters``, shape ``(n-f, d)``), copies each one into the
shared ``model`` via ``copy_parameters_to_model``, and evaluates on
the test set.

Continuing training
-------------------

State persists on the simulation instance, so you can call ``run()``
multiple times:

.. code-block:: python

   sim = MonnaSimulation(
       model=model, data=workers_data, loss_fn=nn.CrossEntropyLoss(),
       n=4, f=0, learning_rate=0.1, seed=42,
   )

   first_50 = sim.run(50)
   next_50  = sim.run(50)   # continues from round 51
   total_100 = first_50 + next_50

   print(f"Step index after 100 rounds: {sim.step_index}")

Accessing per-worker state
--------------------------

The simulation's ``parameters`` attribute is a ``(n - f, d)`` tensor, one row
per honest worker:

.. code-block:: python

   # All worker parameters after the last round
   all_params = sim.parameters  # shape: (n-f, d)

   # First worker's parameter vector
   worker_0 = sim.parameters[0]

   # Momentum buffer (MonnaSimulation only)
   worker_0_momentum = sim.momentum[0]

   # Copy a specific worker's parameters back into the Model to inspect
   sim.copy_parameters_to_model(sim.parameters[2])
   # Now use sim.model.module for evaluation, logging, etc.

Next steps
----------

* :doc:`centralised_simulation_walkthrough`: if you have not yet tried
  the parameter-server simulations.
* :doc:`structured_experiments`: collect structured results across
  multiple configurations with ``Metric`` and ``Orchestrator``.
* :doc:`implement_simulation`: create your own decentralised simulation
  by subclassing
  :class:`~krum.simulations.decentralised.DecentralisedSimulation`.
* :doc:`/reference/simulations/decentralised/index`: the full
  decentralised simulation reference.
