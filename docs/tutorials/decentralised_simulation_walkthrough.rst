Decentralised simulation walkthrough
====================================

**Problem:** Your scenario needs peer-to-peer communication with
per-worker models instead of a central parameter server. Each worker
trains its own model and the simulation handles model mixing between
neighbours each round.

Krum ships with one built-in decentralised (peer-to-peer) simulation.
This tutorial covers the peer-to-peer framework where each worker holds
its **own** model and workers exchange models through a communication
topology.

All decentralised simulations share the lifecycle:
**instantiate → step** (or ``run(rounds)``), with a per-round snapshot.

.. seealso::

   :doc:`/reference/simulations/decentralised/index`
      Full reference for
      :class:`~krum.simulations.decentralised.monna_icml_2023.MonnaSimulation`.

Each round runs two phases:

1. **Local optimisation**: each honest worker computes a gradient on its own
   batch and updates its own model.
2. **Model mixing**: each worker gathers ``n - f`` models from other nodes
   (*received set*) and replaces its model with an aggregate of that set.

Minimal example
---------------

Data preparation
^^^^^^^^^^^^^^^^

The ``data`` argument of a decentralised simulation is a **sequence of
iterables**, one per honest worker. The standard pattern wraps a
:class:`~torch.utils.data.DataLoader` in an infinite cycle. Without the
cycle a stream that runs out raises ``StopIteration``:

.. code-block:: python

   import random
   import torch
   import torch.nn as nn
   from itertools import cycle
   from torch.utils.data import DataLoader, Subset
   from torchvision import datasets, transforms

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
   test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

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

Any iterable of ``(inputs, targets)`` tuples also works:

.. code-block:: python

   workers_data = [
       [(torch.randn(4, 784), torch.randint(0, 10, (4,))) for _ in range(100)]
       for _ in range(4)
   ]

The IID ``Subset`` splitting above gives each worker an equal, shuffled
portion of the dataset. The experiment scripts in
``experiments/decentralised/`` add a ``split_dirichlet`` variant for
non-IID data (class-skew controlled by an ``alpha`` parameter).

Instantiation
^^^^^^^^^^^^^

The model is wrapped in a :class:`~krum.primitives.models.Model` container
that exposes a ``.parameters`` tensor and a ``.module`` (the underlying
``nn.Module``). Pass the model, data streams, and hyperparameters to
the simulation constructor:

.. code-block:: python

   from krum.primitives.models.mlp import Monna2023SmallMnist
   from krum.primitives.models import Model
   from krum.simulations.decentralised.monna_icml_2023 import MonnaSimulation

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

Training
^^^^^^^^

Call ``run(rounds)`` to train. The result is a list of result dicts, one
per round:

.. code-block:: python

   results = sim.run(50)

   # results is a list of MonnaStepResult dicts, one per round
   print(f"Ran {len(results)} rounds")
   print(f"Final per-worker losses: {results[-1]['losses']}")

Inspecting the round snapshot
-----------------------------

:meth:`~krum.simulations.decentralised.DecentralisedSimulation.step` returns a
:class:`~krum.simulations.decentralised.StepResult` dict (or a subclass like
:class:`~krum.simulations.decentralised.monna_icml_2023.MonnaStepResult`):

.. code-block:: python

   result = sim.step()  # a single round

The returned dict contains:

.. list-table::
   :header-rows: 1
   :widths: 20 40 15

   * - Key
     - Description
     - Shape
   * - ``step``
     - Round counter (1-indexed)
     - scalar
   * - ``parameters``
     - Committed parameters after mixing
     - ``(n - f, d)``
   * - ``momentum``
     - Momentum buffer (``MonnaStepResult`` only)
     - ``(n - f, d)``
   * - ``honest_gradients``
     - Computed gradients before local update
     - ``(n - f, d)``
   * - ``local_parameters``
     - Parameters after local update, before mixing
     - ``(n - f, d)``
   * - ``byzantine_parameters``
     - Byzantine models injected this round
     - ``(f, d)``
   * - ``mixed_parameters``
     - Same as ``parameters`` (already committed)
     - ``(n - f, d)``
   * - ``losses``
     - Per-worker scalar losses
     - ``(n - f,)``

Byzantine workers
-----------------

Add Byzantine workers by setting ``f > 0`` and providing an attack. Each
round the attack generates ``f`` Byzantine parameter vectors from the
honest ones. The
:attr:`~krum.simulations.decentralised.monna_icml_2023.MonnaSimulation.byzantine_reach`
mode controls which workers receive them:

* ``"all"``: every Byzantine model reaches every worker (worst-case
  adversary).
* ``"sampled"``: responders are drawn uniformly from all other nodes,
  so a worker receives ``0`` to ``f`` Byzantine models.

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
with ``num_closest = n - 2f``. Override it with any
:class:`~krum.primitives.aggregators.Aggregator` subclass. Pass extra
aggregator parameters through ``aggregator_kwargs``:

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

Evaluating worker models
------------------------

In the decentralised setting each worker has its own parameters.
Evaluate every honest worker on the same test set and average the
results. The function below loads each worker's parameter vector into
the shared model, runs the full test set, and averages across workers:

.. code-block:: python

   @torch.no_grad()
   def evaluate_workers(model, parameters, test_loader, loss_fn):
       losses, accuracies = [], []
       for worker_params in parameters:
           # copy worker params into the model
           model.parameters.copy_(worker_params)
           model.module.eval()

           # run the full test set for this worker
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

       # average across all honest workers
       return sum(losses) / len(losses), sum(accuracies) / len(accuracies)

   test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
   avg_loss, avg_acc = evaluate_workers(
       model, sim.parameters, test_loader, nn.CrossEntropyLoss()
   )
   print(f"Average test loss: {avg_loss:.4f}, accuracy: {avg_acc:.2%}")

Next steps
----------

* :doc:`centralised_simulation_walkthrough`: try the simpler
  parameter-server simulation first if you haven't.
* :doc:`implement_aggregator`: write your own aggregation rule and test it
  in this simulation.
* :doc:`implement_attack`: write your own Byzantine attack and test it
  in this simulation.
* :doc:`structured_experiments`: collect structured results across
  multiple configurations with ``Metric`` and ``Orchestrator``.
* :doc:`/reference/simulations/decentralised/index`: the full
  decentralised simulation reference.
