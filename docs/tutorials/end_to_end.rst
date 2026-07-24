End-to-end example
==================

**Problem:** You want to see the complete pipeline from synthetic data
to exported analysis table, in a single reference script you can adapt
to your own experiments.

This tutorial walks through a complete experiment from start to finish:
loading a custom dataset, defining a model, running simulations under
attack, collecting results with ``Orchestrator``, and exporting a
comparison table.

It combines the patterns from :doc:`custom_dataset`,
:doc:`centralised_simulation_walkthrough`, :doc:`working_with_orchestrator`,
and :doc:`results_analysis` into a single script.

Step 1: Dataset and model
-------------------------

.. code-block:: python

   import torch
   import torch.nn as nn
   from torch.utils.data import TensorDataset

   from krum.primitives.models import Model
   from krum.simulations.centralised import KrumSimulation

   # Synthetic dataset: 10 000 samples, 64 features, 10 classes
   x = torch.randn(10000, 64)
   y = torch.randint(0, 10, (10000,))
   train_set = TensorDataset(x, y)
   test_set = TensorDataset(x[:2000], y[:2000])

   class MLP(nn.Sequential):
       def __init__(self):
           super().__init__(
               nn.Flatten(),
               nn.Linear(64, 32), nn.ReLU(),
               nn.Linear(32, 10),
           )

Step 2: Experiment function
---------------------------

.. code-block:: python

   from krum.orchestration import Orchestrator, Metric
   from krum.aggregators import MultiKrum, Average
   from krum.attacks import SignFlipAttack, ALIEAttack

   def run_experiment(
       *,
       label: str,
       aggregator,
       attack,
       f: int,
       lr: float,
       seed: int,
   ) -> None:
       sim = KrumSimulation(
           model_cls=MLP,
           train_set=train_set,
           test_set=test_set,
           aggregator=aggregator,
           attack=attack,
           n=10, f=f,
           rounds=50,
           batch_size=32,
           lr=lr,
           seed=seed,
       )
       sim.setup()

       test_acc = Metric("test_accuracy", float)
       test_loss = Metric("test_loss", float)

       for step in range(50):
           sim.step()
           if step % 10 == 0 or step == 49:
               loss, acc = sim.evaluate()
               test_loss.push(step, loss)
               test_acc.push(step, acc)

Step 3: Run the sweep
---------------------

.. code-block:: python

   orch = Orchestrator("e2e_example")

   configs = [
       (Average, None, 2, 0.01),
       (MultiKrum, SignFlipAttack, 2, 0.01),
       (MultiKrum, ALIEAttack, 2, 0.01),
       (Average, SignFlipAttack, 2, 0.01),
   ]

   for agg, atk, f, lr in configs:
       atk_label = atk.__name__ if atk else "NoAttack"
       label = f"{agg.__name__} + {atk_label}"
       for seed in [42, 43, 44]:
           orch.run(
               run_experiment,
               label=label,
               aggregator=agg,
               attack=atk,
               f=f, lr=lr, seed=seed,
           )

Step 4: Analyse and export
--------------------------

.. code-block:: python

   acc = orch.get("test_accuracy").to_pandas()
   final = acc[acc["step"] == 49]

   stats = (
       final.groupby(["aggregator", "attack"])["value"]
       .agg(["mean", "std"])
       .reset_index()
   )

   table = stats.pivot_table(
       index="attack",
       columns="aggregator",
       values="mean",
   )
   print(table.round(3))

   stats.to_csv("e2e_results.csv", index=False)

Going further
-------------

This tutorial used built-in aggregators, attacks, and the default
``KrumSimulation``. To go beyond:

* :doc:`implement_aggregator` — write your own aggregation rule by
  subclassing ``Aggregator``
* :doc:`implement_attack` — write your own Byzantine attack by
  subclassing ``Attack``
* :doc:`implement_simulation` — create a custom simulation with a
  custom evaluation metric, learning-rate schedule, or communication
  topology
* :doc:`custom_dataset` — more dataset patterns (torchvision, disk
  loading)
* :doc:`results_analysis` — advanced plotting, merging, and
  multi-metric comparison
* :doc:`systematic_benchmark` — scaling the sweep to more aggregators
  and attacks
* :doc:`troubleshooting` — common errors and how to fix them
