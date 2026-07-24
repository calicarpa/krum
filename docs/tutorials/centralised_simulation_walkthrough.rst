Centralised simulation walkthrough
==================================

**Problem:** You want to reproduce a published Byzantine resilience
experiment (e.g., Krum NIPS 2017) but don't know how to configure the
workers, aggregator, attack, or training loop.

Krum ships with ready-to-use centralised (parameter-server) simulations
that reproduce published protocols. This tutorial shows how to configure
and run them.

All centralised simulations share the lifecycle:
**instantiate → setup → step → evaluate**.

.. seealso::

   :doc:`/reference/simulations/centralised/index`
      Full reference for :class:`~krum.simulations.centralised.KrumSimulation`
      and :class:`~krum.simulations.centralised.HiddenVulnerabilitySimulation`.

Minimal example
---------------

.. code-block:: python

   from torchvision import datasets, transforms

   from krum.primitives.aggregators.multikrum import MultiKrum
   from krum.primitives.attacks.sign_flip import SignFlipAttack
   from krum.primitives.models.mlp import Krum2017MLPMnist
   from krum.simulations.centralised.krum_nips_2017 import KrumSimulation

   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,)),
   ])
   train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
   test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

   sim = KrumSimulation(
       model_cls=Krum2017MLPMnist,
       train_set=train_set,
       test_set=test_set,
       aggregator=MultiKrum,
       attack=SignFlipAttack,
       attack_kwargs={"scale": 1.5},
       n=10,
       f=2,
       batch_size=64,
       lr=0.01,
       seed=42,
   )
   sim.setup()

   for round_idx in range(50):
       sim.step()
       if round_idx % 10 == 0:
           loss, accuracy = sim.evaluate()
           print(f"round {round_idx}: loss={loss:.4f}  accuracy={accuracy:.4f}")

The lifecycle
-------------

#. **Instantiation**: pass the model class, datasets, aggregator, attack,
   and hyperparameters. Aggregator and attack are **classes**, not
   instances. The simulation calls their classmethods each round. Use
   ``aggregator_kwargs`` and ``attack_kwargs`` for extra parameters.

#. ``setup()``:
   initialises the model, splits the training set into IID shards (one per
   worker), and seeds all RNG. Deterministic for a given ``seed``.

#. ``step()``:
   runs one synchronous round:

   * Broadcast the model to all workers.
   * Honest workers compute gradients on their shard.
   * Byzantine workers generate attack gradients.
   * The aggregator combines all :math:`n` gradients.
   * SGD update is applied.

#. ``evaluate()``:
   returns the metrics specific to the protocol (loss, accuracy, etc.).

Using the ICML 2018 simulation
------------------------------

Switch to the other built-in simulation by changing the import and adding
the parameters it requires:

.. code-block:: python

   from krum.simulations.centralised.hidden_vulnerability_icml_2018 import (
       HiddenVulnerabilitySimulation,
   )

   sim = HiddenVulnerabilitySimulation(
       model_cls=Krum2017MLPMnist,
       train_set=train_set,
       test_set=test_set,
       aggregator=MultiKrum,
       attack=SignFlipAttack,
       attack_kwargs={"scale": 1.5},
       n=10,
       f=2,
       batch_size=64,
       lr=0.01,
       r_eta=10.0,  # required by Robbins-Monro schedule
       seed=42,
   )
   sim.setup()

   for round_idx in range(50):
       sim.step()
       if round_idx % 10 == 0:
           loss, error, accuracy = sim.evaluate()
           print(f"round {round_idx}: loss={loss:.4f}  error={error:.4f}  accuracy={accuracy:.4f}")

   train_loss = sim.evaluate_train()
   print(f"final training loss: {train_loss:.4f}")

The ICML 2018 variant returns three values: ``(test_loss, test_error,
test_accuracy)``, applies Xavier weight initialisation and L2 regularisation,
and uses the Robbins-Monro learning-rate schedule.

Comparing two configurations
----------------------------

Create a second simulation with the non-robust ``Average`` aggregator to see
the effect of Byzantine workers:

.. code-block:: python

   from krum.primitives.aggregators.average import Average

   baseline = KrumSimulation(
       model_cls=Krum2017MLPMnist,
       train_set=train_set,
       test_set=test_set,
       aggregator=Average,
       attack=SignFlipAttack,
       attack_kwargs={"scale": 1.5},
       n=10, f=2, batch_size=64, lr=0.01, seed=42,
   )
   baseline.setup()
   for _ in range(50):
       baseline.step()

   _, robust_acc = sim.evaluate()
   _, baseline_acc = baseline.evaluate()
   print(f"MultiKrum: {robust_acc:.2%}")
   print(f"Average:   {baseline_acc:.2%}")

Next steps
----------

* :doc:`implement_aggregator`: write your own aggregation rule and test it
  in this simulation.
* :doc:`implement_attack`: write your own Byzantine attack and test it
  in this simulation.
* :doc:`decentralised_simulation_walkthrough`: peer-to-peer simulations
  with per-worker models and model mixing.
* :doc:`structured_experiments`: collect structured results with
  ``Metric`` and ``Orchestrator``, from single runs to systematic benchmarks.
* :doc:`/reference/simulations/index`: all bundled experiment scripts
  reproducing published papers.
