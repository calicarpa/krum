Centralised simulation walkthrough
==================================

**Problem:** You need to run a parameter-server simulation with
multiple workers, a gradient aggregator, and Byzantine attacks, but
you are not sure how to configure the training loop.

Krum ships with ready-to-use centralised simulations that handle
the worker loop, gradient computation, and evaluation for you.

.. seealso::

   :doc:`/reference/simulations/centralised/index`
      Reference for :class:`~krum.simulations.centralised.KrumSimulation`
      and :class:`~krum.simulations.centralised.HiddenVulnerabilitySimulation`.

Minimal example
---------------

Instantiation
^^^^^^^^^^^^^

Aggregator and attack are passed as **classes**, not instances.
The simulation calls ``aggregator.aggregate()`` and ``attack.generate()``
each round. Extra parameters go through ``aggregator_kwargs`` and
``attack_kwargs``:

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
        rounds=50,
        batch_size=64,
        lr=0.01,
        seed=42,
    )

Note that :class:`~krum.simulations.centralised.CentralisedSimulation` does
**not** provide a ``run(rounds)`` method. The training loop is always
manual so you can evaluate when you want (compare with the
:doc:`decentralised_simulation_walkthrough`, which does have ``run``).

Setup
^^^^^

``setup()`` initialises the model parameters, splits the training set
into IID shards (one per worker), and seeds all RNG. The result is
deterministic for a given ``seed``:

.. code-block:: python

   sim.setup()

Step
^^^^

Each call to ``step()`` runs one synchronous round:

* **Broadcast** the model to all :math:`n` workers.
* **Honest workers** compute gradients on their data shard.
* **Byzantine workers** generate attack gradients.
* **Aggregator** combines all :math:`n` gradients into one.
* **SGD update** is applied.

.. code-block:: python

   for round_idx in range(50):
       sim.step()
       if round_idx % 10 == 0:
           loss, accuracy = sim.evaluate()
           print(f"round {round_idx}: loss={loss:.4f}  accuracy={accuracy:.4f}")

Evaluate
^^^^^^^^

``evaluate()`` returns the metrics specific to the protocol
(``(test_loss, test_accuracy)`` for :class:`~krum.simulations.centralised.KrumSimulation`,
``(test_loss, test_error, test_accuracy)`` for
:class:`~krum.simulations.centralised.HiddenVulnerabilitySimulation`).
You can also read the training loss with ``evaluate_train()``.

Using the ICML 2018 simulation
------------------------------

Switching to the other built-in simulation changes only the import
and one extra parameter. The ICML 2018 variant adds Xavier weight
initialisation, L2 regularisation, and the Robbins-Monro learning-rate
schedule:

.. code-block:: python

   from krum.simulations.centralised.hidden_vulnerability_icml_2018 import (
       HiddenVulnerabilitySimulation,
   )

   sim = HiddenVulnerabilitySimulation(
        # same arguments as KrumSimulation ...
        r_eta=10.0,  # required by Robbins-Monro schedule
        rounds=50,
    )
   sim.setup()
   for round_idx in range(50):
       sim.step()
       if round_idx % 10 == 0:
           loss, error, accuracy = sim.evaluate()
           print(f"round {round_idx}: loss={loss:.4f}  error={error:.4f}  accuracy={accuracy:.4f}")

   train_loss = sim.evaluate_train()
   print(f"final training loss: {train_loss:.4f}")

``HiddenVulnerabilitySimulation`` also accepts ``stop_attack_at``. This is an
optional round index after which the Byzantine attack is disabled (used
by the ICML 2018 paper, Experiment 1). Pass ``stop_attack_at=50`` to
stop the attack after round 50 while continuing training.

Note that unlike the decentralised simulation, the centralised one
stores the round count at construction time (``rounds=50`` above) but
you drive the loop yourself. This lets you evaluate, log, or even change
hyperparameters between rounds.

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
