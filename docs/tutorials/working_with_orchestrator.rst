Collecting results with Metric and Orchestrator
================================================

The :doc:`centralised_simulation_walkthrough` tutorial used
``sim.evaluate()`` and ``print()`` to show results. For real research you need
structured collection: run many configurations, record metrics at every step,
and analyse the results as a table.

Krum provides two classes for this:

* :class:`~krum.orchestration.metric.Metric` — a named channel you push ``(step, value)``
  samples into during a run.
* :class:`~krum.orchestration.orchestrator.Orchestrator` — drives multiple runs, owns all
  the collected metrics, and returns them as a
  :class:`~krum.orchestration.dataframe.MetricDataFrame`.

The Metric object
-----------------

A :class:`~krum.orchestration.metric.Metric` is created inside an experiment function
with a name and a value type:

.. code-block:: python

   from krum.orchestration import Metric

   loss = Metric("test_loss", dtype=float)
   accuracy = Metric("test_accuracy", dtype=float)

.. warning::

   :class:`~krum.orchestration.metric.Metric` can only be created **inside** an
   :meth:`~krum.orchestration.orchestrator.Orchestrator.run` call. Creating one outside an
   active run raises ``RuntimeError``. The metric is a write handle that
   routes every push to the orchestrator driving the current experiment.

Each call to :meth:`~krum.orchestration.metric.Metric.push` records one sample,
tagged with the current run's parameters:

.. code-block:: python

   loss.push(step=10, value=0.1523)
   accuracy.push(step=10, value=0.9531)

The metric is just a **write handle** — it does not store the data itself.
Every push is routed to the orchestrator that is running the current
experiment.

The Orchestrator
----------------

An :class:`~krum.orchestration.orchestrator.Orchestrator` runs a function multiple times
with different parameters and collects all the metrics pushed during each run.

.. code-block:: python

   from krum.orchestration import Orchestrator

   orchestrator = Orchestrator("my_campaign")

   for lr in [0.01, 0.001]:
       orchestrator.run(my_experiment, lr=lr, label=f"lr_{lr}")

Once all runs are finished, retrieve every sample of a metric:

.. code-block:: python

   frame = orchestrator.get("test_loss").to_pandas()
   print(frame)

The resulting ``pandas.DataFrame`` has one row per recorded step, with
columns for the run parameters (``label``, ``lr``, etc.), ``step``, and
``value``.

Filtering
~~~~~~~~~

Filter a :class:`~krum.orchestration.dataframe.MetricDataFrame` before
materialising the DataFrame:

.. code-block:: python

   subset = orchestrator.get("test_loss").filter(label="lr_0.01")
   filtered_frame = subset.to_pandas()

A complete example
------------------

The following experiment runs a Krum simulation twice — once with a robust
aggregator and once with the Average baseline — and collects the results as
structured metrics.

.. code-block:: python

   from krum.orchestration import Metric, Orchestrator
   from krum.primitives.aggregators.average import Average
   from krum.primitives.aggregators.multikrum import MultiKrum
   from krum.primitives.attacks.sign_flip import SignFlipAttack
   from krum.primitives.models.mlp import Krum2017MLPMnist
   from krum.simulations.centralised.krum_nips_2017 import KrumSimulation

   from torchvision import datasets, transforms

   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,)),
   ])
   train_set = datasets.MNIST(
       root="./data", train=True, download=True, transform=transform
   )
   test_set = datasets.MNIST(
       root="./data", train=False, download=True, transform=transform
   )


   def run_experiment(
       *,
       label: str,
       aggregator,
       f: int,
       lr: float,
   ) -> None:
       sim = KrumSimulation(
           model_cls=Krum2017MLPMnist,
           train_set=train_set,
           test_set=test_set,
           aggregator=aggregator,
           attack=SignFlipAttack,
           attack_kwargs={"scale": 1.5},
           n=10,
           f=f,
           rounds=50,
           batch_size=64,
           lr=lr,
           seed=42,
       )
       sim.setup()

       test_loss = Metric("test_loss", float)
       test_accuracy = Metric("test_accuracy", float)
       train_loss = Metric("train_loss", float)

       for step in range(50):
           sim.step()
           if step % 10 == 0 or step == 49:
               loss_val, acc_val = sim.evaluate()
               test_loss.push(step, loss_val)
               test_accuracy.push(step, acc_val)
               train_loss.push(step, sim.evaluate_train())

       print(f"  {label}: final accuracy {acc_val:.2%}")


   orchestrator = Orchestrator("mnist_comparison")

   orchestrator.run(
       run_experiment,
       label="MultiKrum (robust)",
       aggregator=MultiKrum,
       f=2,
       lr=0.01,
   )
   orchestrator.run(
       run_experiment,
       label="Average (non-robust)",
       aggregator=Average,
       f=2,
       lr=0.01,
   )


   print("\nAll results (last 5 rows):")
   print(orchestrator.get("test_accuracy").to_pandas().tail(5))

   print("\nMultiKrum only:")
   print(orchestrator.get("test_accuracy").filter(label="MultiKrum (robust)").to_pandas())

If you also have ``matplotlib`` installed (``pip install krum[experiments]``),
you can plot directly from the DataFrame:

.. code-block:: python

   import matplotlib.pyplot as plt
   import seaborn as sns

   frame = orchestrator.get("test_accuracy").to_pandas()
   sns.lineplot(data=frame, x="step", y="value", hue="label")
   plt.title("MultiKrum vs Average under sign-flip attack")
   plt.show()

Next steps
----------

* :doc:`centralised_simulation_walkthrough` — using the built-in simulations directly.
* :doc:`implement_simulation` — creating a custom simulation with a custom
  ``evaluate`` method.
* See :doc:`/reference/orchestration/index` for the full Orchestrator and
  Metric API.
* Check the :doc:`/reference/orchestration/metricdataframe` documentation for
  filtering options.
