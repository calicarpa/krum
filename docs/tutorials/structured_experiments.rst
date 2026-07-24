Structured experiments
======================

**Problem:** Running a single simulation is fine for quick tests, but
research requires comparing configurations, collecting metrics at every
step, analysing results across seeds, and producing tables for papers.
How do you go from a one-off run to a reproducible, structured
experiment?

Krum provides three tools for this:

* :class:`~krum.orchestration.metric.Metric`: a named channel you push
  ``(step, value)`` samples into during a run.
* :class:`~krum.orchestration.orchestrator.Orchestrator`: drives multiple
  runs, owns all collected metrics, and returns them as a
  :class:`~krum.orchestration.dataframe.MetricDataFrame`.
* :class:`~krum.orchestration.dataframe.MetricDataFrame`: a filtered,
  queryable view of one metric channel, convertible to ``pandas``.

The Metric object
-----------------

A :class:`~krum.orchestration.metric.Metric` is created inside an experiment
function with a name and a value type:

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

The metric is just a **write handle**; it does not store the data itself.
Every push is routed to the orchestrator that is running the current
experiment.

The Orchestrator
-----------------

An :class:`~krum.orchestration.orchestrator.Orchestrator` runs a function multiple times
with different parameters and collects all the metrics pushed during each run:

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

How Metric, Orchestrator, and MetricDataFrame work together
------------------------------------------------------------

These three objects form a pipeline: you push data through a ``Metric``,
it lands in the ``Orchestrator``'s internal store tagged with run
parameters, and you retrieve a filtered view via ``Orchestrator.get()``
which returns a ``MetricDataFrame``.

The flow::

   Orchestrator.run(fn, label="A", seed=42, …)
          │
          ▼
   ┌────────────────────────────────────────┐
   │ fn(**params)                           │
   │                                        │
   │  Metric("acc").push(step, 0.95)        │
   │         │                              │
   │         │  thread-local context        │
   │         ▼                              │
   │  Orchestrator._record()                │
   │         │                              │
   │         ▼                              │
   │  Internal store                        │
   │  ┌─────┬──────┬──────┬───────┬──────┐  │
   │  │name │ step │  val │ label │ seed │  │
   │  ├─────┼──────┼──────┼───────┼──────┤  │
   │  │ acc │   0  │ 0.92 │   A   │  42  │  │
   │  │ acc │  10  │ 0.95 │   A   │  42  │  │
   │  │ acc │  20  │ 0.96 │   A   │  42  │  │
   │  │ acc │  10  │ 0.88 │   B   │  43  │  │
   │  └─────┴──────┴──────┴───────┴──────┘  │
   └────────────────────────────────────────┘
          │
          ▼
   Orchestrator.get("acc")
          │
          ▼
   MetricDataFrame  ─── .filter(label="A")
                        .to_pandas()  ───►  pandas.DataFrame

Key design decisions:

- **Orchestrator owns the data.** Metric is a light proxy that discovers
  the active orchestrator through thread-local state; you never pass the
  orchestrator to the metric explicitly.
- **MetricDataFrame is a lazy view.** ``filter()`` chains without copying
  data; ``to_pandas()`` materialises only at the end.

A complete example
------------------

The following experiment runs a Krum simulation twice: once with a robust
aggregator and once with the Average baseline, collecting the results as
structured metrics.

Setup
^^^^^

Imports, MNIST, and an MLP:

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

Experiment function
^^^^^^^^^^^^^^^^^^^

Creates the simulation, loops over rounds, and
pushes metrics. The function accepts every configurable parameter so it
can be driven by the ``Orchestrator``:

.. code-block:: python

   def run_experiment(
       *,
       label: str,
       aggregator,
       attack,
       f: int,
       n: int = 10,
       lr: float = 0.01,
       seed: int = 42,
       attack_kwargs: dict | None = None,
       rounds: int = 50,
       batch_size: int = 64,
       eval_every: int = 10,
   ) -> None:
       sim = KrumSimulation(
           model_cls=Krum2017MLPMnist,
           train_set=train_set, test_set=test_set,
           aggregator=aggregator, attack=attack,
           attack_kwargs=attack_kwargs,
           n=n, f=f, rounds=rounds,
           batch_size=batch_size, lr=lr, seed=seed,
       )
       sim.setup()

       test_loss = Metric("test_loss", float)
       test_accuracy = Metric("test_accuracy", float)
       train_loss = Metric("train_loss", float)

       for step in range(rounds):
           sim.step()
           if step % eval_every == 0 or step == rounds - 1:
               loss_val, acc_val = sim.evaluate()
               test_loss.push(step, loss_val)
               test_accuracy.push(step, acc_val)
               train_loss.push(step, sim.evaluate_train())

       print(f"  {label}: final accuracy {acc_val:.2%}")

Run the two configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^

Each ``orchestrator.run()`` call records
every parameter so the data is self-describing:

.. code-block:: python

   orchestrator = Orchestrator("mnist_comparison")

   orchestrator.run(
       run_experiment,
       label="MultiKrum (robust)",
       aggregator=MultiKrum,
       attack=SignFlipAttack,
       attack_kwargs={"scale": 1.5},
       f=2,
   )
   orchestrator.run(
       run_experiment,
       label="Average (non-robust)",
       aggregator=Average,
       attack=SignFlipAttack,
       attack_kwargs={"scale": 1.5},
       f=2,
   )

Inspect the results
^^^^^^^^^^^^^^^^^^^

``Orchestrator.get()`` returns a
``MetricDataFrame`` that supports filtering:

.. code-block:: python

   print("\nAll results (last 5 rows):")
   print(orchestrator.get("test_accuracy").to_pandas().tail(5))

   print("\nMultiKrum only:")
   print(orchestrator.get("test_accuracy").filter(label="MultiKrum (robust)").to_pandas())

Analysing results
-----------------

Once you have a :class:`~krum.orchestration.dataframe.MetricDataFrame`,
convert it to ``pandas`` and use your usual toolkit:

.. code-block:: python

   import matplotlib.pyplot as plt
   import seaborn as sns

   df = orchestrator.get("test_accuracy").to_pandas()

   # Filter to one configuration, get the final value
   final = df[df["step"] == 49]
   best = final.loc[final["value"].idxmax()]
   print(f"{best['label']}: {best['value']:.2%}")

   # Compare curves across labels
   sns.lineplot(data=df, x="step", y="value", hue="label")
   plt.title("Test accuracy per configuration")
   plt.show()

   # Pivot so each run is a column
   pivoted = df.pivot_table(index="step", columns="label", values="value")
   pivoted.to_csv("accuracy.csv")

See :doc:`/reference/orchestration/metricdataframe` for filtering options
and the :doc:`/reference/orchestration/index` for the full API.

Systematic benchmark
--------------------

Byzantine-robust research typically compares multiple aggregation rules
against multiple attacks on a shared dataset. This section shows how to
run such a benchmark with ``Orchestrator`` and produce a comparison table.

We build on the same MNIST + MLP setup from the previous example, but run
every combination of aggregators and attacks across multiple seeds:

.. code-block:: python

   from krum.primitives.aggregators.average import Average
   from krum.primitives.aggregators.median import Median
   from krum.primitives.aggregators.trimmed_mean import TrimmedMean
   from krum.primitives.aggregators.multikrum import MultiKrum
   from krum.primitives.attacks.sign_flip import SignFlipAttack
   from krum.primitives.attacks.alie import ALIEAttack
   from krum.primitives.attacks.gaussian import GaussianAttack

   orch = Orchestrator("mnist_benchmark")
   N, F, ROUNDS = 15, 3, 50
   SEEDS = [42, 43, 44]

   for agg in [Average, Median, TrimmedMean, MultiKrum]:
       for atk in [None, SignFlipAttack, ALIEAttack, GaussianAttack]:
           atk_label = atk.__name__ if atk else "NoAttack"
           label = f"{agg.__name__} + {atk_label}"
           for seed in SEEDS:
               orch.run(
                   run_experiment,
                   label=label, aggregator=agg, attack=atk,
                   f=F, n=N, lr=0.1, seed=seed,
               )

The ``Orchestrator`` records every run parameter (including the
``aggregator`` and ``attack`` classes), so we can group and pivot later
without parsing labels.

Building the comparison table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Grab the final-step accuracy, average across seeds, and pivot into a
matrix:

.. code-block:: python

   df = orch.get("test_accuracy").to_pandas()
   final = df[df["step"] == ROUNDS - 1]

   stats = final.groupby(["aggregator", "attack"])["value"].agg(["mean", "std"]).reset_index()

   table = stats.pivot_table(
       index="attack", columns="aggregator", values="mean",
   )
   table.index = [a.__name__ if a else "None" for a in table.index]

   print(table.round(2))

The output is a matrix where each cell is the mean accuracy for one
aggregator-attack pair, averaged across seeds. Use ``.std`` for error
bars in follow-up plots.

As a rule of thumb, robust aggregators (MultiKrum, TrimmedMean) maintain
high accuracy across attack types, while non-robust baselines (Average)
collapse. Results vary with ``n``, ``f``, model size, and dataset; real
papers report ``mean ± std`` over 5–10 seeds. See the aggregator and
attack docstrings for configuration-specific constraints
(e.g., minimum ``n`` for Bulyan, extra kwargs for attacks like
SmallPerturbation).

Next steps
---------

* :doc:`implement_aggregator`: write your own aggregation rule and
  benchmark it with the patterns from this tutorial.
* :doc:`implement_attack`: write your own Byzantine attack and
  benchmark it.
* :doc:`/reference/orchestration/index`: full Orchestrator and Metric API.
* :doc:`/reference/orchestration/metricdataframe`: available filtering
  options.
