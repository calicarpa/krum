Structured experiments
======================

**Problem:** Running a single simulation is fine for quick tests, but
research requires comparing configurations, collecting metrics at every
step, analysing results across seeds, and producing tables for papers.
And once a sweep takes an hour, you do not want a one-line change to
repeat the runs that were already fine. How do you go from a one-off run
to a reproducible, structured experiment?

Krum provides two tools for this:

* :class:`~krum.orchestration.metrics.Metric`: a named channel you push
  ``(step, value)`` samples into during a run.
* :class:`~krum.orchestration.Orchestrator`: drives multiple runs, keeps
  each one's output in a folder of its own, skips the runs it has already
  recorded, and reads metrics back as a ``pandas.DataFrame``.

The Metric object
-----------------

A :class:`~krum.orchestration.metrics.Metric` is created inside an
experiment function with a name and a value type:

.. code-block:: python

   from krum.orchestration import Metric

   loss = Metric("test_loss", dtype=float)
   accuracy = Metric("test_accuracy", dtype=float)

.. warning::

   :class:`~krum.orchestration.metrics.Metric` can only be created
   **inside** a run driven by
   :meth:`~krum.orchestration.Orchestrator.run`. Creating one outside an
   active job raises :exc:`~krum.orchestration.metrics.NoActiveJob`. The
   metric finds the running job through a context variable, so you never
   pass the orchestrator to it.

Each call to :meth:`~krum.orchestration.metrics.Metric.push` appends one
sample to that job's own file:

.. code-block:: python

   loss.push(step=10, value=0.1523)
   accuracy.push(step=10, value=0.9531)

Rows are flushed as they are pushed, so a run you interrupt leaves its
partial output readable on disk.

The Orchestrator
-----------------

An :class:`~krum.orchestration.Orchestrator` runs a function many times
with different parameters:

.. code-block:: python

   from krum.orchestration import Orchestrator

   orchestrator = Orchestrator("results/my_campaign")

   for lr in [0.01, 0.001]:
       orchestrator.run(my_experiment, lr=lr, label=f"lr_{lr}")

   frame = orchestrator.get("test_loss")
   print(frame)

Two things differ from a plain loop:

* :meth:`~krum.orchestration.Orchestrator.run` only **enqueues**. The
  queue is drained by :meth:`~krum.orchestration.Orchestrator.drain`, by
  the first :meth:`~krum.orchestration.Orchestrator.get`, or on leaving a
  ``with Orchestrator(...)`` block. Nothing runs before that.
* Each run gets a folder under the campaign directory, named by a key
  derived from its parameters *and* from the code it executes. **A run
  already recorded there is skipped**, so re-running the sweep only
  computes what actually changed.

:meth:`~krum.orchestration.Orchestrator.get` returns a
``pandas.DataFrame`` with one row per recorded step, columns for every run
parameter, and the ``step``, ``value`` and ``job_key`` of the sample.

How it fits together
--------------------

.. code-block:: text

   Orchestrator.run(fn, aggregator=MultiKrum, f=2, seed=42)   → enqueued
            │
            ▼   drained on the first get(), or on leaving the with block
   ┌──────────────────────────────────────────────────────┐
   │ key = hash(fn's code + its bound parameters)         │
   │                                                      │
   │ results/campaign/<key>/ already marked DONE?         │
   │     yes ──► skip, the recorded answer stands         │
   │     no  ──► fn(**params)                             │
   │               Metric("test_accuracy").push(0, 0.92)  │
   │                     │  context variable              │
   │                     ▼                                │
   │               <key>/metrics/test_accuracy.csv        │
   └──────────────────────────────────────────────────────┘
            │
            ▼
   Orchestrator.get("test_accuracy")    reads this sweep's folders
            │
            ▼
   pandas.DataFrame
   ┌──────┬───────┬────────────┬───┬──────┬──────────────┐
   │ step │ value │ aggregator │ f │ seed │ job_key      │
   ├──────┼───────┼────────────┼───┼──────┼──────────────┤
   │   0  │ 0.92  │ MultiKrum  │ 2 │  42  │ 3ca7515b…    │
   │  10  │ 0.95  │ MultiKrum  │ 2 │  42  │ 3ca7515b…    │
   └──────┴───────┴────────────┴───┴──────┴──────────────┘

Key design decisions:

- **A job's folder is named by its identity.** The identity covers the
  parameters and the code the run reaches, so asking the same question
  twice finds the answer already there.
- **Metric finds the job, not the other way round.** It resolves a context
  variable, so an experiment never threads a handle through its own call
  stack.
- **Reading is scoped to the sweep.** A campaign directory accumulates one
  folder per version of the code; :meth:`~krum.orchestration.Orchestrator.get`
  reads the runs *this* orchestrator enqueued, so two code versions are not
  mixed into one plot.

Parameters identify the run
---------------------------

Because the parameters are hashed into the job's identity, they should
*describe* the run rather than *be* its materials.

.. code-block:: python

   # Don't: the dataset is hashed into the key on every enqueue, and the
   # key then depends on the data's contents rather than on its name.
   train_datasets = IidPartitioner.partition(load_mnist(), n=10, seed=42)

   def run_experiment(aggregator, f):
       sim = KrumSimulation(train_datasets=train_datasets, ...)

   # Do: pass what names the data, and build it inside the function.
   def run_experiment(dataset, n, aggregator, f, seed):
       train_set, test_set = make_datasets(dataset)
       train_datasets = IidPartitioner.partition(train_set, n=n, seed=seed)
       sim = KrumSimulation(train_datasets=train_datasets, ...)

.. note::

   ``step``, ``value`` and ``job_key`` are the frame's own columns, so a
   parameter of one of those names is refused when the run is enqueued.

A complete example
------------------

The following experiment runs a Krum simulation twice: once with a robust
aggregator and once with the Average baseline, collecting the results as
structured metrics.

Experiment function
^^^^^^^^^^^^^^^^^^^

It builds its own data and simulation from plain parameters, loops over
rounds, and pushes metrics:

.. code-block:: python

   from krum.orchestration import Metric, Orchestrator
   from krum.primitives.aggregators.average import Average
   from krum.primitives.aggregators.multikrum import MultiKrum
   from krum.primitives.attacks.sign_flip import SignFlipAttack
   from krum.primitives.data_partitioners.iid import IidPartitioner
   from krum.primitives.models.mlp import Krum2017MLPMnist
   from krum.simulations.centralised.krum_nips_2017 import KrumSimulation

   from torchvision import datasets, transforms

   def make_datasets(root="./data"):
       transform = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,)),
       ])
       return (
           datasets.MNIST(root=root, train=True, download=True, transform=transform),
           datasets.MNIST(root=root, train=False, download=True, transform=transform),
       )

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
       train_set, test_set = make_datasets()
       train_datasets = IidPartitioner.partition(train_set, n=n, seed=seed)

       sim = KrumSimulation(
           model_cls=Krum2017MLPMnist,
           train_datasets=train_datasets, test_set=test_set,
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

Every parameter is recorded, so the data is self-describing:

.. code-block:: python

   orchestrator = Orchestrator("results/mnist_comparison")

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

   print(orchestrator.drain().report())

:meth:`~krum.orchestration.Orchestrator.drain` returns a summary saying
which runs executed, which were skipped, and why each one ran:

.. code-block:: text

   2 done
     done     3ca7515b22969224     7.102s  (no recorded result)
     done     ad74a48355e052fc     4.416s  (no recorded result)

Inspect the results
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   accuracy = orchestrator.get("test_accuracy")

   print("\nAll results (last 5 rows):")
   print(accuracy.tail(5))

   print("\nMultiKrum only:")
   print(accuracy[accuracy["label"] == "MultiKrum (robust)"])

Analysing results
-----------------

The frame is ordinary ``pandas``, so use your usual toolkit:

.. code-block:: python

   import matplotlib.pyplot as plt
   import seaborn as sns

   df = orchestrator.get("test_accuracy")

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

See the :doc:`/reference/orchestration/index` for the full API.

Running it again
----------------

Run the same script a second time and nothing is recomputed:

.. code-block:: text

   2 skipped

A run is repeated when there is no recorded result for it, when its last
attempt failed, when the environment it was recorded in has changed, or
when a function it called has since been edited.
:meth:`~krum.orchestration.Orchestrator.plan` reports what a sweep would
do without running any of it:

.. code-block:: python

   for decision in orchestrator.plan():
       print(decision)

.. code-block:: text

   skip 3ca7515b22969224
   run ad74a48355e052fc (uv_lock changed (651669ac047d… -> 81583a2f1641…))

Editing an aggregator changes the identity of the runs that use it, and
leaves the others alone — so a change to ``MultiKrum`` recomputes the
MultiKrum configurations and skips the Average ones. Pass ``force=True``
to the orchestrator to recompute regardless.

Systematic benchmark
--------------------

Byzantine-robust research typically compares multiple aggregation rules
against multiple attacks on a shared dataset. This section shows how to
run such a benchmark and produce a comparison table.

Running the grid
^^^^^^^^^^^^^^^^

Loop over every aggregator-attack-seed combination:

.. code-block:: python

   from krum.primitives.aggregators.average import Average
   from krum.primitives.aggregators.median import Median
   from krum.primitives.aggregators.trimmed_mean import TrimmedMean
   from krum.primitives.aggregators.multikrum import MultiKrum
   from krum.primitives.attacks.sign_flip import SignFlipAttack
   from krum.primitives.attacks.alie import ALIEAttack
   from krum.primitives.attacks.gaussian import GaussianAttack

   N, F, ROUNDS = 15, 3, 50
   SEEDS = [42, 43, 44]

   with Orchestrator("results/mnist_benchmark") as orch:
       for agg in [Average, Median, TrimmedMean, MultiKrum]:
           for atk in [None, SignFlipAttack, ALIEAttack, GaussianAttack]:
               atk_label = atk.__name__ if atk else "NoAttack"
               for seed in SEEDS:
                   orch.run(
                       run_experiment,
                       label=f"{agg.__name__} + {atk_label}",
                       aggregator=agg, attack=atk,
                       f=F, n=N, lr=0.1, seed=seed, rounds=ROUNDS,
                   )

Adding a fifth aggregator later re-runs only its twelve new combinations;
the forty-eight already recorded are skipped.

Building the comparison table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A class passed as a parameter reads back as its short name, so group on
the column directly:

.. code-block:: python

   df = orch.get("test_accuracy")
   final = df[df["step"] == ROUNDS - 1]

   stats = (
       final.groupby(["aggregator", "attack"], dropna=False)["value"]
       .agg(["mean", "std"])
       .reset_index()
   )

   table = stats.pivot_table(
       index="attack", columns="aggregator", values="mean", dropna=False,
   )
   print(table.round(2))

.. warning::

   ``attack=None`` reads back as a missing value, and ``groupby`` drops
   rows with missing keys unless you pass ``dropna=False``. Without it the
   no-attack baseline disappears from the table without a word. Passing a
   sentinel aggregator-free label instead of ``None`` avoids the question
   entirely.

The output is a matrix where each cell is the mean accuracy for one
aggregator-attack pair, averaged across seeds. Use ``std`` for error
bars in follow-up plots.

As a rule of thumb, robust aggregators (MultiKrum, TrimmedMean) maintain
high accuracy across attack types, while non-robust baselines (Average)
collapse. Results vary with ``n``, ``f``, model size, and dataset; real
papers report ``mean ± std`` over 5–10 seeds. See the aggregator and
attack docstrings for configuration-specific constraints
(e.g., minimum ``n`` for Bulyan, extra kwargs for attacks like
SmallPerturbation).

Next steps
----------

* :doc:`implement_aggregator`: write your own aggregation rule and
  benchmark it with the patterns from this tutorial.
* :doc:`implement_attack`: write your own Byzantine attack and
  benchmark it.
* :doc:`/reference/orchestration/index`: the full orchestration API.
