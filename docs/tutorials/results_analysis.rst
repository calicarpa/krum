Analysing experiment results
============================

The :doc:`working_with_orchestrator` tutorial showed how to collect structured
metrics with ``Metric`` and ``Orchestrator``. Once you have a
:class:`~krum.orchestration.dataframe.MetricDataFrame`, the real work begins.

This tutorial assumes you have ``pandas``, ``matplotlib``, and ``seaborn``
installed (available via ``pip install krum[experiments]``).

Recap: from Orchestrator to a frame
-----------------------------------

After a campaign, each metric channel is a
:class:`~krum.orchestration.dataframe.MetricDataFrame`:

.. code-block:: python

   orchestrator = Orchestrator("my_campaign")
   # ... runs ...

   frame = orchestrator.get("test_accuracy").to_pandas()
   print(frame.columns)  # parameter columns + "step" + "value"

Filtering
---------

:meth:`~krum.orchestration.dataframe.MetricDataFrame.filter` returns a narrowed
copy matching the given parameter values:

.. code-block:: python

   acc = orchestrator.get("test_accuracy")

   # Single parameter
   robust = acc.filter(label="MultiKrum (robust)").to_pandas()

   # Multiple parameters — all must match (AND)
   subset = acc.filter(label="MultiKrum (robust)", f=2).to_pandas()

   # Chaining
   no_attack = acc.filter(f=0).to_pandas()
   final_step = no_attack[no_attack["step"] == 49]

Unknown parameter names produce an empty frame. Filtering is eager but the
copies are shallow (samples are immutable scalars), so chaining is cheap.

Merging metrics
---------------

Each ``orchestrator.get(name)`` returns a single metric. To compare multiple
metrics side by side, merge their frames on ``step`` and parameter columns:

.. code-block:: python

   acc = orchestrator.get("test_accuracy").to_pandas()
   loss = orchestrator.get("test_loss").to_pandas()

   merged = acc.merge(
       loss,
       on=[c for c in acc.columns if c not in ("value",)],
       suffixes=("_acc", "_loss"),
   )
   print(merged.head())

The merged frame has ``value_acc`` and ``value_loss`` columns alongside the
parameter columns and ``step``.

Pivoting for comparison
-----------------------

To compare two runs directly, pivot so each run's metric is a column:

.. code-block:: python

   acc = orchestrator.get("test_accuracy").to_pandas()

   pivoted = acc.pivot_table(
       index="step",
       columns="label",
       values="value",
   )
   print(pivoted.head())

The resulting frame has one row per step and one column per run label.
This shape is ideal for plotting.

Exporting
---------

Write the merged or pivoted frame to disk for later analysis:

.. code-block:: python

   # CSV (one metric)
   orchestrator.get("test_accuracy").to_pandas().to_csv("accuracy.csv", index=False)

   # Merged frame
   merged.to_csv("full_results.csv", index=False)

   # JSON
   pivoted.to_json("accuracy_pivoted.json")

Aggregating across seeds
------------------------

Run the same configuration with different seeds, then compute mean and
standard deviation:

.. code-block:: python

   for seed in range(5):
       orchestrator.run(
           run_experiment,
           label="MultiKrum",
           aggregator=MultiKrum,
           f=2, lr=0.01, seed=seed,
       )

   acc = orchestrator.get("test_accuracy").to_pandas()
   stats = (
       acc.groupby(["step", "label"])["value"]
       .agg(["mean", "std"])
       .reset_index()
   )
   print(stats[stats["label"] == "MultiKrum"].tail())

Use the ``std`` column for error bars or confidence bands in plots.

Plotting recipes
----------------

.. code-block:: python

   import matplotlib.pyplot as plt
   import seaborn as sns

   acc = orchestrator.get("test_accuracy").to_pandas()

   # Compare runs by label
   sns.lineplot(data=acc, x="step", y="value", hue="label")
   plt.title("Test accuracy per configuration")
   plt.show()

   # Mean +/- std band across seeds
   stats = (
       acc.groupby(["step", "label"])["value"]
       .agg(["mean", "std"])
       .reset_index()
   )
   for label in stats["label"].unique():
       subset = stats[stats["label"] == label]
       plt.plot(subset["step"], subset["mean"], label=label)
       plt.fill_between(
           subset["step"],
           subset["mean"] - subset["std"],
           subset["mean"] + subset["std"],
           alpha=0.2,
       )
   plt.title("Accuracy with std band")
   plt.legend()
   plt.show()

   # Two metrics on shared axes
   merged = acc.merge(
       orchestrator.get("test_loss").to_pandas(),
       on=[c for c in acc.columns if c not in ("value",)],
       suffixes=("_acc", "_loss"),
   )
   fig, ax1 = plt.subplots()
   ax2 = ax1.twinx()
   for label in merged["label"].unique():
       subset = merged[merged["label"] == label]
       ax1.plot(subset["step"], subset["value_acc"], label=f"{label} (acc)", color="C0")
       ax2.plot(subset["step"], subset["value_loss"], label=f"{label} (loss)", color="C1", linestyle="--")
   ax1.set_ylabel("accuracy")
   ax2.set_ylabel("loss")
   fig.legend()
   plt.show()

Inspecting available metrics
----------------------------

List every channel collected during a campaign:

.. code-block:: python

   print(orchestrator.metrics())
   # e.g. ['test_accuracy', 'test_loss', 'train_loss']

   for name in orchestrator.metrics():
       frame = orchestrator.get(name).to_pandas()
       print(f"{name}: {len(frame)} samples")

Next steps
----------

* :doc:`working_with_orchestrator` — setting up ``Metric`` and
  ``Orchestrator`` for the first time.
* :doc:`/reference/orchestration/metricdataframe` — the full
  ``MetricDataFrame`` API.
