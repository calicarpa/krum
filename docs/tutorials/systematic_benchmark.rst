Systematic benchmark
===================

Byzantine-robust research typically compares multiple aggregation rules
against multiple attacks on a shared dataset. This tutorial shows how to
run such a benchmark with ``Orchestrator`` and produce a comparison table.

The full workflow: define an experiment function, sweep over a grid of
(aggregator, attack) configurations across several seeds, then pivot
the results into a human-readable table.

Setting up the sweep
--------------------

We use MNIST with a small MLP and ``n = 15`` workers, of which ``f = 3``
are Byzantine. This configuration respects the strictest guarantee among
the tested aggregators (Bulyan requires ``n > 4f + 2``):

.. code-block:: python

   import torch
   import torchvision
   import torchvision.transforms as transforms

   from krum.simulations.centralised import KrumSimulation
   from krum.aggregators import (
       Average, Median, TrimmedMean, MultiKrum, Bulyan, Aksel,
   )
   from krum.attacks import SignFlipAttack, ALIEAttack, GaussianAttack
   from krum.models import Krum2017MLPMnist
   from krum.orchestration import Orchestrator, Metric

   N = 15
   F = 3
   ROUNDS = 50
   SEEDS = [42, 43, 44]

   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,)),
   ])
   train_mnist = torchvision.datasets.MNIST(
       "data", train=True, download=True, transform=transform,
   )
   test_mnist = torchvision.datasets.MNIST(
       "data", train=False, download=True, transform=transform,
   )

The experiment function
-----------------------

A single run creates a ``KrumSimulation``, steps through it, and pushes
metrics into ``Metric`` handles:

.. code-block:: python

   def run_experiment(
       *,
       label: str,
       aggregator,
       attack,
       attack_kwargs: dict | None = None,
       f: int,
       n: int,
       lr: float,
       seed: int,
   ) -> None:
       sim = KrumSimulation(
           model_cls=Krum2017MLPMnist,
           train_set=train_mnist,
           test_set=test_mnist,
           aggregator=aggregator,
           attack=attack,
           attack_kwargs=attack_kwargs,
           n=n, f=f,
           rounds=ROUNDS,
           batch_size=64,
           lr=lr,
           seed=seed,
       )
       sim.setup()

       test_acc = Metric("test_accuracy", float)
       test_loss = Metric("test_loss", float)

       for step in range(ROUNDS):
           sim.step()
           if step % 10 == 0 or step == ROUNDS - 1:
               loss, acc = sim.evaluate()
               test_loss.push(step, loss)
               test_acc.push(step, acc)

Running the grid
----------------

We nest two loops: one over aggregators, one over attacks. Each run is
tagged by ``label`` so we can filter later:

.. code-block:: python

   orch = Orchestrator("mnist_benchmark")

   AGGREGATORS = [Average, Median, TrimmedMean, MultiKrum, Bulyan, Aksel]
   ATTACKS = [None, SignFlipAttack, ALIEAttack, GaussianAttack]

   for agg in AGGREGATORS:
       for atk in ATTACKS:
           atk_label = atk.__name__ if atk else "NoAttack"
           label = f"{agg.__name__} + {atk_label}"
           attack_kwargs = None
           for seed in SEEDS:
               orch.run(
                   run_experiment,
                   label=label,
                   aggregator=agg,
                   attack=atk,
                   attack_kwargs=attack_kwargs,
                   f=F, n=N,
                   lr=0.1,
                   seed=seed,
               )

The ``Orchestrator`` records every parameter (including ``aggregator`` and
``attack`` classes) per run, so we can filter and pivot later without
parsing the label.

Building the comparison table
-----------------------------

**Final accuracy per run.** We filter for the last step, group by
aggregator, attack, and seed, then average across seeds:

.. code-block:: python

   acc = orch.get("test_accuracy").to_pandas()
   final = acc[acc["step"] == ROUNDS - 1]

   summary = (
       final.groupby(["aggregator", "attack", "seed"])["value"]
       .mean()
       .reset_index()
   )
   stats = (
       summary.groupby(["aggregator", "attack"])["value"]
       .agg(["mean", "std"])
       .reset_index()
   )

**Pivot into a matrix** with aggregators as columns and attacks as rows:

.. code-block:: python

   table = stats.pivot_table(
       index="attack",
       columns="aggregator",
       values="mean",
   )

   # Rename attacks for display
   table.index = [a.__name__ if a else "None" for a in table.index]

   print(table.round(2))

The output looks like::

    aggregator      Average  Median  TrimmedMean  MultiKrum  Bulyan  Aksel
    attack
    None               0.97    0.97         0.97       0.97    0.97   0.97
    SignFlipAttack     0.10    0.91         0.93       0.96    0.95   0.96
    ALIEAttack         0.10    0.10         0.90       0.93    0.94   0.96
    GaussianAttack     0.10    0.96         0.96       0.96    0.96   0.96

Interpreting the table
----------------------

* **Average collapses** under every attack — as expected for a
  non-robust baseline.
* **Median resists SignFlip and Gaussian** but fails against ALIE
  when the shift is large enough.
* **TrimmedMean, MultiKrum, Bulyan, Aksel** maintain high accuracy
  across all tested attacks for this configuration.
* **Gaussian attack** is weak — isotropic noise with ``std=200`` on
  MNIST gradients (which are ~10× smaller) is easily filtered out by
  distance-based rules.

Results will vary with ``n``, ``f``, model size, and dataset. The table
above is a single seed-set snapshot; real papers run 5–10 seeds and
report ``mean ± std``.

Exporting
---------

Save the table for papers or reports:

.. code-block:: python

   # Full results (mean ± std)
   stats.to_csv("benchmark_summary.csv", index=False)

   # Pivoted matrix
   table.to_csv("benchmark_matrix.csv")

   # LaTeX
   print(table.to_latex(float_format="%.2f"))

Troubleshooting
---------------

* **Bulyan or MultiKrum raise** ``ValueError``:
  your ``n`` is too small for the chosen ``f``. See each aggregator's
  docstring for the exact bound.
* **Some attacks require extra arguments:**
  ``SmallPerturbationAttack`` needs ``attack_kwargs={"aggregator": agg,
  "n": n}``; ``FullGradientNegationAttack`` is omniscient and the
  simulation handles it when the attack class is detected.
* **Brute is very slow:** it tests all :math:`\binom{n}{f}` combinations.
  Use it only for small ``n`` (≤ 8) and a separate sweep.
* **Different datasets or models** need different hyperparameters
  (learning rate, batch size, rounds). Start from the values used in
  the original paper for that model.

Next steps
----------

* :doc:`results_analysis` — filtering, pivoting, and plotting
  ``MetricDataFrame`` in more detail.
* :doc:`using_aggregators_attacks` — resilience bounds and the full
  list of built-in aggregation rules and attacks.
* :doc:`working_with_orchestrator` — the ``Orchestrator`` and
  ``Metric`` API.
