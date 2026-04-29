Benchmarking aggregators against Byzantine attacks
==================================================

This tutorial shows you how to compare several robust aggregation rules
under the same Byzantine attack. By the end you will have produced a single
plot that ranks aggregators by their ability to keep accuracy high when a
subset of workers is malicious.

We use MNIST and a small convolutional network so that each run finishes in
a few minutes on a CPU. The methodology, however, applies to any model or
dataset.

.. note::

   This tutorial assumes you have completed :doc:`getting-started` and are
   familiar with ``train.py``, the ``results/`` directory layout, and the
   ``eval`` file format.

Step 1: Choose your candidates
------------------------------

We will benchmark four aggregation rules against the same attack:

- ``average`` — the baseline with no defence.
- ``median`` — a simple coordinate-wise robust rule.
- ``krum`` — distance-based selection.
- ``bulyan`` — two-stage robust aggregation.

All runs share the same hyper-parameters so that differences come only from
the aggregator:

- 11 workers total
- 4 declared Byzantine workers (``--nb-decl-byz 4``)
- 4 real Byzantine workers (``--nb-real-byz 4``)
- 300 training steps, evaluation every 100 steps
- NaN attack (blind but effective against averaging)

Step 2: Run the experiments
---------------------------

Create a small script ``benchmark.sh`` (or run the commands one by one):

.. code-block:: bash

   #!/bin/bash
   COMMON="--nb-workers 11 --nb-decl-byz 4 --nb-real-byz 4 \
           --attack nan --nb-steps 300 --evaluation-delta 100"

   python train.py $COMMON --gar average  --result-directory results/bench_average
   python train.py $COMMON --gar median   --result-directory results/bench_median
   python train.py $COMMON --gar krum     --result-directory results/bench_krum
   python train.py $COMMON --gar bulyan   --result-directory results/bench_bulyan

Wait for the four runs to finish. Each produces its own ``results/bench_*/``
folder containing ``config``, ``config.json``, ``eval``, and ``study``.

Step 3: Load and plot the results
---------------------------------

Create ``plot_benchmark.py``:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   from pathlib import Path

   def load_eval(directory):
       path = Path(directory) / "eval"
       data = np.loadtxt(path, skiprows=1)
       return data[:, 0], data[:, 1]   # steps, accuracy

   fig, ax = plt.subplots(figsize=(8, 5))

   styles = {
       "average":  ("tab:red",   "o", "Average (no defence)"),
       "median":   ("tab:blue",  "s", "Median"),
       "krum":     ("tab:green", "^", "Krum"),
       "bulyan":   ("tab:purple","D", "Bulyan"),
   }

   for name, (color, marker, label) in styles.items():
       steps, acc = load_eval(f"results/bench_{name}")
       ax.plot(steps, acc * 100, label=label, color=color,
               marker=marker, markevery=1)

   ax.set_xlabel("Training step")
   ax.set_ylabel("Test accuracy (%)")
   ax.set_title("Robustness comparison under NaN attack (4 / 11 Byzantine)")
   ax.legend(loc="lower right")
   ax.grid(True, linestyle="--", alpha=0.5)
   ax.set_ylim(0, 100)

   plt.tight_layout()
   plt.savefig("benchmark.png", dpi=150)
   print("Saved benchmark.png")

Run it::

   pip install matplotlib
   python plot_benchmark.py

What to expect
~~~~~~~~~~~~~~

Open ``benchmark.png``. You should see a pattern similar to this:

- **Average** collapses almost immediately and stays near 10 % (random guess).
- **Median** stabilises above random guess but plateaus lower than the
  distance-based rules.
- **Krum** keeps climbing and ends close to the clean baseline.
- **Bulyan** follows Krum closely, sometimes slightly above or below
  depending on the random seed.

The exact numbers will vary because MNIST training has stochasticity
(dropout, data shuffling), but the ranking is usually stable.

Step 4: Add a second attack
---------------------------

A single attack is not enough to declare a winner. Different rules have
different blind spots. Repeat the benchmark with the ``identical`` attack
using the ``little`` variant:

.. code-block:: bash

   COMMON2="--nb-workers 11 --nb-decl-byz 4 --nb-real-byz 4 \
            --attack identical --attack-args variant:little \
            --nb-steps 300 --evaluation-delta 100"

   python train.py $COMMON2 --gar average  --result-directory results/bench2_average
   python train.py $COMMON2 --gar median   --result-directory results/bench2_median
   python train.py $COMMON2 --gar krum     --result-directory results/bench2_krum
   python train.py $COMMON2 --gar bulyan   --result-directory results/bench2_bulyan

Update ``plot_benchmark.py`` to overlay both attacks (use dashed lines for
the second attack, or create a two-panel figure). You will likely notice
that ``median`` suffers more against ``identical`` than against ``NaN``,
while ``krum`` and ``bulyan`` remain stable.

Step 5: Draw conclusions
------------------------

When you present or publish results, readers expect more than a plot. Add
a short summary table:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Aggregator
     - NaN accuracy @ 300
     - Identical accuracy @ 300
     - Runtime (seconds)
   * - Average
     - ~10 %
     - ~10 %
     - fastest
   * - Median
     - ~60 %
     - ~35 %
     - fast
   * - Krum
     - ~90 %
     - ~88 %
     - medium
   * - Bulyan
     - ~90 %
     - ~89 %
     - medium

These numbers are illustrative — fill them in with your actual measurements.

What to do next
---------------

- Add ``brute`` to the benchmark. It is exponentially slower but gives an
  optimal baseline for small :math:`n`.
- Try a different number of Byzantine workers (e.g. 1, 2, 6) to see where
  each rule breaks down.
- Read :doc:`/how-to/add-aggregator` to implement your own rule and include
  it in the benchmark.
