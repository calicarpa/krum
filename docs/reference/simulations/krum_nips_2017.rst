Krum — NIPS 2017
================

.. module:: krum.simulations.krum_nips_2017

Reproduces the experimental evaluation from:

    Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer.
    *"Machine learning with adversaries: Byzantine tolerant gradient descent."*
    In Advances in Neural Information Processing Systems 30 (NIPS 2017).

Overview
--------

This simulation package evaluates the **Krum** aggregation rule against Byzantine
workers in a synchronous parameter-server distributed SGD setting. It compares
Average, Krum, and Multi-Krum under Gaussian and Omniscient attacks on Spambase
and MNIST datasets.

Architecture:

.. code-block:: text

   Parameter Server ← gradients from n workers (up to f Byzantine)
          │
          │ broadcast parameters
          ▼
   ┌───────────────────────────────────────────────────────────┐
   │   Worker 1   │   Worker 2   │   ...   │     Worker n      │
   │   (honest)   │   (honest)   │         │ (maybe Byzantine) │
   └───────────────────────────────────────────────────────────┘
          │
          ▼
     Local SGD on dataset shard

Models
------

.. list-table:: Model Architectures
   :header-rows: 1
   :widths: 20 40 20 20

   * - Model
     - Architecture
     - Parameters
     - Dataset
   * - ``MLPSpambase``
     - 57 → 20 (ReLU) → 20 (ReLU) → 2
     - ≈ 1.6 × 10³
     - Spambase
   * - ``MLPMnist``
     - 784 → 100 (ReLU) → 10
     - ≈ 8 × 10⁴
     - MNIST

Experiments
-----------

Experiment 1 — Resilience to Byzantine Processes (Fig. 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compares Average and Krum under 0% and 33% Gaussian Byzantine workers.

.. list-table::
   :widths: 30 70

   * - **Attack**
     - Gaussian Byzantine (``std=200``), f = 33% (6 Byzantine out of 20)
   * - **Dataset**
     - Spambase
   * - **Mini-batch size**
     - 3 per worker
   * - **Compare**
     - Average vs Krum, with and without Byzantine workers (f = 0, f = 6)
   * - **Metric**
     - Cross-validation error over rounds
   * - **Expected**
     - Krum with 33% Byzantine matches Average with 0% Byzantine

Run with:

.. code-block:: bash

   python -m krum.simulations.krum_nips_2017.experiment_1

Experiment 2 — Cost of Resilience (Fig. 5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sweeps mini-batch sizes comparing Average and Krum under 0% and 45% Byzantine workers.

.. list-table::
   :widths: 30 70

   * - **Attack**
     - Omniscient Byzantine (``kappa=100``), f = 45% (9 Byzantine out of 20)
   * - **Datasets**
     - Spambase and MNIST
   * - **Mini-batch sizes**
     - 3, 5, 10, 20, 40, 80, 160 (log₂ scale)
   * - **Compare**
     - Average vs Krum, for f = 0 and f = 9
   * - **Metric**
     - Cross-validation error at round 500 vs batch size
   * - **Expected**
     - With batch size ≥ 10, Krum with 45% Byzantine matches Average with 0%

Run with:

.. code-block:: bash

   python -m krum.simulations.krum_nips_2017.experiment_2

Experiment 3 — Multi-Krum (Fig. 6)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compares Average (0%), Krum (33%), and Multi-Krum (33%) under Gaussian attack.

.. list-table::
   :widths: 30 70

   * - **Attack**
     - Gaussian Byzantine (``std=200``), f = 33% (6 Byzantine out of 20)
   * - **Dataset**
     - Spambase
   * - **Mini-batch size**
     - 3 per worker
   * - **Multi-Krum parameter**
     - ``m = n − f − 2 = 12``
   * - **Compare**
     - Average (f = 0), Krum (f = 6), Multi-Krum (f = 6)
   * - **Expected**
     - Multi-Krum with 33% Byzantine converges as fast as Average with 0%

Run with:

.. code-block:: bash

   python -m krum.simulations.krum_nips_2017.experiment_3

Hyperparameters
---------------

.. list-table:: Common Hyperparameters
   :widths: 40 60

   * - Learning rate
     - 0.01
   * - Optimizer
     - SGD (no momentum, no weight decay)
   * - Number of workers ``n``
     - 20
   * - Byzantine ratios ``f/n``
     - 0%, 33% (f = 6), 45% (f = 9)
   * - Rounds ``T``
     - 500
   * - Gaussian attack ``std``
     - 200
   * - Omniscient attack ``kappa``
     - 100
   * - Evaluation interval
     - Every 10 rounds
   * - Random seed
     - 42

Output Artifacts
----------------

Per simulation run, when ``results_dir`` is configured:

PyTorch trace (``{label}.pt``):

.. code-block:: python

   {
       "traces": [(round, error, loss), ...],
       "label": "Krum_f6",
       "seed": 42
   }

Figures:

- **Figure 4** (``figure_4.png``): Two side-by-side subplots — 0% and 33% Byzantine,
  error vs rounds, Average vs Krum.
- **Figure 5** (``figure_5.png``): Two side-by-side subplots — Spambase and MNIST,
  error at round 500 vs batch size (log₂ scale).
- **Figure 6** (``figure_6.png``): Single plot — Average (0%), Krum (33%),
  Multi-Krum (33%), error vs rounds.

API Reference
-------------

.. automodule:: krum.simulations.krum_nips_2017.simulation
   :members:
   :undoc-members:
   :show-inheritance:
