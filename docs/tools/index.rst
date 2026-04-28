Tools
=====

Utility modules for Krum providing infrastructure, tensor operations, and job management.

Overview
--------

.. list-table:: Tools
   :header-rows: 1
   :widths: 20 60 20

   * - Module
     - Description
     - Key Functions
   * - ``tools``
     - Core utilities (logging, exceptions, parsing)
     - Context, UserException, parse_keyval
   * - ``tools.pytorch``
     - PyTorch helpers (tensor operations, gradients)
     - flatten, relink, compute_avg_dev_max
   * - ``tools.misc``
     - Miscellaneous utilities (registries, timing)
     - pairwise, line_maximize, fullqual
   * - ``tools.jobs``
     - Job management for experiments
     - Command, Jobs, dict_to_cmdlist

Quick Start
-----------

.. code-block:: python

    import tools

    # Logging with context
    with tools.Context("training", "info"):
        tools.info("Starting training...")

    # Parse CLI arguments
    args = tools.parse_keyval(["lr:0.01", "batch:32"])

    # Flatten model parameters
    from tools import flatten
    flat_params = flatten(model.parameters())

    # Run jobs
    from tools import Command, Jobs
    cmd = Command(["python", "train.py"])
    jobs = Jobs("./results", devices=["cuda:0"])

.. seealso::

   For the training loop that uses these utilities, see
   :doc:`experiments/index`.
   For aggregation rules that rely on ``tools.pairwise``, see
   :doc:`aggregators/index`.

Available Tools
---------------

.. toctree::
   :maxdepth: 1
   :caption: Utility Modules:

   classes/pytorch
   classes/misc
   classes/jobs

API Reference
-------------

.. automodule:: tools
   :members:
   :undoc-members:
   :show-inheritance:
