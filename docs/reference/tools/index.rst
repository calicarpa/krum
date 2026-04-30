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
   :no-index:
