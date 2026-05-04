Repository Architecture
=======================

This page is a map of the repository for researchers who want to
navigate the codebase.

Repository Map
--------------

.. csv-table::
   :header: "Location", "Main Role", "What Researchers Extend"
   :widths: 20, 40, 40

   "``aggregators/``", "Robust aggregation rules", "New rules, native variants, influence functions"
   "``attacks/``", "Byzantine attacks", "New attack directions, multi-worker scenarios"
   "``experiments/``", "Model / dataset / loss / criterion / optimizer", "New models, datasets, metrics"
   "``native/``", "Auto-compiled C++/CUDA extensions", "Fast implementations of hot paths"
   "``tools/``", "Cross-cutting utilities", "Logging, parallelism, parsing, registries"
