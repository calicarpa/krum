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
   "``tools/``", "Cross-cutting utilities", "Tensor flatten/relink, logging, parallelism"
   "``primitives/``", "Core abstractions", "Model wrapper (zero-copy flat views)"
