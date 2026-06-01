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
   "``attacks/``", "Byzantine attack strategies", "New attacks, custom perturbation models"
   "``primitives/``", "Core abstractions", "Model wrapper (zero-copy flat views)"
   "``simulations/``", "Full-paper reproduction suites", "New experiments, custom datasets, custom architectures"
