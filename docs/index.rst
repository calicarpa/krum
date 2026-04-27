Welcome to Krum's documentation!
================================

Krum is a research framework for Byzantine-resilient distributed learning. It provides a modular environment for comparing robust aggregation rules, simulating Byzantine attacks, running experiment campaigns, and analyzing results.

The core idea is simple:

1. An entry script builds a model, dataset, loss, criterion, and optimizer
2. Each training step produces honest gradients
3. An attack generates Byzantine gradients
4. An aggregation rule combines all gradients
5. Results are measured, plotted, and saved to tabular files or JSON
6. Post-processing scripts read these artifacts to generate visualizations

.. toctree::
   :maxdepth: 2
   :caption: Overview

   overview
   architecture

.. toctree::
   :maxdepth: 2
   :caption: Core Modules

   aggregators
   attacks
   experiments
   tools

.. toctree::
   :maxdepth: 2
   :caption: Native Acceleration

   native

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`