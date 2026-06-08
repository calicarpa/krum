Simulations
===========

Krum ships two full-paper reproduction suites as simulation packages. Each package
is a standalone example demonstrating how to use the library's aggregators, attacks,
and model primitives to reproduce published Byzantine-resilient distributed learning
results.

.. note::

   These simulation packages are **examples**, not part of the public API.
   They demonstrate real-world usage patterns and reproduce published experiments.

Available Simulations
---------------------

.. list-table:: Simulation Packages
   :header-rows: 1
   :widths: 30 35 35

   * - Package
     - Paper
     - Experiments
   * - :doc:`krum_nips_2017`
     - Blanchard et al., NIPS 2017
     - 3 experiments on Spambase/MNIST
   * - :doc:`hidden_vulnerability_icml_2018`
     - El Mhamdi et al., ICML 2018
     - 3 experiments on MNIST/CIFAR-10

Both simulations inherit from the shared
:class:`~krum.simulations.centralised.CentralisedSimulation` base class, which
implements the parameter-server distributed SGD loop.

.. toctree::
   :maxdepth: 1
   :caption: Simulation Packages:

   krum_nips_2017
   hidden_vulnerability_icml_2018

CentralisedSimulation
---------------------

.. automodule:: krum.simulations.centralised
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _detect_device, _set_seed, _train_one_worker, _apply_robbins_monro_lr, _xavier_init_, _set_full_gradient_for_attack, _save_pt, _save_csv
