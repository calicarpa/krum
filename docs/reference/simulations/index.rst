Simulations
===========

Krum ships two full-paper reproduction suites as simulation modules:

- **Krum — NIPS 2017**: Blanchard et al., "Machine learning with adversaries:
  Byzantine tolerant gradient descent." Three experiments evaluating Average /
  Krum / Multi-Krum under Gaussian and Omniscient attacks on Spambase and MNIST.

- **Hidden Vulnerability — ICML 2018**: El Mhamdi, Guerraoui, Rouault, "The
  Hidden Vulnerability of Distributed Learning in Byzantium." Three experiments
  evaluating six aggregators (Average through Bulyan) under ALIE, Gaussian,
  and Sign-flip attacks on MNIST and CIFAR-10.

Both simulations inherit from the shared
:class:`~simulations.centralised.CentralisedSimulation` base class.

CentralisedSimulation
---------------------

.. automodule:: simulations.centralised
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _detect_device, _set_seed, _train_one_worker, _set_full_gradient_for_attack
