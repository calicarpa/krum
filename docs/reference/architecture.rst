Repository Architecture
=======================

This page is a map of the repository for researchers who want to
navigate the codebase. Krum is organized into four main packages,
each with a clear responsibility.

Repository Map
--------------

.. csv-table::
   :header: "Location", "Main Role", "What Researchers Extend"
   :widths: 20, 40, 40

   "``aggregators/``", "Robust aggregation rules", "New rules, native variants, influence functions"
   "``attacks/``", "Byzantine attack strategies", "New attacks, custom perturbation models"
   "``primitives/``", "Core abstractions", "Model wrapper (zero-copy flat views), Attack base class"
   "``simulations/``", "Full-paper reproduction suites", "New experiments, custom datasets, custom architectures"

Core Design
-----------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Concept
     - Description
   * - Stateless API
     - Both aggregators and attacks are stateless. You call them as classmethods,
       no instantiation needed.
   * - Keyword-only parameters
     - Specialized parameters (``f``, ``n``, ``m``) are keyword-only, reducing
       the risk of misconfiguration.
   * - Validation
     - All aggregators validate their inputs and raise ``ValueError`` on misuse.
   * - Composition
     - Aggregators and attacks compose naturally with the ``Model`` wrapper and
       simulation framework.

Data Flow
---------

::

   Workers (honest + Byzantine)
        │
        │  gradients shape (d,)
        ▼
   Aggregator.aggregate(gradients, n, f)
        │
        │  robust gradient shape (d,)
        ▼
   Parameter Server (update model)
        │
        │  broadcast updated parameters
        ▼
   Workers (next iteration)

Package Overview
----------------

``krum.primitives``
~~~~~~~~~~~~~~~~~~~

The foundation layer providing:

- :class:`~primitives.model.Model` — zero-copy flat view wrapper around PyTorch models
- :class:`~primitives.attacks.Attack` — abstract base class for all attacks
- Concrete aggregators and attacks in sub-packages

``krum.primitives.aggregators``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Byzantine-resilient Gradient Aggregation Rules (GARs):

- :class:`~aggregators.average.Average` — arithmetic mean (no resilience, baseline)
- :class:`~aggregators.median.Median` — coordinate-wise median
- :class:`~aggregators.trimmed_mean.TrimmedMean` — coordinate-wise trimmed mean
- :class:`~aggregators.krum.Krum` — distance-based selection
- :class:`~aggregators.multikrum.MultiKrum` — multi-gradient averaging variant
- :class:`~aggregators.bulyan.Bulyan` — two-stage strong resilience
- :class:`~aggregators.brute.Brute` — optimal subset selection (baseline)
- :class:`~aggregators.geomed.GeoMed` — geometric median (medoid)

``krum.primitives.attacks``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Byzantine attack strategies:

- :class:`~attacks.sign_flip.SignFlip` — scaled gradients in opposite direction
- :class:`~attacks.alie.ALIE` — mean-shifted gradients using honest statistics
- :class:`~attacks.gaussian.Gaussian` — random Gaussian noise
- :class:`~attacks.omniscient.Omniscient` — negated full-dataset gradient
- :class:`~attacks.no_attack.NoAttack` — no-op baseline
- :class:`~attacks.small_perturbation.SmallPerturbation` — curse-of-dimensionality exploit

``krum.simulations``
~~~~~~~~~~~~~~~~~~~~

Full-paper reproduction suites built on
:class:`~krum.simulations.centralised.CentralisedSimulation`:

- :doc:`simulations/krum_nips_2017` — Blanchard et al., NIPS 2017
- :doc:`simulations/hidden_vulnerability_icml_2018` — El Mhamdi et al., ICML 2018

Each simulation package is a standalone example demonstrating how to use the
library's components to reproduce published results.
