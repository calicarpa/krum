Using Aggregators and Attacks
=============================

**Problem:** You need to pick an aggregation rule and an attack strategy
for your experiment, but the library offers many options. How do you
choose the right ones and understand their resilience guarantees?

This tutorial covers all built-in aggregation rules and attack strategies
available in Krum.

Aggregators
-----------

Aggregators are **stateless** gradient aggregation rules. Call them as classmethods:

.. code-block:: python

   from krum.primitives.aggregators.average import Average
   from krum.primitives.aggregators.median import Median
   from krum.primitives.aggregators.trimmed_mean import TrimmedMean
   from krum.primitives.aggregators.krum import Krum
   from krum.primitives.aggregators.multikrum import MultiKrum
   from krum.primitives.aggregators.bulyan import Bulyan
   from krum.primitives.aggregators.aksel import Aksel
   from krum.primitives.aggregators.geomed import GeoMed

   # Simple average (baseline, no resilience)
   result = Average.aggregate(gradients)

   # Coordinate-wise median (basic resilience)
   result = Median.aggregate(gradients)

   # Geometric median (basic resilience; n and f accepted for API uniformity)
   result = GeoMed.aggregate(gradients, n=10, f=2)

   # Trimmed mean (basic resilience, requires 2f+1 workers)
   result = TrimmedMean.aggregate(gradients, f=2)

   # Krum (moderate resilience, requires 2f+3 workers)
   result = Krum.aggregate(gradients, n=10, f=2)

   # Multi-Krum (moderate resilience, averages m = n - 2f - 3 gradients by default)
   result = MultiKrum.aggregate(gradients, n=10, f=2)

   # Bulyan (strong resilience, two-stage, requires 4f+3 workers)
   result = Bulyan.aggregate(gradients, n=15, f=2)

   # Aksel (optimal breakdown point, requires n > 2f)
   result = Aksel.aggregate(gradients, f=2)

Available aggregators
~~~~~~~~~~~~~~~~~~~~~

* :class:`~krum.primitives.aggregators.average.Average` computes the plain mean over all
  gradients; non-robust baseline (McMahan et al., AISTATS 2017).
* :class:`~krum.primitives.aggregators.median.Median` takes the coordinate-wise
  median (Yin et al., ICML 2018).
* :class:`~krum.primitives.aggregators.geomed.GeoMed` selects the geometric median
  (medoid): the submitted gradient that minimises the sum of
  distances to all others (Yin et al., ICML 2018).
* :class:`~krum.primitives.aggregators.trimmed_mean.TrimmedMean` drops the
  :math:`f` smallest and :math:`f` largest values per coordinate, then
  averages the rest (Yin et al., ICML 2018).
* :class:`~krum.primitives.aggregators.aksel.Aksel` averages the
  :math:`n - f` gradients closest to the coordinate-wise median; optimal
  breakdown point in :math:`\mathcal{O}(nd)` (Boussetta et al., OPODIS 2020).
* :class:`~krum.primitives.aggregators.krum.Krum` selects the single
  gradient with the smallest sum of squared distances to its closest peers
  (Blanchard et al., NIPS 2017).
* :class:`~krum.primitives.aggregators.multikrum.MultiKrum` averages the
  :math:`m` best Krum-scored gradients (Blanchard et al., NIPS 2017).
* :class:`~krum.primitives.aggregators.brute.Brute` finds the exact most-clumped
  :math:`n - f` subset, exponential in cost.
* :class:`~krum.primitives.aggregators.bulyan.Bulyan` uses a two-stage rule:
  MultiKrum pre-selection, then a coordinate-wise trimmed average
  (El Mhamdi et al., ICML 2018).
* :class:`~krum.primitives.aggregators.nearest_neighbor_average.NearestNeighborAverage`
  averages the ``num_closest`` vectors nearest to a per-worker ``pivot``;
  the default mixing rule of the decentralised MoNNA simulation.

Input shape
~~~~~~~~~~~

All aggregators expect a 2D tensor of shape ``(n, d)`` where:

* ``n``: number of workers
* ``d``: gradient dimension (total number of parameters)

The output is a 1D tensor of shape ``(d,)``.

Resilience guarantees
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Aggregator
     - Resilience
     - Requirement
   * - :class:`~krum.primitives.aggregators.average.Average`
     - None (baseline)
     - ``n >= 1``
   * - :class:`~krum.primitives.aggregators.median.Median`
     - ``f < n / 2``
     - ``n >= 1``
   * - :class:`~krum.primitives.aggregators.geomed.GeoMed`
     - ``f < n / 2``
     - ``n >= 1``
   * - :class:`~krum.primitives.aggregators.trimmed_mean.TrimmedMean`
     - ``f < n / 2``
     - ``n >= 2f + 1``
   * - :class:`~krum.primitives.aggregators.aksel.Aksel`
     - ``f < n / 2`` (optimal breakdown point)
     - ``n > 2f``
   * - :class:`~krum.primitives.aggregators.krum.Krum`
     - ``2f + 2 < n``
     - ``n >= 2f + 3``
   * - :class:`~krum.primitives.aggregators.multikrum.MultiKrum`
     - ``2f + 2 < n``
     - ``n >= 2f + 3``
   * - :class:`~krum.primitives.aggregators.brute.Brute`
     - ``f < n / 2`` (exact, exponential cost)
     - ``n >= 2f + 1`` and ``f >= 1``
   * - :class:`~krum.primitives.aggregators.bulyan.Bulyan`
     - ``4f + 2 < n``
     - ``n >= 4f + 3``

Two rules deserve a caveat:

* :class:`~krum.primitives.aggregators.brute.Brute` enumerates all
  :math:`\binom{n}{n-f}` subsets to find the most clumped one. It is exact but
  only feasible for small worker counts.
* :class:`~krum.primitives.aggregators.nearest_neighbor_average.NearestNeighborAverage`
  (not in the table) averages the ``num_closest`` vectors nearest to a
  per-worker ``pivot``. It is the default mixing rule of the decentralised
  MoNNA simulation, not a parameter-server aggregator.

Attacks
-------

Attacks generate Byzantine gradients from honest worker gradients:

.. code-block:: python

   from krum.primitives.aggregators.multikrum import MultiKrum
   from krum.primitives.attacks.sign_flip import SignFlipAttack
   from krum.primitives.attacks.alie import ALIEAttack
   from krum.primitives.attacks.gaussian import GaussianAttack
   from krum.primitives.attacks.small_perturbation import SmallPerturbationAttack
   from krum.primitives.attacks.full_gradient_negation import FullGradientNegationAttack

   # Sign flip attack
   byzantine = SignFlipAttack.generate(honest_gradients, f=2, scale=1.5)

   # ALIE (A Little Is Enough) attack
   byzantine = ALIEAttack.generate(honest_gradients, f=2, z=2.0)

   # Gaussian attack
   byzantine = GaussianAttack.generate(honest_gradients, f=2, std=10.0)

   # Full gradient negation attack (requires the full-dataset gradient)
   byzantine = FullGradientNegationAttack.generate(
       honest_gradients, f=2, full_gradient=full_grad
   )

   # Small perturbation attack (targets a specific aggregator)
   byzantine = SmallPerturbationAttack.generate(
       honest_gradients, f=2, aggregator=MultiKrum, n=10
   )

All attacks follow the same pattern: pass the honest gradients and the number
of Byzantine workers ``f``, and they return a tensor of shape ``(f, d)``.

Available attacks
~~~~~~~~~~~~~~~~~

* :class:`~krum.primitives.attacks.sign_flip.SignFlipAttack` sends the
  sign-flipped honest mean, scaled by ``scale`` (Xie et al., AISTATS 2019).
* :class:`~krum.primitives.attacks.alie.ALIEAttack` shifts the honest mean
  by :math:`z` standard deviations, small enough to stay hidden inside the
  honest distribution (Baruch et al., NeurIPS 2019).
* :class:`~krum.primitives.attacks.gaussian.GaussianAttack` applies isotropic
  Gaussian noise, independent of the honest gradients.
* :class:`~krum.primitives.attacks.full_gradient_negation.FullGradientNegationAttack`
  sends the negated full-dataset gradient, scaled by ``kappa``; an
  omniscient attack that assumes knowledge of the entire dataset.
* :class:`~krum.primitives.attacks.small_perturbation.SmallPerturbationAttack`
  shifts one coordinate as much as possible while remaining selected by a
  *target* aggregator, exploiting the curse of dimensionality
  (El Mhamdi et al., ICML 2018).

Combining everything
--------------------

.. code-block:: python

   import torch
   from krum.primitives.aggregators.multikrum import MultiKrum
   from krum.primitives.attacks.sign_flip import SignFlipAttack

   n_workers, n_byzantine, dim = 10, 2, 100

   honest = torch.randn(n_workers - n_byzantine, dim)
   malicious = SignFlipAttack.generate(honest, f=n_byzantine, scale=1.5)

   all_grads = torch.cat([honest, malicious], dim=0)
   result = MultiKrum.aggregate(all_grads, n=n_workers, f=n_byzantine)

   print(f"Result norm: {result.norm():.4f}")

Next steps
---------

* :doc:`working_with_models`: how aggregators access flat gradient tensors
  and the standard models bundled with Krum.
* :doc:`centralised_simulation_walkthrough`: run aggregators and attacks
  inside a full training loop.
* :doc:`structured_experiments`: compare configurations systematically
  with ``Orchestrator`` and ``Metric``.
* :doc:`implement_aggregator`: write your own aggregation rule.
* :doc:`implement_attack`: write your own Byzantine attack.
* :doc:`/reference/primitives/aggregators/index`: full API reference.
* :doc:`/reference/primitives/attacks/index`: full API reference.
