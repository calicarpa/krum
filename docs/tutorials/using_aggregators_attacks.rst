Using Aggregators and Attacks
=============================

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

   # Simple average (baseline, no resilience)
   result = Average.aggregate(gradients)

   # Coordinate-wise median (basic resilience)
   result = Median.aggregate(gradients)

   # Trimmed mean (basic resilience, requires 2f+1 workers)
   result = TrimmedMean.aggregate(gradients, f=2)

   # Krum (moderate resilience, requires 2f+3 workers)
   result = Krum.aggregate(gradients, n=10, f=2)

   # Multi-Krum (moderate resilience, averages m = n - f - 2 gradients)
   result = MultiKrum.aggregate(gradients, n=10, f=2)

   # Bulyan (strong resilience, two-stage, requires 4f+3 workers)
   result = Bulyan.aggregate(gradients, n=15, f=2)

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
   * - ``Average``
     - None (baseline)
     - ``n >= 1``
   * - ``Median``
     - ``f < n / 2``
     - ``n >= 2``
   * - ``TrimmedMean``
     - ``f < n / 2``
     - ``n >= 2f + 1``
   * - ``Krum``
     - ``2f + 2 < n``
     - ``n >= 2f + 3``
   * - ``MultiKrum``
     - ``2f + 2 < n``
     - ``n >= 2f + 3``
   * - ``Bulyan``
     - ``4f + 2 < n``
     - ``n >= 4f + 3``
   * - ``Brute``
     - Brute-force Krum
     - ``n >= 4f + 3``
   * - ``GeoMed``
     - ``f < n / 2``
     - ``n >= 2``

Attacks
-------

Attacks generate Byzantine gradients from honest worker gradients:

.. code-block:: python

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

   # Full gradient negation attack
   byzantine = FullGradientNegationAttack.generate(honest_gradients, f=2)

   # Small perturbation attack (exploits curse of dimensionality)
   byzantine = SmallPerturbationAttack.generate(honest_gradients, f=2, n=10)

All attacks follow the same pattern: pass the honest gradients and the number
of Byzantine workers ``f``, and they return a tensor of shape ``(f, d)``.

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
----------

* :doc:`using_simulations` to run this inside a full training loop
* :doc:`/reference/primitives/aggregators/index` for the full API reference
* :doc:`/reference/primitives/attacks/index` for the full API reference
