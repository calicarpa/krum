Aggregators
===========

Krum provides several Byzantine-resilient gradient aggregation rules (GARs).
Aggregators are **stateless** — you call them as classmethods without
instantiating any object. Specialized parameters (``f``, ``n``, ``m``) are
keyword-only:

.. code-block:: python

    from krum.primitives.aggregators import Average, Krum, TrimmedMean

    result = Average.aggregate(gradients)
    result = Krum.aggregate(gradients, n=5, f=1)
    result = TrimmedMean.aggregate(gradients, f=2)

All aggregators validate their parameters and raise ``ValueError`` on misuse.

Available Aggregators
---------------------

.. toctree::
   :maxdepth: 1
   :caption: Robust Aggregation Rules:

   classes/average
   classes/median
   classes/trimmed_mean
   classes/krum
   classes/multikrum
   classes/bulyan
