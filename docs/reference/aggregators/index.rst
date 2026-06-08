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

Overview
--------

.. list-table:: Aggregators
   :header-rows: 1
   :widths: 16 42 24 18

   * - Aggregator
     - Complexity
     - Min. Workers
     - Byzantine Res.
   * - Average
     - :math:`\mathcal{O}(nd)`
     - :math:`1`
     - None (baseline)
   * - Median
     - :math:`\mathcal{O}(nd)`
     - :math:`1`
     - Basic
   * - Trimmed Mean
     - :math:`\mathcal{O}(nd \log n)`
     - :math:`2f + 1`
     - Basic
   * - Krum
     - :math:`\mathcal{O}(n^2 d)`
     - :math:`2f + 3`
     - Moderate
   * - MultiKrum
     - :math:`\mathcal{O}(n^2 d)`
     - :math:`2f + 3`
     - Moderate
   * - Bulyan
     - :math:`\mathcal{O}(n^2 d)`
     - :math:`4f + 3`
     - Strong

where:

- :math:`n` = total number of workers
- :math:`f` = number of Byzantine workers
- :math:`d` = gradient dimension

Available Aggregators
---------------------

.. toctree::
   :maxdepth: 1

   classes/average
   classes/median
   classes/trimmed_mean
   classes/krum
   classes/multikrum
   classes/bulyan

API Reference
-------------

.. automodule:: krum.primitives.aggregators
   :members:
   :undoc-members:
   :show-inheritance:
