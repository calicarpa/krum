Aggregators
===========

Krum provides several Byzantine-resilient gradient aggregation rules (GARs).
Each rule follows a keyword-only contract and includes parameter validation
and optional theoretical bounds.

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
   * - Krum
     - :math:`\mathcal{O}(n^2 d)`
     - :math:`2f + 3`
     - Moderate
   * - Bulyan
     - :math:`\mathcal{O}(n^2 d)`
     - :math:`4f + 3`
     - Strong
   * - Brute
     - :math:`\mathcal{O}(\binom{n}{n-f}d)`
     - :math:`2f + 1`
     - Optimal

where:

- :math:`n` = total number of workers
- :math:`f` = number of Byzantine workers
- :math:`d` = gradient dimension
- :math:`\binom{n}{k}` = binomial coefficient

Available Aggregators
---------------------

.. toctree::
   :maxdepth: 1
   :caption: Robust Aggregation Rules:

   classes/average
   classes/median
   classes/krum
   classes/bulyan
   classes/brute

API Reference
-------------

.. automodule:: aggregators
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
