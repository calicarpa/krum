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

Quick Start
-----------

All aggregators can be called with the same keyword-only interface:

.. code-block:: python

    import aggregators

    # Using average (baseline)
    result = aggregators.average(gradients=list_of_gradients)

    # Using Krum with Byzantine tolerance
    result = aggregators.krum(gradients=list_of_gradients, f=2)

    # Checking validity before calling
    if aggregators.krum.check(gradients=list_of_gradients, f=2) is None:
        result = aggregators.krum(gradients=list_of_gradients, f=2)

.. seealso::

   To see how aggregators fit into a full training loop, see the
   :doc:`../experiments/index` quick start.
   For Byzantine attacks that test aggregation rules, see :doc:`../attacks/index`.

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
