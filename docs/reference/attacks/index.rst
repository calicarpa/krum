Attacks
=======

Krum provides several Byzantine attack strategies. Each attack follows a
keyword-only contract and includes parameter validation.

Overview
--------

.. list-table:: Attacks
   :header-rows: 1
   :widths: 20 42 20 18

   * - Attack
     - Description
     - Complexity
     - Target
   * - NaN
     - Generates gradients with NaN values
     - :math:`\mathcal{O}(d)`
     - Simple aggregators
   * - Identical (Bulyan)
     - Uses ones vector as attack direction
     - :math:`\mathcal{O}(nd)`
     - Multi-Krum based
   * - Identical (Empire)
     - Uses negative average as direction
     - :math:`\mathcal{O}(nd)`
     - Inner product based
   * - Identical (Little)
     - Uses std dev as attack direction
     - :math:`\mathcal{O}(nd)`
     - Distance-based

where:

- :math:`n` = total number of workers
- :math:`f` = number of Byzantine workers
- :math:`d` = gradient dimension

Available Attacks
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Byzantine Attack Strategies:

   classes/identical
   classes/nan

API Reference
-------------

.. automodule:: attacks
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
