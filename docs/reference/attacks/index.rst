Attacks
=======

Krum provides gradient attacks that generate Byzantine gradients from honest
worker gradients. Each attack follows a shared contract:

.. code-block:: python

   attack(honest_gradients, num_byzantine)

where:

- :math:`h` = number of honest workers
- :math:`b` = number of Byzantine gradients to generate
- :math:`d` = gradient dimension

Overview
--------

.. list-table:: Attacks
   :header-rows: 1
   :widths: 18 44 20 18

   * - Attack
     - Description
     - Parameters
     - Output
   * - NaN
     - Sends gradients filled with NaN values
     - None
     - :math:`b \times d`
   * - Inf
     - Sends gradients filled with positive or negative infinity
     - ``sign``
     - :math:`b \times d`
   * - ALIE
     - Sends mean-shifted gradients using exact honest coordinate-wise statistics
     - ``z``, ``direction``
     - :math:`b \times d`

Available Attacks
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Gradient Attacks:

   classes/nan
   classes/inf
   classes/alie

API Reference
-------------

.. automodule:: attacks
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
