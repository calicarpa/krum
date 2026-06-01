Attacks
=======

Krum provides gradient attacks that generate Byzantine gradients from honest
worker gradients. Every attack implements the :class:`attacks.Attack` contract:

.. code-block:: python

   attack(honest_gradients, num_byzantine)

where:

- ``honest_gradients`` is a sequence of :math:`h` per-worker gradient vectors,
  each of shape :math:`(d,)` (a stacked :math:`(h, d)` tensor also works)
- ``num_byzantine`` (:math:`b`) is the number of Byzantine gradients to generate
- the returned tensor has shape :math:`(b, d)`

Overview
--------

.. list-table:: Attacks
   :header-rows: 1
   :widths: 18 44 20 18

   * - Attack
     - Description
     - Parameters
     - Output
   * - SignFlip
     - Sends scaled gradients in the opposite direction of the honest mean
     - ``scale``
     - :math:`b \times d`
   * - ALIE
     - Sends mean-shifted gradients using exact honest coordinate-wise statistics
     - ``z``, ``direction`` (:class:`attacks.alie.Direction`)
     - :math:`b \times d`

Available Attacks
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Gradient Attacks:

   classes/sign_flip
   classes/alie

API Reference
-------------

.. automodule:: attacks
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
