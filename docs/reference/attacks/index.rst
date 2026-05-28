Attacks
=======

Krum provides gradient attacks that generate Byzantine gradients from honest
worker gradients. Attacks are stateless: each one implements the
:class:`attacks.Attack` contract as a classmethod, called directly on the class:

.. code-block:: python

   SomeAttack.generate(honest_gradients, f=...)

where:

- ``honest_gradients`` is a sequence of :math:`h` per-worker gradient vectors,
  each of shape :math:`(d,)` (a stacked :math:`(h, d)` tensor also works)
- ``f`` (:math:`b`) is the number of Byzantine gradients to generate
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
   * - Gaussian
      - Sends random Gaussian vectors with zero mean and configurable standard deviation
      - ``std``
      - :math:`b \times d`
   * - Omniscient
      - Sends negated full-dataset gradient scaled by a factor kappa
      - ``kappa``, full gradient via ``set_full_gradient()``
      - :math:`b \times d`

Available Attacks
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Gradient Attacks:

   classes/sign_flip
   classes/alie
   classes/gaussian
   classes/omniscient

API Reference
-------------

.. automodule:: krum.primitives.attacks
   :members:
   :undoc-members:
   :show-inheritance: