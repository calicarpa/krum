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
- ``f`` is the number of Byzantine gradients to generate
- the returned tensor has shape :math:`(f, d)`

Available Attacks
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Gradient Attacks:

   classes/sign_flip
   classes/alie