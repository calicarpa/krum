Omniscient
==========

.. automodule:: attacks.omniscient
   :members:
   :undoc-members:
   :show-inheritance:

.. seealso::

   For a simple noise-based attack, see :doc:`gaussian`.
   For a sign-reversal attack, see :doc:`sign_flip`.

.. note::

   The full-dataset gradient is passed as a keyword argument to
   :meth:`~attacks.omniscient.OmniscientAttack.generate`:

   .. code-block:: python

      OmniscientAttack.generate(honest_gradients, f=..., full_gradient=full_grad, kappa=100.0)
