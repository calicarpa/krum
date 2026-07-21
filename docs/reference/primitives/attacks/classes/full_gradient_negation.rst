Full Gradient Negation
======================

.. automodule:: krum.primitives.attacks.full_gradient_negation
   :members:
   :undoc-members:
   :show-inheritance:

.. seealso::

   For a simple noise-based attack, see :doc:`gaussian`.
   For a sign-reversal attack, see :doc:`sign_flip`.

.. note::

   The full-dataset gradient is passed as a keyword argument to
   :meth:`~attacks.full_gradient_negation.FullGradientNegationAttack.generate`:

   .. code-block:: python

      FullGradientNegationAttack.generate(honest_gradients, f=..., full_gradient=full_grad, kappa=100.0)
