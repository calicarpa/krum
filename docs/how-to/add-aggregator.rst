How to add a new aggregator
===========================

1. Create a new file in :doc:`/reference/aggregators/index`.
2. Expose an aggregation function and a ``check`` function.
3. Respect the keyword-only contract (``gradients``, ``f``, ``model``).
4. **Never** return a tensor that aliases an input tensor.
5. Optionally expose ``upper_bound`` and/or ``influence`` metadata.

.. seealso::

   For the full contract, see :doc:`/explanation/key-concepts`.
