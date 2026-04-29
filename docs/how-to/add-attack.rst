How to add a new attack
=======================

1. Create a new file in :doc:`/reference/attacks/index`.
2. Respect the reserved arguments: ``grad_honests``, ``f_decl``, ``f_real``,
   ``model``, ``defense``.
3. Return exactly ``f_real`` gradients.
4. Maintain separation between parameter validation (``check``) and actual
   execution.
