Key Concepts
============

Registration System
-------------------

Krum uses a lightweight registration pattern. Each family (aggregators,
attacks, models, datasets) has its own registry:

- Components are discovered by name from CLI arguments.
- Extension without centralization: drop a file in the right folder.
- Local validation: every component carries its own ``check()``.
- Python / native fallback: a rule can expose the same public name while
  switching from Python to native under the hood.

Keyword-Only Contracts
----------------------

All aggregators and attacks accept **keyword-only** arguments.

**Aggregator Contract**

- ``gradients`` — non-empty list of gradients to aggregate.
- ``f`` — number of Byzantine gradients the rule must tolerate.
- ``model`` — model instance with configured defaults.

**Attack Contract**

- ``grad_honests`` — non-empty list of honest gradients.
- ``f_decl`` — number of declared Byzantine gradients.
- ``f_real`` — number of actual Byzantine gradients to generate.
- ``model`` — model used for the attack.
- ``defense`` — aggregation rule to defeat.

.. important::

   Returned tensors must **never alias** any input tensor. This guarantee is
   central to the correctness of the framework.
