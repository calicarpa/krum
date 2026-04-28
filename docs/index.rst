Krum, the Library
=================

A research framework for Byzantine-resilient distributed learning.

Krum provides a modular environment for comparing robust aggregation rules,
simulating Byzantine attacks, running experiment campaigns, and analyzing
results. The project is intentionally lightweight: clone the repository, add a
Python file to the right folder, and the new component is immediately
available by name.

.. note::

   This documentation is organized by module. If you are extending the
   codebase, start with :doc:`architecture`. If you want to understand the
   registration system or contracts, read :ref:`key-concepts` below.


The Five-Step Workflow
----------------------

A typical Krum experiment follows a simple pipeline:

1. **Build** a model, dataset, loss, criterion, and optimizer.
2. **Train** honestly — each step produces gradients from good workers.
3. **Attack** — Byzantine workers submit malicious gradients.
4. **Aggregate** — a robust rule combines honest and Byzantine gradients.
5. **Measure** — results are logged, saved, and compared.


Three Layers
------------

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Layer
     - Purpose
     - Entry Point
   * - **Research Components**
     - Robust aggregation rules, Byzantine attacks, model/dataset wrappers
     - :doc:`aggregators/index`, :doc:`attacks/index`, :doc:`experiments/index`
   * - **Infrastructure**
     - Contextual logging, argument parsing, tensor helpers, job management
     - :doc:`tools/index`
   * - **Native Acceleration**
     - Auto-compiled C++/CUDA extensions for hot paths
     - :doc:`native`


.. _key-concepts:

Key Concepts
------------

Registration System
~~~~~~~~~~~~~~~~~~~

Krum uses a lightweight registration pattern. Each family (aggregators,
attacks, models, datasets) has its own registry:

- Components are discovered by name from CLI arguments.
- Extension without centralization: drop a file in the right folder.
- Local validation: every component carries its own ``check()``.
- Python / native fallback: a rule can expose the same public name while
  switching from Python to native under the hood.

Keyword-Only Contracts
~~~~~~~~~~~~~~~~~~~~~~

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


Learn more
----------

.. toctree::
   :maxdepth: 2
   :caption: Modules

   architecture
   aggregators/index
   attacks/index
   experiments/index
   tools/index
   native


License
-------

Krum is open-sourced under the MIT License. See the `License file on GitHub
<https://github.com/calicarpa/krum/blob/main/LICENSE>`__ for details. Source
code and issue tracker live at `<https://github.com/calicarpa/krum>`__. We
welcome contributions and feedback from the community.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
