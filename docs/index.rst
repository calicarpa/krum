Krum, the Library
=================

A research framework for Byzantine-resilient distributed learning.

Krum provides a modular environment for comparing robust aggregation rules,
simulating Byzantine attacks, running experiment campaigns, and analyzing
results. The project is intentionally lightweight: clone the repository, add a
Python file to the right folder, and the new component is immediately
available by name.

Documentation
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Type
     - Purpose
   * - :doc:`Tutorials </tutorials/index>`
     - Step-by-step lessons for beginners.
   * - :doc:`How-to guides </how-to/index>`
     - Task-oriented recipes for common operations.
   * - :doc:`Reference </reference/aggregators/index>`
     - Technical descriptions of modules, classes, and functions.
   * - :doc:`Explanation </explanation/key-concepts>`
     - Background, design rationale, and conceptual discussion.


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
     - :doc:`reference/aggregators/index`, :doc:`reference/attacks/index`, :doc:`reference/experiments/index`
   * - **Infrastructure**
     - Contextual logging, argument parsing, tensor helpers, job management
     - :doc:`reference/tools/index`
   * - **Native Acceleration**
     - Auto-compiled C++/CUDA extensions for hot paths
     - :doc:`reference/native`


Learn more
----------

.. toctree::
   :caption: Home

   self

.. toctree::
   :maxdepth: 2

   tutorials/index

.. toctree::
   :maxdepth: 2

   how-to/index

.. toctree::
   :maxdepth: 2
   :caption: Explanation

   explanation/key-concepts
   explanation/byzantine-resilience
   explanation/debug-mode
   explanation/native-compilation
   explanation/tensor-lifecycle
   explanation/cli-format

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/architecture
   reference/aggregators/index
   reference/attacks/index
   reference/experiments/index
   reference/tools/index
   reference/native

.. toctree::
   :maxdepth: 1
   :caption: Project

   contributors


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
