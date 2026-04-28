Welcome to Krum's documentation, a research framework for Byzantine-resilient distributed learning. 

What is Krum, the Library?
==========================

Krum, the Library, is a research framework for Byzantine-resilient distributed learning. It provides a modular environment for comparing robust aggregation rules, simulating Byzantine attacks, running experiment campaigns, and analyzing results.

The core idea is simple:

1. Build a model, dataset, loss, criterion, and optimizer
2. Each training step produces honest gradients
3. An attack generates Byzantine gradients
4. An aggregation rule combines all gradients
5. Results are measured and saved

The project is intentionally modular. Most components are loaded automatically from their directories, allowing researchers to clone the repository, add a Python file, and immediately make the new component available by name.

Three Layers
------------

- **Research Components**: :doc:`aggregators/index` contains robust aggregation rules, :doc:`attacks/index` contains Byzantine attacks, :doc:`experiments/index` contains wrappers around PyTorch and TorchVision
- **Infrastructure**: :doc:`tools/index` provides contextual logging, argument conversion, tensor helpers, job management, and general utilities
- **Native Acceleration**: :doc:`native` automatically compiles C++/CUDA modules when available

Key Concepts
------------

**Registration System**

Krum uses a registration pattern for extensibility. Each family (aggregators, attacks, models, datasets) has its own registry:

- Components are discovered by name from CLI arguments
- Extension without centralization: add a file to the appropriate folder
- Local validation: each component carries its own ``check()``
- Python/native fallback: a rule can keep the same public name while switching from Python to native implementation

**Keyword-Only Contracts**

All aggregators and attacks use keyword-only arguments:

*Aggregator Contract:*

- ``gradients``: non-empty list of gradients to aggregate
- ``f``: number of Byzantine gradients to support
- ``model``: model with configured defaults

*Attack Contract:*

- ``grad_honests``: non-empty list of honest gradients
- ``f_decl``: number of declared Byzantine gradients
- ``f_real``: number of actual Byzantine gradients to generate
- ``model``: model used for the attack
- ``defense``: aggregation rule to defeat

.. note::
   Returned tensors must never alias input tensors.

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

Krum is open-sourced and distributed under the MIT License. See `License <https://github.com/calicarpa/krum/blob/main/LICENSE>`_ for details.
Our code is available on GitHub at `<https://github.com/calicarpa/krum>`_. We welcome contributions and feedback from the community.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
