Overview
========

Krum is designed for researchers who want to compare robust aggregation rules, simulate Byzantine attacks, run experiment campaigns, and analyze results.

The project is intentionally modular. Most components are loaded automatically from their directories, allowing a researcher to clone the repository, add a Python file, and immediately make the new component available by name.

The Four Layers
---------------

The repository organizes into three layers:

- **Research Components**: :doc:`aggregators` contains robust aggregation rules, :doc:`attacks` contains Byzantine attacks, :doc:`experiments` contains wrappers around PyTorch and TorchVision
- **Infrastructure**: :doc:`tools` provides contextual logging, argument conversion, tensor helpers, job management, and general utilities
- **Native Acceleration**: :doc:`native` automatically compiles C++/CUDA modules when available

Key Concepts
------------

Registration System
~~~~~~~~~~~~~~~~~~~

Krum uses a registration pattern for extensibility. Each family (aggregators, attacks, models, datasets) has its own registry:

- Components are discovered by name from CLI arguments
- Extension without centralization: add a file to the appropriate folder
- Local validation: each component carries its own ``check()``
- Python/native fallback: a rule can keep the same public name while switching from Python to native implementation

Keyword-Only Contracts
~~~~~~~~~~~~~~~~~~~~~~~

All aggregators and attacks use keyword-only arguments:

**Aggregator Contract:**
- ``gradients``: non-empty list of gradients to aggregate
- ``f``: number of Byzantine gradients to support
- ``model``: model with configured defaults

**Attack Contract:**
- ``grad_honests``: non-empty list of honest gradients
- ``f_decl``: number of declared Byzantine gradients
- ``f_real``: number of actual Byzantine gradients to generate
- ``model``: model used for the attack
- ``defense``: aggregation rule to defeat

Important: Returned tensors must never alias input tensors.