Experiments
===========

The ``experiments`` module provides dataset and model wrappers that form the
foundation of a Krum training loop. It handles model construction, dataset
loading, loss composition, optimization, checkpointing, and device/dtype
configuration.

Components
----------

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Component
     - Role
     - Key Class
   * - **Configuration**
     - Device, dtype, and memory-transfer settings
     - :class:`experiments.Configuration`
   * - **Model**
     - Model wrapper with parameter flattening and gradient handling
     - :class:`experiments.Model`
   * - **Dataset**
     - Infinite-batch wrapper around torchvision and custom datasets
     - :class:`experiments.Dataset`
   * - **Loss**
     - Derivable loss with regularization and composition support
     - :class:`experiments.Loss`
   * - **Criterion**
     - Non-derivable evaluation metric (top-k, sigmoid)
     - :class:`experiments.Criterion`
   * - **Optimizer**
     - Optimizer wrapper with learning-rate control
     - :class:`experiments.Optimizer`
   * - **Checkpoint**
     - Save/restore state dictionaries for models and optimizers
     - :class:`experiments.Checkpoint`

Core Modules
------------

.. toctree::
   :maxdepth: 1
   :caption: Core Modules:

   classes/configuration
   classes/dataset
   classes/model
   classes/loss
   classes/optimizer
   classes/checkpoint

Custom Models & Datasets
------------------------

.. toctree::
   :maxdepth: 1
   :caption: Custom Models & Datasets:

   classes/models
   classes/datasets

API Reference
-------------

.. automodule:: experiments
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
