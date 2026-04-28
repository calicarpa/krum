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

Quick Start
-----------

.. code-block:: python

    from experiments import (
        Configuration, Model, Dataset,
        Loss, Criterion, Optimizer, Checkpoint,
        make_datasets,
    )

    # Configuration
    config = Configuration(device="cuda:0", dtype=torch.float32)

    # Model
    model = Model("resnet18", config, num_classes=10)

    # Datasets
    trainset, testset = make_datasets("cifar10", train_batch=128)
    model.default("trainset", trainset)
    model.default("testset", testset)

    # Loss and optimizer
    loss = Loss("crossentropy") + 0.01 * Loss("l2")
    optim = Optimizer("adam", model, lr=0.001)
    model.default("loss", loss)
    model.default("optimizer", optim)

    # Training step
    gradient, loss_val = model.backprop(outloss=True)
    model.update(gradient)

    # Evaluation
    accuracy = model.eval(criterion=Criterion("top-k", k=1))

    # Checkpoint
    ckpt = Checkpoint()
    ckpt.snapshot(model).snapshot(optimizer).save("run.pt")

.. seealso::

   For robust gradient aggregation, see :doc:`../aggregators/index`.
   For Byzantine attacks that generate malicious gradients, see
   :doc:`../attacks/index`.
   For utility functions (logging, parsing, timing), see :doc:`../tools/index`.

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
