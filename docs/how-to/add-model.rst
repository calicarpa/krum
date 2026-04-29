How to add a new model
======================

This guide walks you through adding a new model architecture to the
experiments module.

Step 1 — Create the module
--------------------------

Create a new Python file in ``experiments/models/``. The file is
auto-discovered at import time — no ``__init__.py`` edit is needed.

.. code-block:: text

   experiments/models/
   ├── simples.py
   └── my_model.py      ← your new file

Step 2 — Define the ``nn.Module``
---------------------------------

Define a private class that extends ``torch.nn.Module``. The model must be
flatten-able — all parameters must be reachable via ``model.parameters()``
so that :class:`experiments.Model` can serialize and restore gradients.

.. code-block:: python

   # experiments/models/my_model.py

   import torch


   class _ResNet(torch.nn.Module):
       def __init__(self, num_classes=10):
           super().__init__()
           self._conv = torch.nn.Conv2d(3, 16, 3, padding=1)
           self._fc = torch.nn.Linear(16 * 32 * 32, num_classes)

       def forward(self, x):
           x = torch.relu(self._conv(x))
           x = x.view(x.size(0), -1)
           return torch.log_softmax(self._fc(x), dim=1)

Step 3 — Expose a constructor in ``__all__``
--------------------------------------------

Define a plain function that instantiates the module and list it in
``__all__``. The loader uses ``__all__`` to discover constructors.

.. code-block:: python

   def resnet(*args, **kwargs):
       return _ResNet(*args, **kwargs)


   __all__ = ["resnet"]

Step 4 — Use it
---------------

The model is registered under the compound name
``<filename>-<constructor>``. For a file named ``my_model.py`` exposing
``resnet``, the identifier is ``my_model-resnet``:

.. code-block:: python

   from experiments import Model, Configuration

   config = Configuration(device="cpu")
   model = Model("my_model-resnet", config, num_classes=10)

.. code-block:: bash

   python -m experiments --model my_model-resnet ...

Naming convention
-----------------

.. list-table::
   :header-rows: 1

   * - File
     - ``__all__``
     - Registered name
   * - ``simples.py``
     - ``["full", "conv"]``
     - ``simples-full``, ``simples-conv``
   * - ``my_model.py``
     - ``["resnet"]``
     - ``my_model-resnet``

.. seealso::

   For symlink-based registration of external models, see
   :doc:`add-custom-model`. For the :class:`experiments.Model` class, see
   :doc:`/reference/experiments/classes/model`.
