How to add a new dataset
========================

This guide walks you through adding a new dataset loader to the experiments
module.

Step 1 — Create the module
--------------------------

Create a new Python file in ``experiments/datasets/``. The file is
auto-discovered at import time — no ``__init__.py`` edit is needed.

.. code-block:: text

   experiments/datasets/
   ├── svm.py
   └── my_dataset.py      ← your new file

Step 2 — Implement the loader
-----------------------------

Write a function that returns an infinite-batch generator. Use
:func:`experiments.batch_dataset` to handle train/test splitting and
batching automatically.

.. code-block:: python

   # experiments/datasets/my_dataset.py

   import torch
   import experiments
   from experiments import Dataset

   _default_root = Dataset.get_default_root()
   _cache = None


   def _load(root, url):
       """Download and cache the raw tensors."""
       global _cache
       if _cache is not None:
           return _cache

       cache_file = root / "my_dataset.pt"
       if cache_file.exists():
           with cache_file.open("rb") as fd:
               _cache = torch.load(fd)
               return _cache

       # Download and parse (adapt to your format)
       import requests
       response = requests.get(url)
       # ... parse response into inputs, labels ...
       inputs = torch.randn(1000, 32)   # placeholder
       labels = torch.randint(0, 2, (1000, 1))

       with cache_file.open("wb") as fd:
           torch.save((inputs, labels), fd)

       _cache = (inputs, labels)
       return _cache


   def my_dataset(train=True, batch_size=None, root=None, download=False,
                  *args, **kwargs):
       inputs, labels = _load(
           root or _default_root,
           "https://example.com/data.csv",
       )
       return experiments.batch_dataset(
           inputs, labels, train, batch_size, split=800,
       )

Step 3 — Export in ``__all__``
-----------------------------

List the builder function in ``__all__`` so the loader can discover it.

.. code-block:: python

   __all__ = ["my_dataset"]

Step 4 — Use it
---------------

The dataset is registered under the compound name
``<filename>-<function>``. For a file named ``my_dataset.py`` exposing
``my_dataset``, the identifier is ``my_dataset-my_dataset``:

.. code-block:: python

   from experiments import Dataset

   dataset = Dataset("my_dataset-my_dataset", train=True, batch_size=32)

.. code-block:: bash

   python -m experiments --dataset my_dataset-my_dataset ...

Builder function parameters
----------------------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``train``
     - ``bool``
     - ``True`` for training set, ``False`` for test set
   * - ``batch_size``
     - ``int | None``
     - Batch size (``None`` for full-batch)
   * - ``root``
     - ``Path | None``
     - Cache directory (default: ``~/.krum/datasets/``)
   * - ``download``
     - ``bool``
     - Whether to download if not cached
   * - ``*args, **kwargs``
     -
     - Forward-compatibility

Naming convention
-----------------

.. list-table::
   :header-rows: 1

   * - File
     - ``__all__``
     - Registered name
   * - ``svm.py``
     - ``["phishing"]``
     - ``svm-phishing``
   * - ``my_dataset.py``
     - ``["my_dataset"]``
     - ``my_dataset-my_dataset``

.. seealso::

   For symlink-based registration of external datasets, see
   :doc:`add-custom-dataset`. For the :class:`experiments.Dataset` class, see
   :doc:`/reference/experiments/classes/dataset`.
