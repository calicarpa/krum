Using a custom dataset
======================

Krum simulations accept any PyTorch ``Dataset`` object, so bringing your
own data requires no special wrapper. This tutorial shows three common
patterns and how to connect them to a simulation.

Requirements
------------

A dataset for ``CentralisedSimulation`` must satisfy a minimal contract:

* It is a ``torch.utils.data.Dataset`` — implements ``__len__`` and
  ``__getitem__``.
* ``__getitem__(i)`` returns a ``(x, y)`` tuple where ``x`` is a model
  input tensor and ``y`` is compatible with the loss function (default
  ``cross_entropy`` expects integer class indices).
* ``__len__`` returns the total number of samples (used to compute per-
  worker shard sizes).

The simplest implementation that meets these requirements is
``TensorDataset``.

Pattern 1: torchvision datasets
--------------------------------

For standard vision benchmarks, ``torchvision.datasets`` works directly:

.. code-block:: python

   from torchvision import datasets, transforms

   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,)),
   ])
   train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
   test_set = datasets.MNIST("data", train=False, download=True, transform=transform)

The ``transform`` parameter converts PIL images to tensors and normalises
them. Without it, the model would receive a PIL image instead of a tensor.

Pattern 2: numpy arrays with TensorDataset
-------------------------------------------

If your data is already loaded in memory as numpy arrays, wrap it with
``TensorDataset`` — this is what the Spambase experiment does:

.. code-block:: python

   import numpy as np
   import torch
   from torch.utils.data import TensorDataset

   # x: (num_samples, num_features), y: (num_samples,)
   x = np.load("features.npy")
   y = np.load("labels.npy")

   x_t = torch.from_numpy(x).float()
   y_t = torch.from_numpy(y).long()

   dataset = TensorDataset(x_t, y_t)

``TensorDataset`` is just a thin wrapper around two tensors. It pairs
each feature row with its label in ``__getitem__``.

Pattern 3: loading from disk
-----------------------------

For data that does not fit in memory, subclass ``Dataset`` and load
samples on demand:

.. code-block:: python

   import os
   import torch
   from torch.utils.data import Dataset

   class ImageFolderDataset(Dataset):
       def __init__(self, root: str):
           self.paths = sorted(
               os.path.join(root, f) for f in os.listdir(root)
           )

       def __len__(self):
           return len(self.paths)

       def __getitem__(self, idx):
           # Load and return (x, y) — implement your logic
           data = torch.load(self.paths[idx])
           return data["x"], data["y"]

The simulation creates per-worker ``DataLoader`` instances from this
dataset, so the on-disk format only needs to support random access by
index.

Matching the model
------------------

The model's input and output dimensions must match the dataset:

.. code-block:: python

   class MyModel(nn.Sequential):
       def __init__(self):
           super().__init__(
               nn.Flatten(),
               nn.Linear(64, 32), nn.ReLU(),
               nn.Linear(32, 10),
           )

Check that the first ``Linear`` input equals the feature dimension of
``x`` and the last ``Linear`` output equals the number of classes in
``y``.

Controlling dataset size
-------------------------

For quick prototyping, truncate the dataset with ``Subset``:

.. code-block:: python

   from torch.utils.data import Subset

   small_train = Subset(train_set, range(5000))
   small_test = Subset(test_set, range(1000))

This is useful during development to reduce experiment time.

Putting it all together
------------------------

.. code-block:: python

   import torch
   from torch.utils.data import TensorDataset
   from krum.simulations.centralised import KrumSimulation
   from krum.aggregators import MultiKrum
   from krum.attacks import SignFlipAttack

   # 1. Custom dataset from numpy
   x = torch.randn(10000, 64)
   y = torch.randint(0, 10, (10000,))
   train_set = TensorDataset(x, y)
   test_set = TensorDataset(x[:2000], y[:2000])

   # 2. Matching model
   class MyModel(torch.nn.Sequential):
       def __init__(self):
           super().__init__(
               torch.nn.Flatten(),
               torch.nn.Linear(64, 32),
               torch.nn.ReLU(),
               torch.nn.Linear(32, 10),
           )

   # 3. Simulation
   sim = KrumSimulation(
       model_cls=MyModel,
       train_set=train_set,
       test_set=test_set,
       aggregator=MultiKrum,
       attack=SignFlipAttack,
       n=10, f=2,
       rounds=50,
       batch_size=32,
       lr=0.01,
   )
   sim.setup()

   for step in range(50):
       sim.step()
       if step % 10 == 0 or step == 49:
           loss, acc = sim.evaluate()
           print(f"step {step}: loss={loss:.3f} acc={acc:.3f}")

Data type and device
---------------------

Make sure the dataset tensor types match the model:

* **Features** should be ``float32`` (``torch.float32``) — the default
  for ``torch.from_numpy(…).float()``.
* **Labels** should be ``int64`` (``torch.long``) — the default for
  ``torch.from_numpy(…).long()`` and the expected input to
  ``cross_entropy``.

The simulation moves the model to the configured ``device``
auto‑matically; the dataset stays on CPU and is moved per‑batch by
the ``DataLoader``.

Next steps
----------

* :doc:`centralised_simulation_walkthrough` — a complete training
  loop with dataset setup, evaluation, and baseline comparison.
* :doc:`using_aggregators_attacks` — built-in aggregation rules
  and attack strategies.
* :doc:`working_with_models` — the ``Model`` wrapper and how to
  build models for Krum.
