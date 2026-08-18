Working with Data Partitioners
==============================

**Problem:** Simulations need one dataset per worker, but real-world
federated data is rarely IID — clients hold skewed, partial views of the
full dataset. How do you split one dataset into ``n`` per-worker datasets,
from perfectly IID to pathologically skewed?

This tutorial covers the ``DataPartitioner`` family: the base API plus
four built-in partitioning strategies, and how to plug them into a
simulation.

The DataPartitioner API
-----------------------

All partitioners are **stateless**: each strategy is a ``@classmethod``
invoked directly on the class. The dataset is the sole positional
argument; ``n``, ``seed``, and any partitioner-specific hyperparameters
are keyword-only:

.. code-block:: python

   from torchvision import datasets, transforms

   from krum.primitives.data_partitioners.iid import IidPartitioner

   dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
   train_datasets = IidPartitioner.partition(dataset, n=10)

``partition`` returns one :class:`~torch.utils.data.Dataset` per worker.
Wrapping those into :class:`~torch.utils.data.DataLoader` instances (batch
size, shuffling) is the simulation's job, not the partitioner's — that
separation is what lets partitioners compose without reaching back into a
``DataLoader`` to get at the underlying samples.

Every partitioner is deterministic: the same ``seed`` produces the same
split. Worker ``w``'s mini-batch sampling is seeded with ``seed + w``, so
the whole training run is reproducible end to end.

IidPartitioner: equal-size shards
---------------------------------

The IID baseline of McMahan et al. (AISTATS 2017): the dataset is shuffled
and cut into ``n`` equal-size, disjoint, uniformly random shards. Any
remainder (``len(dataset) % n`` samples) is dropped:

.. code-block:: python

   from krum.primitives.data_partitioners.iid import IidPartitioner

   train_datasets = IidPartitioner.partition(dataset, n=10, seed=7)

DirichletPartitioner: per-class label skew
------------------------------------------

For each class ``k``, draws a proportion vector
``p_k ~ Dirichlet(alpha, ..., alpha)`` over the ``n`` workers, then gives
worker ``w`` a ``p_k,w`` fraction of class ``k``'s samples. ``alpha``
controls the skew: large ``alpha`` collapses every ``p_k`` to
``(1/n, ..., 1/n)`` (near-IID); small ``alpha`` collapses each class to a
single worker (extreme imbalance). Every sample is assigned exactly once —
no remainder is dropped:

.. code-block:: python

   from krum.primitives.data_partitioners.dirichlet import DirichletPartitioner

   near_iid = DirichletPartitioner.partition(dataset, n=10, alpha=100.0)
   skewed = DirichletPartitioner.partition(dataset, n=10, alpha=0.1)

A worker can legitimately end up with zero samples of a class, or even
zero samples overall; it gets an empty (but valid) dataset.

PerLabelsPartitioner: from pathological skew to IID
---------------------------------------------------

Sorts the dataset by label, cuts it into ``n_shards`` equal-size
contiguous shards, shuffles the shard order, then deals shards to workers
round-robin. ``n_shards`` interpolates geometrically with ``lambda_``:

.. math::

   \text{n_shards} = n \cdot \left(\frac{N}{n}\right)^{\lambda}

At ``lambda_ = 0``, ``n_shards = n`` — one near-single-label shard per
worker, the most pathological split this mechanism can produce. At
``lambda_ = 1``, ``n_shards = N`` — every shard is a single sample, and
shuffle-then-round-robin reduces to exactly what ``IidPartitioner`` does,
recovering IID as a special case of the same mechanism:

.. code-block:: python

   from krum.primitives.data_partitioners.per_labels import PerLabelsPartitioner

   pathological = PerLabelsPartitioner.partition(dataset, n=10, lambda_=0.0)
   halfway = PerLabelsPartitioner.partition(dataset, n=10, lambda_=0.5)

Because shards are dealt round-robin, any ``n_shards % n`` remainder
shards are spread one at a time across the first few workers instead of
being dropped, so worker dataset sizes never differ by more than one
shard.

MixingPartitioner: interpolate between any two partitioners
-----------------------------------------------------------

Shuffles the dataset, splits it into a ``(1 - gamma)`` fraction and a
``gamma`` fraction, partitions each fraction independently with ``p1`` and
``p2`` (both across the same ``n`` workers), then gives worker ``w`` the
concatenation of its two slices:

.. code-block:: python

   from krum.primitives.data_partitioners.dirichlet import DirichletPartitioner
   from krum.primitives.data_partitioners.iid import IidPartitioner
   from krum.primitives.data_partitioners.mixing import MixingPartitioner

   train_datasets = MixingPartitioner.partition(
       dataset,
       n=10,
       p1=IidPartitioner,
       p2=DirichletPartitioner,
       gamma=0.3,
       p2_kwargs={"alpha": 0.5},
   )

``gamma = 0`` recovers ``p1`` alone; ``gamma = 1`` recovers ``p2`` alone.
This generalizes the "gamma-similarity" scheme of Karimireddy et al.
(ICML 2020, SCAFFOLD, Section 7.1) — there ``p1`` is always an IID split
and ``p2`` always a sort-by-label split — to any pair of partitioners.
Whether a sample can be dropped depends on ``p1``/``p2`` themselves: an
``IidPartitioner`` drops a remainder within its own slice, a
``DirichletPartitioner`` never does.

Using partitioners in a simulation
----------------------------------

Simulations consume the ``Sequence[Dataset]`` shape directly. For
example, :class:`~krum.simulations.decentralised.DecentralisedSimulation`
takes the per-worker datasets as ``train_datasets`` and wraps each honest
worker's dataset into a ``DataLoader`` itself. Putting the pieces
together: load MNIST, split the train set with a Dirichlet skew, then run
a MoNNA simulation on the resulting per-worker datasets.

.. code-block:: python

   import torch
   import torch.nn as nn
   from torchvision import datasets, transforms

   from krum.primitives.attacks.sign_flip import SignFlipAttack
   from krum.primitives.data_partitioners.dirichlet import DirichletPartitioner
   from krum.primitives.models import Model
   from krum.primitives.models.mlp import Monna2023SmallMnist
   from krum.simulations.decentralised.monna_icml_2023 import MonnaSimulation

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   n, f, seed = 10, 2, 42

   # 1. Load the data: one shared train set and one test set
   transform = transforms.ToTensor()
   train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
   test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

   # 2. Partition the train set across the n workers (Dirichlet label skew)
   train_datasets = DirichletPartitioner.partition(train_set, n=n, alpha=0.5, seed=seed)

   # 3. Build the simulation from the per-worker datasets
   #    (the f Byzantine workers apply the sign-flip attack)
   model = Model(Monna2023SmallMnist().to(device))

   sim = MonnaSimulation(
       model=model,
       train_datasets=train_datasets,   # Sequence[Dataset], one per worker
       train_batch_size=32,
       test_set=test_set,
       test_batch_size=64,
       loss_fn=nn.CrossEntropyLoss(),
       n=n,
       f=f,
       attack=SignFlipAttack,
       learning_rate=0.1,
       seed=seed,
   )

   # 4. Train for 50 rounds
   results = sim.run(50)

Any of the four strategies from this tutorial can be dropped in at step 2:
``train_datasets`` is always a ``Sequence[Dataset]``, one per worker, no
matter which partitioner produced it.

The ``Sequence[Dataset]`` shape is the contract of the decentralised
simulations; the partitioning choice is entirely the caller's
responsibility — pick any strategy above and pass its output straight
in. (For the parameter-server setup,
:class:`~krum.simulations.centralised.CentralisedSimulation` takes a
single ``train_set`` instead — the dataset is shared and each worker's
minibatches are drawn from it internally.)

.. seealso::

   :doc:`/reference/primitives/data_partitioners/index` for the full API
   reference.