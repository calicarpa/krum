Data partitioners
=================

Dataset-to-worker partitioning strategies for simulations.

A :class:`~krum.primitives.data_partitioners.DataPartitioner` turns one
dataset into ``n`` per-worker :class:`~torch.utils.data.Dataset`
instances. Both
:class:`~krum.simulations.centralised.CentralisedSimulation` and
:class:`~krum.simulations.decentralised.DecentralisedSimulation` consume
this same shape, so partitioning is entirely the caller's responsibility,
IID or not. Wrapping each worker's dataset into a
:class:`~torch.utils.data.DataLoader` (batch size, shuffling) is the
simulation's job, not the partitioner's — that separation is what lets
partitioners compose (e.g. mixing two partitioners' outputs) without
reaching back into a ``DataLoader`` to get at the underlying samples.

Like :mod:`~krum.primitives.aggregators` and :mod:`~krum.primitives.attacks`,
each strategy is **stateless**: a ``@classmethod`` invoked directly on the
class. The dataset is the sole positional argument; ``n``, ``seed``, and any
partitioner-specific hyperparameters are keyword-only.

Available partitioners
----------------------

.. toctree::
   :maxdepth: 1

   classes/iid
   classes/dirichlet
   classes/per_labels
   classes/mixing

Base class
----------

.. autoclass:: krum.primitives.data_partitioners.DataPartitioner
   :members:
   :undoc-members:
   :show-inheritance: