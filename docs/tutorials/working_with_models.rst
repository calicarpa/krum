Working with models
===================

**Problem:** Aggregators need flat 1-D gradient tensors, but copying
parameters every round is slow and wasteful. How do you efficiently
access and manipulate model parameters and gradients as flat tensors?

This tutorial covers the ``Model`` wrapper (zero-copy flat tensor views
of PyTorch parameters and gradients) and the standard models bundled with
Krum.

The Model wrapper
-----------------

Krum's aggregators operate on flat gradient tensors. The
:class:`~krum.primitives.models.Model` class wraps any ``nn.Module`` and
provides zero-copy flat views:

.. code-block:: python

   from krum.primitives.models import Model
   import torch.nn as nn

   module = nn.Linear(10, 5)
   model = Model(module)

   # Flat parameter view (zero-copy, lazy-initialized)
   flat_params = model.parameters  # shape: (55,)

   # Flat gradients after backward()
   loss = module(torch.randn(3, 10)).sum()
   loss.backward()
   flat_grads = model.gradients  # shape: (55,)

   # Write aggregated gradients back (zero-copy relink)
   model.gradients = aggregated_flat

The flat tensors are **views** into the original parameter and gradient
storage, so zero-copy applies to both and no memory is copied.

How zero-copy works
-------------------

Krum's aggregators need a flat ``(n, d)`` tensor of all worker gradients.
Naively, you would copy each parameter's data into a flat vector:

.. code-block:: python

   flat = torch.nn.utils.parameters_to_vector(module.parameters())
   # flat is a *copy*: 80,000 new floats for a small MLP

With 10 workers and a CNN, copying every gradient per round adds up
fast. ``Model`` avoids this by building a flat **view** that shares the
original storage:

.. code-block:: python

   model = Model(module)

   # model.parameters is a view that shares the same memory
   flat = model.parameters
   flat[:] = 0                          # zeros every parameter in module instantly
   assert module.weight.count_nonzero() == 0  # the module sees it too

Under the hood, ``Model._relink()`` replaces each parameter's ``.data``
with a slice of a single contiguous buffer:

.. code-block:: text

   Before:   param_0.data  ──→ [w0 w1 w2 ...]
             param_1.data  ──→ [b0 b1 ...]

   After:    flat          ──→ [w0 w1 w2 ... b0 b1 ...]   ← one allocation
             param_0.data  ──→  ↑ slice
             param_1.data  ──→              ↑ slice

Because every parameter points into the same storage, reading or writing
``flat`` propagates to the module, and vice versa, with zero
data movement.

This matters for performance: for a typical CNN with 1.2 million
parameters, a flat view costs one allocation of 4.8 MB, created once,
rather than 4.8 MB of copying every aggregation round.

The same mechanism applies to gradients: ``model.gradients`` gives a
flat view of every ``param.grad``, so an aggregator's output can be
written back without copying:

.. code-block:: python

   model.gradients = aggregated_flat  # all .grad tensors relinked in place

The one caveat is that the view breaks when PyTorch replaces a
parameter's ``.data`` or ``.grad`` (e.g. after
``zero_grad(set_to_none=True)``). The relink pattern below restores it.

The relink pattern
------------------

``zero_grad(set_to_none=True)`` (the default in recent PyTorch versions)
replaces each ``.grad`` with ``None``, breaking the cached flat gradient view.
After calling ``zero_grad()``, access ``.gradients`` via
``relink_gradients()`` to restore the link in a single call:

.. code-block:: python

   optimizer.zero_grad()                    # drops .grad tensors
   grads = model.relink_gradients()         # re-link + get flat view
   grads[:] = 0                             # equivalent to zero_grad

The same pattern applies to :meth:`~krum.primitives.models.Model.relink_parameters`
when a parameter's ``.data`` has been replaced externally.

Both methods return the flat tensor directly, so no further property access is
needed.

Standard models
---------------

Krum provides standard models from the Byzantine resilience literature:

.. code-block:: python

   from krum.primitives.models.mlp import Krum2017MLPMnist, Krum2017MLPSpambase, Monna2023SmallMnist
   from krum.primitives.models.cnn import Krum2017CNN, Monna2023CNNMnist, Monna2023CNNCifar10

   # MLP for MNIST (784, 100, 10). Krum NIPS 2017.
   mlp = Krum2017MLPMnist()

   # MLP for Spambase (57, 20, 20, 2). Krum NIPS 2017.
   spambase = Krum2017MLPSpambase()

   # CNN for CIFAR-10 (3, 32, 32, 10). Hidden Vulnerability ICML 2018.
   cnn = Krum2017CNN()

   # Small MLP for MNIST (784, 128, 10). MONNA ICML 2023.
   small_mnist = Monna2023SmallMnist()

   # CNN for MNIST. MONNA ICML 2023.
   cnn_mnist = Monna2023CNNMnist()

   # CNN for CIFAR-10. MONNA ICML 2023.
   cnn_cifar = Monna2023CNNCifar10()

These models can be wrapped with ``Model`` for zero-copy flat views:

.. code-block:: python

   from krum.primitives.models import Model
   from krum.primitives.models.mlp import Krum2017MLPMnist

   model = Model(Krum2017MLPMnist())
   flat_params = model.parameters  # shape: (d,) where d ≈ 80,000

All models take no constructor arguments and instantiate fixed
architectures matching their respective papers. See
:doc:`/reference/primitives/models/index` for the full list with
architectural details.

Next steps
---------

* :doc:`using_aggregators_attacks`: see how ``Model`` gradient views
  connect to aggregation rules.
* :doc:`centralised_simulation_walkthrough`: use these models in a
  full simulation.
* :doc:`implement_aggregator`: write an aggregation rule that operates
  on the flat gradient tensors from a ``Model``.
* :doc:`/reference/primitives/models/index`: ``Model`` API reference.
