Quickstart
==========

This guide walks you through installation, basic usage, and key concepts
of Krum.

Installation
------------

Supported Python versions
~~~~~~~~~~~~~~~~~~~~~~~~~

This project supports Python **3.10 through 3.14**.

From PyPI
~~~~~~~~~

.. code-block:: bash

   pip install krum

With ``uv`` (recommended):

.. code-block:: bash

   uv add krum
   # or, equivalently:
   uv pip install krum

From source
~~~~~~~~~~~

For development, or if you want to modify the source, clone the repository and
install in editable mode with the development dependencies:

.. code-block:: bash

   git clone https://github.com/calicarpa/krum.git
   cd krum
   pip install -e ".[dev]"

With ``uv`` (recommended):

.. code-block:: bash

   git clone https://github.com/calicarpa/krum.git
   cd krum
   uv sync --extra dev

Dependencies
~~~~~~~~~~~~

Krum's only runtime dependencies are **PyTorch** and **torchvision**. If you plan
to use CUDA, ensure your PyTorch build matches your CUDA version. All other
requirements are pulled in automatically when you install Krum.

Basic Usage
-----------

Here's a minimal example showing how to use Krum's aggregators and attacks:

.. code-block:: python

   import torch
   from krum.primitives.aggregators import Krum, Average
   from krum.primitives.attacks import Gaussian, SignFlip

   # Simulate gradients from 10 workers (each gradient has 100 parameters)
   n_workers = 10
   grad_dim = 100
   n_byzantine = 2

   # Honest worker gradients (normally distributed)
   honest_gradients = torch.randn(n_workers - n_byzantine, grad_dim)

   # Byzantine attack: generate malicious gradients
   attack = Gaussian(std=10.0)
   byzantine_gradients = attack.generate(honest_gradients, f=n_byzantine)

   # Combine all gradients
   all_gradients = torch.cat([honest_gradients, byzantine_gradients], dim=0)

   # Aggregate using Krum (Byzantine-resilient)
   robust_result = Krum.aggregate(all_gradients, n=n_workers, f=n_byzantine)

   # Compare with simple average (not resilient)
   naive_result = Average.aggregate(all_gradients)

   print(f"Krum result norm: {robust_result.norm().item():.4f}")
   print(f"Average result norm: {naive_result.norm().item():.4f}")

Key Concepts
------------

Aggregators
~~~~~~~~~~~

Aggregators are **stateless** gradient aggregation rules. Call them as classmethods:

.. code-block:: python

   from krum.primitives.aggregators import Average, Median, TrimmedMean, Krum, MultiKrum, Bulyan, Brute, GeoMed

   # Simple average (baseline, no resilience)
   result = Average.aggregate(gradients)

   # Coordinate-wise median (basic resilience)
   result = Median.aggregate(gradients)

   # Trimmed mean (basic resilience, requires 2f+1 workers)
   result = TrimmedMean.aggregate(gradients, f=2)

   # Krum (moderate resilience, requires 2f+3 workers)
   result = Krum.aggregate(gradients, n=10, f=2)

   # Multi-Krum (moderate resilience, averages m= n-f-2 gradients)
   result = MultiKrum.aggregate(gradients, n=10, f=2)

   # Bulyan (strong resilience, two-stage, requires 4f+3 workers)
   result = Bulyan.aggregate(gradients, n=15, f=2)

Attacks
~~~~~~~

Attacks generate Byzantine gradients from honest worker gradients:

.. code-block:: python

   from krum.primitives.attacks import SignFlip, ALIE, Gaussian, Omniscient, SmallPerturbation

   # Sign flip attack
   byzantine = SignFlip.generate(honest_gradients, f=2, scale=1.5)

   # ALIE (A Little Is Enough) attack
   byzantine = ALIE.generate(honest_gradients, f=2, z=2.0)

   # Gaussian attack
   byzantine = Gaussian.generate(honest_gradients, f=2, std=10.0)

   # Omniscient attack (requires full dataset gradient)
   byzantine = Omniscient.generate(honest_gradients, f=2, kappa=100.0, full_gradient=full_grad)

   # Small perturbation attack (exploits curse of dimensionality)
   byzantine = SmallPerturbation.generate(honest_gradients, f=2, aggregator=Krum, n=10, p=2)

Model Wrapper
~~~~~~~~~~~~~

Krum provides a ``Model`` wrapper for zero-copy flat views of PyTorch parameters and gradients:

.. code-block:: python

   from krum.primitives import Model
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

.. note::

   **``zero_grad(set_to_none=True)`` — the default since PyTorch 2.11 — replaces
   each ``.grad`` with ``None``, breaking the cached flat gradient view.**
   After calling ``zero_grad()``, access ``.gradients`` via
   ``relink_gradients()`` to restore the link in a single call:

   .. code-block:: python

      optimizer.zero_grad()                    # drops .grad tensors
      grads = model.relink_gradients()         # re-link + get flat view
      grads[:] = 0                             # equivalent to zero_grad

   The same pattern applies to :meth:`~krum.primitives.Model.relink_parameters`
   when a parameter's ``.data`` has been replaced externally.

   Both methods return the flat tensor directly, so no further property access is
   needed:

Next Steps
----------

- Browse the :doc:`reference/aggregators/index` for all available aggregation rules
- Browse the :doc:`reference/attacks/index` for all available attack strategies
- See :doc:`reference/simulations/index` for reproducing published experiments in distributed settings