Primitives
==========

Core abstractions used throughout the framework. The primitives layer provides
the foundational building blocks that aggregators, attacks, and simulations
depend on.

Zero-copy flat-tensor view
--------------------------

The :class:`~primitives.model.Model` class is a zero-copy wrapper around
:class:`torch.nn.Module` that exposes flat views of parameters and gradients.
This is essential for gradient-level operations used by aggregators and attacks.

.. code-block:: python

   from krum.primitives import Model
   import torch.nn as nn

   model = nn.Sequential(
       nn.Linear(784, 100),
       nn.ReLU(),
       nn.Linear(100, 10),
   )
   krum_model = Model(model)

   # Zero-copy flat views (no memory overhead)
   flat_params = krum_model.flat_params  # shape: (d,)
   flat_grads = krum_model.flat_grads    # shape: (d,) after backward()

   # Load updated flat parameters back into the model
   krum_model.load_flat_params(new_flat_params)

.. automodule:: primitives.model
   :members:
   :undoc-members:
   :show-inheritance:

Attack Base Class
-----------------

All attacks inherit from :class:`~primitives.attacks.Attack`. The base class
defines the ``generate`` contract that every concrete attack must implement:

.. code-block:: python

   class Attack:
       @classmethod
       def generate(gradients: torch.Tensor, f: int, **kwargs) -> torch.Tensor:
           ...

.. automodule:: primitives.attacks
   :members:
   :undoc-members:
   :show-inheritance:

Aggregator Pattern
------------------

Aggregators follow a consistent pattern: each is a class with a single
``aggregate`` classmethod. They are stateless — all configuration is passed
as keyword arguments.

.. code-block:: python

   from krum.primitives.aggregators import Average

   # Usage: classmethod, no instantiation needed
   result = Average.aggregate(gradients)
