Quickstart
==========

This guide walks you through installing Krum and running your first command.

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
   pip install -e ".[dev,experiments]"

With ``uv`` (recommended):

.. code-block:: bash

   git clone https://github.com/calicarpa/krum.git
   cd krum
   uv sync --all-extras --all-groups

Dependencies
~~~~~~~~~~~~

Krum's runtime dependencies are **PyTorch**, **torchvision**, and **pandas**.
If you plan to use CUDA, ensure your PyTorch build matches your CUDA version.
For experiments and visualisations, install the optional extras:

.. code-block:: bash

   pip install "krum[experiments]"

This adds ``matplotlib``, ``numpy``, and ``seaborn``.

Sanity check
------------

.. code-block:: python

   import torch
   from krum.primitives.aggregators.krum import Krum

   result = Krum.aggregate(torch.randn(10, 100), n=10, f=2)
   print(result.shape)  # (100,)

Next steps
----------

Dive into the :doc:`tutorials/index` for step-by-step guides:

* :doc:`tutorials/using_simulations` — using the built-in simulations
* :doc:`tutorials/using_aggregators_attacks` — how to use all built-in aggregators and attacks
* :doc:`tutorials/working_with_models` — zero-copy flat tensor views and standard models
