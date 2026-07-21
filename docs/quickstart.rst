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

* :doc:`tutorials/first_simulation` — run a full simulation with the Orchestrator
* :doc:`tutorials/using_aggregators_attacks` — how to use all built-in aggregators and attacks
* :doc:`tutorials/working_with_models` — zero-copy flat tensor views and standard models
