Installation
============

Supported Python versions
-------------------------

This project supports Python **3.10 through 3.14**.

From PyPI
---------

.. code-block:: bash

   pip install krum

With ``uv`` (recommended):

.. code-block:: bash

   uv pip install krum
   # or directly in a uv project
   uv add krum

From source
-----------

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

This installs all linting, type-checking, and documentation tools.

Dependencies
------------

Krum's only runtime dependencies are **PyTorch** and **torchvision**. If you plan
to use CUDA, ensure your PyTorch build matches your CUDA version. All other
requirements are pulled in automatically when you install Krum.
