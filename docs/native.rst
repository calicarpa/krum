Native Acceleration
===================

The ``native/`` folder is not just a repository of C++/CUDA sources. It compiles at Python import time.

What the Loader Does
--------------------

- Inspects subdirectories of ``native/``
- Recognizes ``so_`` and ``py_`` prefixes
- Collects source files matching allowed extensions
- Compiles with ``torch.utils.cpp_extension.load``
- Loads dependencies declared by ``.deps`` files
- Exposes compiled Python modules in the ``native`` namespace

Important Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``NATIVE_OPT``: controls debug/release mode for native build
- ``NATIVE_STD``: sets the C++ standard version
- ``NATIVE_QUIET``: suppresses build messages in release mode

External Dependencies
~~~~~~~~~~~~~~~~~~~~~

- ``ninja``
- CUB in ``native/include/cub``
- A PyTorch installation compatible with extension compilation

Why This Matters for Research
------------------------------

The repository was clearly designed to allow rapid Python prototyping, then native acceleration when an idea becomes stable enough to speed up.

The workflow:

1. Prototype in Python in ``aggregators/`` or ``attacks/``
2. Validate behaviors and metrics
3. Move the expensive part to ``native/`` if needed
4. Keep exactly the same functional contract

If native is unavailable, Python code generally continues to work with warnings or fallbacks.

Available Native Modules
------------------------

When compiled, native variants are exposed as ``native-krum``, ``native-median``, ``native-bulyan``, etc. These serve mainly for hot paths: pairwise distance calculation, subset selection, coordinate-wise median, expensive combinatorics.

The high-level API remains the same, allowing exact comparison of the same rule in Python and native versions.