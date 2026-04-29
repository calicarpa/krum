Native Acceleration
===================

The ``native/`` folder contains C++/CUDA sources that are compiled
automatically at Python import time. There is no separate build step;
importing an aggregator or attack that offers a native variant triggers
the compilation transparently.

For the design rationale behind import-time compilation — why the project
trades a separate build system for zero-friction prototyping — see
:doc:`/explanation/native-compilation`.

Loader behaviour
----------------

When the ``native`` package is first imported, the loader:

1. Inspects every subdirectory of ``native/``.
2. Recognises ``so_`` and ``py_`` prefixes on source files.
3. Collects files whose extensions match the allowed set.
4. Invokes ``torch.utils.cpp_extension.load`` to compile them.
5. Reads ``.deps`` files to discover and load extra dependencies.
6. Exposes the compiled modules in the ``native`` namespace.

If compilation fails, the framework degrades gracefully: Python fallbacks
continue to work and a warning is emitted.

Environment variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Variable
     - Effect
   * - ``NATIVE_OPT``
     - Controls debug/release mode for the native build.
   * - ``NATIVE_STD``
     - Sets the C++ standard version (e.g. ``c++14``).
   * - ``NATIVE_QUIET``
     - Suppresses build messages when compiling in release mode.

External dependencies
---------------------

- ``ninja`` — the build system used by PyTorch extension compilation.
- CUB in ``native/include/cub`` — CUDA primitives.
- A PyTorch installation with headers compatible with extension compilation.

Available native variants
-------------------------

When compilation succeeds, native variants are registered alongside their
Python counterparts with a ``native-`` prefix:

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Name
     - What is accelerated
   * - ``native-krum``
     - Pairwise distance computation and scoring.
   * - ``native-median``
     - Coordinate-wise median.
   * - ``native-bulyan``
     - Multi-Krum selection and median averaging.
   * - ``native-brute``
     - Combinatorial subset search.

Because the high-level API is identical, you can compare the exact same rule
in Python and native versions simply by changing the registered name.