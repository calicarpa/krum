Native Acceleration
===================

The ``native`` package provides automated building and loading of C++/CUDA
native PyTorch extensions. There is no separate build step; importing an
aggregator or attack that offers a native variant triggers compilation
transparently.

For the design rationale behind import-time compilation — why the project
trades a separate build system for zero-friction prototyping — see the explanation of
:doc:`/explanation/native-compilation`.

Overview
--------

.. list-table:: Native Components
   :header-rows: 1
   :widths: 20 60 20

   * - Component
     - Description
     - Environment Control
   * - Module loader
     - Scans subdirectories, compiles sources, injects ``py_`` modules
     - ``NATIVE_OPT``, ``NATIVE_STD``, ``NATIVE_QUIET``
   * - Dependency resolver
     - Reads ``.deps`` files and builds dependencies recursively
     - —
   * - CUDA support
     - Detects ``.cu`` files and compiles with NVCC when available
     - —
   * - Thread pool
     - Shared C++ threadpool dependency (``so_threadpool``) for parallel ops
     - —

Available Native Variants
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

Internal Dependencies
---------------------

Some modules under ``native/`` are shared libraries (``so_`` prefix) that
provide infrastructure for other native variants rather than exposing a
public API:

- ``so_threadpool`` — C++ thread pool used by ``native-bulyan`` for
  parallel distance computations.

These libraries are built automatically when required by a dependent module.

API Reference
-------------

.. automodule:: native
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index: