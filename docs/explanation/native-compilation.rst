Native compilation
==================

The ``native/`` folder compiles on import. If a native module is broken,
import may fail or degrade to a warning. Keep correct Python fallbacks so
that experiments can continue even when native acceleration is unavailable.

Why compile at import time?
---------------------------

The repository is designed for **rapid prototyping**:

1. Write a new rule or attack in pure Python under ``aggregators/`` or
   ``attacks/``.
2. Validate behaviors, metrics, and edge cases.
3. Once the algorithm is stable, move the hot path to ``native/``.
4. Keep exactly the same functional contract — the public API does not change.

This workflow guarantees that researchers never pay the complexity cost of
native code while an idea is still evolving. Import-time compilation removes
the need for a separate build system or Makefile maintenance.

Graceful degradation
--------------------

If compilation fails (missing CUDA toolkit, incompatible PyTorch headers,
broken C++ syntax), the framework catches the error and emits a warning. The
Python fallback implementation is then used transparently, so the experiment
can continue without manual intervention.

Environment variables
---------------------

- ``NATIVE_OPT`` — controls debug/release mode for the native build.
- ``NATIVE_STD`` — sets the C++ standard version (e.g. ``c++14``).
- ``NATIVE_QUIET`` — suppresses build messages when compiling in release mode.
