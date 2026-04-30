# Native Acceleration

The ``native/`` folder contains C++/CUDA sources that compile **automatically at Python import time**.

## Module prefixes

- ``so_`` — "Vanilla" shared object: build and load
- ``py_`` — "Python" shared object: build, load and bind as submodule (e.g. ``py_test`` will be available at ``native.test``)

## Dependencies

In a dependent SO directory, create a ``.deps`` file with a newline-separated list of dependee SO directories.

## External dependencies

- ``ninja`` (``pip install ninja``)
- CUB in ``native/include/cub`` (see https://github.com/NVlabs/cub)
- A PyTorch installation with extension compilation headers

## Environment variables

- ``NATIVE_STD`` — C++ standard (default: ``c++17``)
- ``NATIVE_OPT`` — debug/release mode
- ``NATIVE_QUIET`` — suppress build messages in release mode

## Research workflow

1. Prototype in Python in ``aggregators/`` or ``attacks/``
2. Validate behaviors and metrics
3. Move the expensive part to ``native/`` if needed
4. Keep exactly the same functional contract

If native compilation fails, Python fallbacks continue to work with a warning.
