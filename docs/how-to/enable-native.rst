How to enable native acceleration
===================================

By default, Krum uses pure Python implementations for all aggregators and
attacks. For performance-critical experiments, you can switch to the
C++/CUDA variants that are compiled automatically on first import.

Step 1: Verify that native compilation works
--------------------------------------------

Import an aggregator that offers a native variant. The compilation happens
silently the first time::

    python -c "import aggregators; print(aggregators.gars.keys())"

Look for names prefixed with ``native-``::

    dict_keys(['average', 'median', 'krum', 'bulyan', 'brute',
               'native-krum', 'native-median', 'native-bulyan', 'native-brute'])

If the ``native-*`` entries are present, compilation succeeded.

Step 2: Use a native variant in training
----------------------------------------

Pass the ``native-`` prefixed name to the ``--gar`` argument::

    python train.py \
        --gar native-krum \
        --nb-workers 11 \
        --nb-decl-byz 4 \
        --nb-real-byz 4 \
        --attack nan

The public API is identical to the Python version. Only the internal
computation changes.

Step 3: Troubleshoot compilation failures
-----------------------------------------

If compilation fails, check the following:

- ``ninja`` is installed::

      pip install ninja

- Your PyTorch installation includes development headers.
- The ``NATIVE_OPT`` environment variable is not forcing a debug build that
  might hide errors::

      export NATIVE_OPT=release

If compilation still fails, Krum falls back automatically to the Python
implementation and emits a warning. Your experiment will not crash.
