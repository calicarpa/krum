Architecture
============

This page is a detailed map of the repository for researchers who want to
extend the codebase. It covers where things live, how to add new components,
and the caveats you should keep in mind.

Repository Map
--------------

.. csv-table::
   :header: "Location", "Main Role", "What Researchers Extend"
   :widths: 20, 40, 40

   "``aggregators/``", "Robust aggregation rules", "New rules, native variants, influence functions"
   "``attacks/``", "Byzantine attacks", "New attack directions, multi-worker scenarios"
   "``experiments/``", "Model / dataset / loss / criterion / optimizer", "New models, datasets, metrics"
   "``native/``", "Auto-compiled C++/CUDA extensions", "Fast implementations of hot paths"
   "``tools/``", "Cross-cutting utilities", "Logging, parallelism, parsing, registries"

Adding New Components
---------------------

Adding a New Aggregator
~~~~~~~~~~~~~~~~~~~~~~~

1. Create a new file in :doc:`aggregators/index`.
2. Expose an aggregation function and a ``check`` function.
3. Respect the keyword-only contract (``gradients``, ``f``, ``model``).
4. **Never** return a tensor that aliases an input tensor.
5. Optionally expose ``upper_bound`` and/or ``influence`` metadata.

.. seealso::

   For the full contract, see :ref:`key-concepts` on the front page.

Adding a New Attack
~~~~~~~~~~~~~~~~~~~

1. Create a new file in :doc:`attacks/index`.
2. Respect the reserved arguments: ``grad_honests``, ``f_decl``, ``f_real``,
   ``model``, ``defense``.
3. Return exactly ``f_real`` gradients.
4. Maintain separation between parameter validation (``check``) and actual
   execution.

Adding a New Model
~~~~~~~~~~~~~~~~~~

1. Create a module under ``experiments/models/``.
2. Export the constructor in ``__all__`` when possible.
3. Return a proper ``torch.nn.Module`` instance.
4. Ensure the model can be flattened and updated via ``Model.set(...)``.

Adding a New Dataset
~~~~~~~~~~~~~~~~~~~~

1. Create a module under ``experiments/datasets/``.
2. Expose a generator or dataset construction function.
3. Plan for caching if the dataset is large or downloadable.
4. Ensure ``make_datasets(...)`` (or ``Dataset(...)``) can consume it without
   invasive changes.

Important Caveats
-----------------

Debug Mode Changes Behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   Attack and aggregator wrappers choose between *checked* (validated) and
   *unchecked* (raw) versions based on the global ``__debug__`` flag.
   Running Python with ``-O`` or ``-OO`` removes validation and changes the
   error surface. Always test with validation enabled before deploying
   unchecked code.

Native Compilation May Fail Silently
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. caution::

   The ``native/`` folder compiles on import. If a native module is broken,
   import may fail or degrade to a warning. Keep correct Python fallbacks so
   that experiments can continue even when native acceleration is unavailable.

Tensors Are Often Flattened and Relinked
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project heavily uses ``tools.flatten(...)`` and ``tools.relink(...)``.
When writing code that manipulates parameters or gradients, be careful with
views, copies, and in-place operations. A common mistake is to assume a tensor
is contiguous when it is actually a view created by ``relink``.

CLI Arguments Use ``key:value`` Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Additional arguments such as ``--gar-args``, ``--attack-args``,
``--model-args``, ``--dataset-args``, ``--loss-args``, and
``--criterion-args`` are parsed by :func:`tools.parse_keyval`. Document new
research options in this ``key:value`` format so users can pass them from the
command line without ambiguity.

No Dedicated Test Suite
~~~~~~~~~~~~~~~~~~~~~~~

.. important::

   There is no automated test suite. Practical validation consists of:

   - Running a short training simulation to verify convergence behavior.
   - Importing native modules after changes to check compilation.
   - Comparing influence ratios against theoretical expectations.
