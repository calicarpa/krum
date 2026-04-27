Architecture
============

This section provides a detailed map of the repository for researchers extending the codebase.

Repository Map
--------------

.. csv-table::
   :header: "Location", "Main Role", "What Researchers Extend"
   :widths: 20, 40, 40

   "``aggregators/``", "Robust aggregation rules", "New rules, native variants, influence"
   "``attacks/``", "Byzantine attacks", "New attack directions, scenarios"
   "``experiments/``", "Model/dataset/loss/criterion/optimizer", "New models, datasets, metrics"
   "``native/``", "Auto-compiled C++/CUDA extensions", "Fast implementations of rules"
   "``tools/``", "Cross-cutting utilities", "Logging, parallelism, parsing"

Adding New Components
---------------------

Adding a New Aggregator
~~~~~~~~~~~~~~~~~~~~~~~

1. Create a file in ``aggregators/``
2. Expose an aggregation function and a ``check`` function
3. Respect the keyword-only contract
4. Never return a tensor that aliases an input tensor
5. Optionally expose theoretical bounds or influence ratios

Adding a New Attack
~~~~~~~~~~~~~~~~~~~

1. Create a file in ``attacks/``
2. Respect reserved arguments: ``grad_honests``, ``f_decl``, ``f_real``, ``model``, ``defense``
3. Return exactly ``f_real`` gradients
4. Maintain separation between verification and actual execution

Adding a New Model
~~~~~~~~~~~~~~~~~~

1. Create a module in ``experiments/models/``
2. Export the constructor in ``__all__`` if possible
3. Return a proper ``torch.nn.Module`` instance
4. Ensure the model can be flattened and updated via ``Model.set(...)``

Adding a New Dataset
~~~~~~~~~~~~~~~~~~~~

1. Create a module in ``experiments/datasets/``
2. Expose a generator or dataset construction function
3. Plan for caching if the dataset is large or downloadable
4. Ensure ``make_datasets(...)`` or ``Dataset(...)`` can use it without invasive changes

Important Caveats
-----------------

Debug Mode Changes Behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Attack and aggregator wrappers choose between verified and raw versions based on ``__debug__``. The ``-O`` or ``-OO`` mode changes the validation surface.

Native Compilation May Fail Import
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``native/`` folder compiles on import. If a native module is broken, import may fail or degrade to a warning. Keep correct Python fallbacks.

Tensors Are Often Flattened and Relinked
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project heavily uses ``tools.flatten(...)`` and ``tools.relink(...)``. When adding objects that manipulate parameters or gradients, be careful with views, copies, and inplace operations.

CLI Additional Arguments Use ``key:value`` Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Additional arguments ``--gar-args``, ``--attack-args``, ``--model-args``, ``--dataset-args``, ``--loss-args``, ``--criterion-args`` are parsed by ``tools.parse_keyval(...)``. Document new research options in this format.

No Dedicated Test Suite
~~~~~~~~~~~~~~~~~~~~~~~

Practical validation consists of:
- Running a short training simulation
- Importing native modules if modified
