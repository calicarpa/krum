Datasets
========

.. automodule:: experiments.datasets.svm
   :members:
   :undoc-members:
   :show-inheritance:

.. seealso::

   For the dataset wrapper that loads these constructors, see :doc:`dataset`.

Adding a custom dataset
-----------------------

Custom datasets are discovered automatically from the ``experiments/datasets/``
directory. To register a new dataset, create a Python file in that folder
(e.g. ``experiments/datasets/mydata.py``). The file name (without the
``.py`` extension) becomes the module prefix.

The module must define ``__all__`` as a list of builder names. Every name
listed there must refer to a callable that returns an infinite batch
generator (or any object compatible with :class:`experiments.Dataset`).
The wrapper registers each builder under the compound name
``<file_name>-<builder_name>``.

For example, if you link an external repository under
``submodules/mydata`` and create ``experiments/datasets/mydata.py`` that
exposes a callable ``load``, the dataset can be used with::

    Dataset("mydata-load", root="...")

Because the loader only scans regular ``.py`` files, a symlink pointing to
code inside ``submodules/`` works transparently.
