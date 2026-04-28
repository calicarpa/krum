Models
======

.. automodule:: experiments.models.simples
   :members:
   :undoc-members:
   :show-inheritance:

.. seealso::

   For the model wrapper that loads these constructors, see :doc:`model`.

Adding a custom model
---------------------

Custom models are discovered automatically from the ``experiments/models/``
directory. To register a new model, create a Python file in that folder
(e.g. ``experiments/models/wide_resnet.py``). The file name (without the
``.py`` extension) becomes the module prefix.

The module must define ``__all__`` as a list of constructor names. Every
name listed there must refer to a callable that returns a
``torch.nn.Module``. The wrapper registers each constructor under the
compound name ``<file_name>-<constructor_name>``.

For example, if you link an external repository under
``submodules/wide-resnet`` and create ``experiments/models/wide-resnet.py``
that exposes a callable ``wide_resnet``, the model can be instantiated with::

    Model("wide-resnet-wide_resnet", config, num_classes=10)

Because the loader only scans regular ``.py`` files, a symlink pointing to
code inside ``submodules/`` works transparently.
