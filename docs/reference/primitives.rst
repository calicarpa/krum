Primitives
==========

Core abstractions used throughout the framework.

Zero-copy flat-tensor view
--------------------------

Krum's primitives are built around a **zero-copy flat-tensor view** of PyTorch modules. Instead of working with nested parameter structures, aggregators and attacks operate on a single 1-D vector representation of the model state.

This design provides:

- **Efficiency**: No data copying on every access — the flat tensor shares memory with the module's parameters and gradients
- **Simplicity**: Aggregators work with a single ``(d,)`` tensor instead of iterating over nested structures
- **Flexibility**: Gradients can be read, modified, and written back in place

The :doc:`model` wrapper encapsulates this behavior, exposing ``.parameters`` and ``.gradients`` as flat tensors that share the underlying buffer.

.. toctree::
   :maxdepth: 1
   :caption: Core Abstractions:

   model
