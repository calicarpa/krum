Primitives
==========

Core abstractions used throughout the framework.

Core Abstractions
-----------------

.. toctree::
   :maxdepth: 1

   models/index
   aggregators/index
   attacks/index

Zero-copy flat-tensor view
--------------------------

Krum's primitives are built around a **zero-copy flat-tensor view** of PyTorch modules. Instead of working with nested parameter structures, aggregators and attacks operate on a single 1-D vector representation of the model state.

This design provides:

- **Efficiency**: No data copying on every access — the flat tensor shares memory with the module's parameters and gradients
- **Simplicity**: Aggregators work with a single ``(d,)`` tensor instead of iterating over nested structures
- **Flexibility**: Gradients can be read, modified, and written back in place

The :class:`~krum.primitives.models.Model` wrapper encapsulates this behavior, exposing ``.parameters`` and ``.gradients`` as flat tensors that share the underlying buffer.

Because the flat gradient stores a view of the module's ``.grad`` tensors,
external operations that replace or delete those tensors — such as
``module.zero_grad(set_to_none=True)`` (the default since PyTorch 2.11) —
will leave the cached flat gradient out of sync. Use
:meth:`~krum.primitives.models.Model.relink_gradients` to restore the link in a
single call:

.. code-block:: python

   optimizer.zero_grad()                   # drops .grad tensors
   grads = model.relink_gradients()        # re-link, returns the flat tensor
   grads[:] = 0                            # equivalent to zero_grad

:meth:`~krum.primitives.models.Model.relink_parameters` provides the same for
parameters after an external ``.data`` replacement. Both methods return the
flat ``Tensor`` directly so no further property access is needed.