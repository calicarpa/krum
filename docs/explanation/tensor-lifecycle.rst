Tensor lifecycle
================

The project heavily uses ``tools.flatten(...)`` and ``tools.relink(...)``.
When writing code that manipulates parameters or gradients, be careful with
views, copies, and in-place operations. A common mistake is to assume a tensor
is contiguous when it is actually a view created by ``relink``.

What flatten does
-----------------

``flatten`` takes an iterable of tensors (e.g. model parameters) and returns
a single 1-D tensor that concatenates all values. This is useful for
calculating global norms, distances, or applying vector-space aggregators.

What relink does
----------------

``relink`` takes a flat tensor and a target shape specification, then returns
a list of views that alias the original flat buffer. This is the inverse of
``flatten`` and is used to restore parameter shapes after aggregation.

The aliasing trap
-----------------

Because ``relink`` returns views, modifying one of the relinked tensors
modifies the underlying flat buffer. This is intentional for parameter
updates, but it can be surprising if you expected a copy. Always use
``tensor.clone()`` when you need an independent copy.
