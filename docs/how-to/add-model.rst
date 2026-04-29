How to add a new model
======================

1. Create a module under ``experiments/models/``.
2. Export the constructor in ``__all__`` when possible.
3. Return a proper ``torch.nn.Module`` instance.
4. Ensure the model can be flattened and updated via ``Model.set(...)``.
