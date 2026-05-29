"""Model class encapsulating a nn.Module with zero-copy flat views.

Parameters and gradients are exposed as flat tensors via ``relink``,
so that reading or writing the flat tensor instantly reflects on the
model. Both ``model.parameters`` and ``model.gradients`` support getter
and setter semantics:

    >>> model.parameters[0] = 1.0       # modifies the model weight
    >>> model.gradients = flat           # writes aggregated grads back
"""

import torch
from torch import nn

from krum.tools.pytorch import flatten, relink


class Model:
    """Model encapsulating a nn.Module with zero-copy flat views of parameters and gradients.

    All parameters and gradients are relinked to a single contiguous buffer.
    Reading the ``parameters`` or ``gradients`` property returns a flat tensor
    sharing that buffer — modifying it modifies the model directly.
    Writing to ``model.gradients = flat`` unpacks the flat vector back into
    each parameter's ``.grad``, again with shared memory.

    The end-user is responsible for tensor copy, swap, and sharing
    (e.g. calling ``.clone()`` before sending gradients to a remote worker).

    Args:
        module: The nn.Module to encapsulate.
    """

    def __init__(self, module: nn.Module):
        """Initialize the Model.

        Args:
            module: The nn.Module to encapsulate.
        """
        self._module = module
        self._flat_parameters: torch.Tensor | None = None

    @property
    def module(self) -> nn.Module:
        """The encapsulated nn.Module.

        Returns:
            The nn.Module.
        """
        return self._module

    @module.setter
    def module(self, module: nn.Module) -> None:
        """Set a new module, invalidating the cached flat parameters.

        Args:
            module: The new nn.Module to encapsulate.
        """
        self._module = module
        self._flat_parameters = None  # Cache invalidated

    def __repr__(self) -> str:
        """Return a string representation of the Model.

        Returns:
            String with the module class name and total parameter count.
        """
        d = sum(p.numel() for p in self.module.parameters())
        return f"Model({self.module.__class__.__name__}, d={d})"

    @property
    def numel(self) -> int:
        """Total number of parameters (scalar elements).

        Returns:
            The flat dimension ``d``.
        """
        return sum(p.numel() for p in self.module.parameters())

    @property
    def parameters(self) -> torch.Tensor:
        """Zero-copy flat view of module parameters.

        The returned tensor shares memory with the model's weights.
        Modifying it modifies the model directly. Clone before sending.

        Returns:
            Tensor of shape (d,) sharing memory with all module parameters.
        """
        if self._flat_parameters is None:
            self._flat_parameters = flatten(list(self.module.parameters()))
        return self._flat_parameters

    @property
    def gradients(self) -> torch.Tensor:
        """Zero-copy flat view of module gradients.

        The returned tensor shares memory with each parameter's ``.grad``.
        If a parameter has no gradient yet, a zero-filled gradient is
        allocated and assigned to it.

        Re-flattens on every access since gradients are replaced after
        each ``backward()`` call.

        Each call to this property allocates a new buffer and runs
        ``flatten``. Do NOT keep a stale reference — always access
        ``model.gradients`` right before cloning / sending.

        Returns:
            Tensor of shape (d,) sharing memory with all module gradients.
        """
        grads = []
        for p in self.module.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            grads.append(p.grad)
        return flatten(grads)

    @gradients.setter
    def gradients(self, flat: torch.Tensor) -> None:
        """Write a flat gradient vector into each parameter's ``.grad``.

        Relinks every ``.grad`` to share a common flat buffer, so that
        accessing ``model.gradients`` afterwards returns a view of that
        same memory. If a parameter has no gradient yet, one is allocated
        first.

        Args:
            flat: Tensor of shape (d,) containing the aggregated gradients.
        """
        grads = []
        for p in self.module.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            grads.append(p.grad)
        relink(grads, flat.clone())
