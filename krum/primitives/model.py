"""Model class encapsulating a nn.Module with zero-copy flat views."""

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from krum.tools.pytorch import relink


class Model:
    """Model encapsulating a nn.Module with zero-copy flat views of parameters and gradients.

    The end-user never needs to flatten tensors manually. Parameters and
    gradients are exposed as flat tensors that share memory with the model —
    modify the flat view and the model is updated instantly.

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
        self._flat_parameters = None

    def __repr__(self) -> str:
        """Return a string representation of the Model.

        Returns:
            String with the module class name and total parameter count.
        """
        d = sum(p.numel() for p in self.module.parameters())
        return f"Model({self.module.__class__.__name__}, d={d})"

    @property
    def parameters(self) -> torch.Tensor:
        """Zero-copy flat view of module parameters.

        The returned tensor shares memory with the model's weights.
        Modifying it modifies the model directly. Clone before sending.

        Returns:
            Tensor of shape (d,) sharing memory with all module parameters.
        """
        if self._flat_parameters is None:
            flat = parameters_to_vector(self.module.parameters())
            vector_to_parameters(flat, self.module.parameters())
            self._flat_parameters = flat
        return self._flat_parameters

    @property
    def gradients(self) -> torch.Tensor:
        """Zero-copy flat view of module gradients.

        The returned tensor shares memory with each parameter's ``.grad``.
        Re-flattens on every access since gradients are replaced after
        each ``backward()`` call.

        Returns:
            Tensor of shape (d,) sharing memory with all module gradients.
        """
        grads = [p.grad for p in self.module.parameters()]
        flat = parameters_to_vector(grads)
        relink(grads, flat)
        return flat
