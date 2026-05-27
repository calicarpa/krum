"""Model class encapsulating a nn.Module with zero-copy flat views."""

import torch
from torch import nn

from krum.tools.pytorch import flatten


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

    def set_gradients(self, gradients: torch.Tensor) -> None:
        """Write a flat gradient vector into each parameter's ``.grad``.

        Use this after receiving an aggregated gradient from the server.

        Args:
            gradients: Tensor of shape (d,) containing the aggregated gradients.
        """
        offset = 0
        for p in self.module.parameters():
            numel = p.numel()
            p.grad = gradients[offset : offset + numel].view_as(p).clone()
            offset += numel

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
        Re-flattens on every access since gradients are replaced after
        each ``backward()`` call.

        Each call to this property allocates a new buffer and runs
        ``flatten``. Do NOT keep a stale reference — always access
        ``model.gradients`` right before cloning / sending.

        Returns:
            Tensor of shape (d,) sharing memory with all module gradients.
        """
        grads = [p.grad for p in self.module.parameters()]
        return flatten(grads)
