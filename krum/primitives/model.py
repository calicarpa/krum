"""Model class encapsulating a ``Module`` with zero-copy flat views.

Parameters and gradients are exposed as flat tensors via ``relink``,
so that reading or writing the flat tensor instantly reflects on the
model. Both ``model.parameters`` and ``model.gradients`` support getter
and setter semantics:

    >>> model.parameters[0] = 1.0       # modifies the model weight
    >>> model.gradients = flat           # writes aggregated grads back
"""

from collections.abc import Iterator
import torch
from torch import Tensor
from torch.nn import Module

class Model:
    """Model encapsulating a ``Module`` with zero-copy flat views of parameters
    and gradients.

    All parameters and gradients are relinked to their respective, single
    contiguous buffer.
    Reading the ``parameters`` or ``gradients`` property returns a flat tensor
    sharing that buffer — modifying it modifies the module parameters and
    gradients directly.
    Writing to ``model.gradients = flat`` unpacks the flat vector back into each
    parameter's ``.grad``, sharing the memory of this flat vector.

    The end-user is responsible for tensor copy, swap, and sharing
    (e.g. calling ``.clone()`` before "sending" gradients to a remote worker).

    Args:
        module: The module to encapsulate.
    """

    _module: Module
    _flat_parameters: Tensor | None
    _flat_gradients: Tensor | None

    __slots__ = tuple(__annotations__)

    @classmethod
    def _relink(cls, tensors: tuple[Tensor], common: Tensor) -> Tensor:
        """Relink tensors to share a common contiguous memory storage.

        After relinking, modifying ``common`` or any individual tensor reflects
        in all others — they share the same underlying buffer.

        Args:
            tensors: Tensors to relink. All must have the same dtype.
            common: Flat tensor of sufficient size to use as underlying storage.

        Returns:
            The common tensor.
        """
        with torch.no_grad():
            offset = 0
            for tensor in tensors:
                end = offset + tensor.numel()
                tensor.data = common[offset:end].view(*tensor.shape)
                offset = end
            return common

    @classmethod
    def _flatten(cls, tensors: tuple[Tensor]) -> Tensor:
        """Flatten tensors into a single contiguous tensor sharing memory.

        Args:
            tensors: Tensors to flatten. All must have the same dtype.

        Returns:
            Flat tensor containing all data from input tensors.
            Modifications to the flat tensor reflect in the originals.
        """
        with torch.no_grad():
            common = torch.cat(tuple(tensor.view(-1) for tensor in tensors))
            return cls._relink(tensors, common)

    def __init__(self, module: Module):
        """(Re)initialize the model.

        Args:
            module: The module to encapsulate.
        """
        self._module = module
        self._flat_parameters = None
        self._flat_gradients = None

    def __repr__(self) -> str:
        """Return a string representation of the model.

        Returns:
            String with the module class name.
        """
        mname = type(self._module).__qualname__
        return f"<Model {mname!r}>"

    @property
    def module(self) -> Module:
        """The encapsulated module.

        Returns:
            The encapsulated module.
        """
        return self._module

    @module.setter
    def module(self, module: Module) -> None:
        """Set a new module, invalidating the cached flat parameters/gradients.

        Args:
            module: The new module to encapsulate.
        """
        self.__init__(module)

    @property
    def parameters(self) -> Tensor:
        """Zero-copy flat view of module parameters.

        The returned tensor shares memory with the model's weights.
        Modifying it modifies the model directly. Clone before sending.

        This method assumes the module parameters cannot change unless the
        module itself is changed.

        Returns:
            Tensor of shape (d,) sharing memory with all module parameters.
        """
        if self._flat_parameters is None:
            self._flat_parameters = self._flatten(tuple(self._module.parameters()))
        return self._flat_parameters

    @parameters.setter
    def parameters(self, flat: Tensor) -> None:
        """Zero-copy relink of the module parameters to the given flat tensor.

        Relinking the module parameters does not affect their gradients.

        Args:
            flat: Tensor of shape (d,) containing the module parameters
        """
        self._flat_parameters = self._relink(tuple(self._module.parameters()), flat)

    def _gradients(self, *, empty: bool) -> Iterator[Tensor]:
        """Iterate over the module parameters.

        Iterate over in the same order as ``_module.parameters()``, and lazily
        zero/empty-initialize gradients that have not been instantiated yet.

        Yields:
            Gradient tensors of the module parameters
        """
        for parameter in self._module.parameters():
            if parameter.grad is None:
                parameter.grad = torch.empty(0) if empty else torch.zeros_like(parameter)
            yield parameter.grad

    @property
    def gradients(self) -> Tensor:
        """Zero-copy flat view of module gradients.

        The returned tensor shares memory with each parameter's ``.grad``.
        If a parameter has no gradient yet, a zero-filled gradient is assigned.

        This method assumes the module parameters' gradients may change.
        If a parameter's ``.grad`` has been removed after the flat gradient has
        been built, it is relinked to the flat gradient with its value set to 0.
        If a parameter's ``.grad`` has been replaced and is using a different
        buffer, the ``.grad`` is copied "back" to the flat gradient's buffer,
        and then relinked to that already-existing, flat gradient's buffer.

        Returns:
            Tensor of shape (d,) sharing memory with all module gradients
        """
        if self._flat_gradients is None:
            self._flat_gradients = self._flatten(tuple(self._gradients(empty=False)))
        else:
            flat = self._flat_gradients
            storage = flat.untyped_storage()
            offset = 0
            with torch.no_grad():
                for parameter in self._module.parameters():
                    end = offset + parameter.numel()
                    if parameter.grad is None:
                        grad = flat[offset:end].view(*parameter.shape)
                        parameter.grad = grad.zero_()
                    elif parameter.grad.untyped_storage() is not storage:
                        grad = flat[offset:end].view(*parameter.shape)
                        parameter.grad = grad.copy_(parameter.grad)
                    offset = end
        return self._flat_gradients

    @gradients.setter
    def gradients(self, flat: Tensor | None) -> None:
        """Zero-copy relink of the module parameters' gradients.

        Relinks every ``.grad`` to share the ``flat`` buffer, so that accessing
        ``model.gradients`` afterwards returns a view of that same memory.

        Relinking the gradients does not affect the module parameters.

        Args:
            flat: Tensor of shape (d,) containing the gradients to set
        """
        self._flat_gradients = self._relink(tuple(self._gradients(empty=True)), flat)
