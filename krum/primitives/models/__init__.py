"""Zero-copy flat-tensor view of a ``torch.nn.Module`` and standard models.

The :class:`Model` wrapper relinks a module's parameters and gradients to
contiguous flat 1-D tensors, so that aggregators and attacks can operate on
a single vector representation of the model state without copying data on
every access. Reading :attr:`Model.parameters` or :attr:`Model.gradients`
returns a view sharing the underlying buffer; writing to
``model.gradients = flat`` unpacks the flat vector back into each
parameter's ``.grad`` in place.

Standard models (:class:`Krum2017MLPMnist`, :class:`Krum2017MLPSpambase`,
:class:`Krum2017CNN`, :class:`Monna2023SmallMnistNet`) are provided for the
simulation protocols from the literature. These models implement the
architectures used in the foundational papers on Byzantine-resilient
distributed learning (NIPS 2017, ICML 2018, ICML 2023) and can be reused
for custom experiments or as baselines.

Example::

    from torch.nn import Linear
    from krum.primitives.models import Model

    model = Model(Linear(4, 2))

    # Read the flat parameter vector (shares memory with the module)
    params = model.parameters          # shape (d,)

    # Simulate a backward pass, then read gradients
    loss = model.module(torch.randn(3, 4)).sum()
    loss.backward()
    grads = model.gradients            # shape (d,), lazy-initializes .grad

    # Write aggregated gradients back (zero-copy relink)
    model.gradients = grads.clone()

Using standard models::

    from krum.primitives.models import Krum2017MLPMnist, Krum2017CNN, Krum2017MLPSpambase, Model

    # Use a standard model directly
    mlp = Krum2017MLPMnist()           # MNIST: 784 → 100 → 10
    cnn = Krum2017CNN()                # CIFAR-10: 3×32×32 → 10
    spambase = Krum2017MLPSpambase()   # Spambase: 57 → 20 → 20 → 2

    # Wrap with Model for zero-copy flat views
    wrapped = Model(mlp)
    print(wrapped.parameters.shape)    # (d,) where d ≈ 80,000
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor
from torch.nn import Module

from .cnn import Krum2017CNN
from .mlp import Krum2017MLPMnist, Krum2017MLPSpambase, Monna2023SmallMnistNet

__all__ = [
    "Krum2017CNN",
    "Krum2017MLPMnist",
    "Krum2017MLPSpambase",
    "Model",
    "Monna2023SmallMnistNet",
]


class Model:
    """Model encapsulating a ``Module`` with zero-copy flat views of parameters and gradients.

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

    __slots__ = __annotations__

    @classmethod
    def _relink(cls, tensors: tuple[Tensor, ...], common: Tensor) -> Tensor:
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
    def _flatten(cls, tensors: tuple[Tensor, ...]) -> Tensor:
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

    def __init__(self, module: Module) -> None:
        """Initialize the Model wrapper.

        Args:
            module: The module to encapsulate.
        """
        self._module = module
        self._flat_parameters = None
        self._flat_gradients = None

    def _reset(self) -> None:
        """Invalidate cached flat views, e.g. after swapping the encapsulated module."""
        self._flat_parameters = None
        self._flat_gradients = None

    def __repr__(self) -> str:
        """Return a string representation of the model.

        Returns:
            A ``<Model 'ModuleName'>`` style string.
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
        self._module = module
        self._reset()

    @property
    def parameters(self) -> Tensor:
        r"""Zero-copy flat view of module parameters.

        The returned tensor shares memory with the model's weights.
        Modifying it modifies the model directly. Clone before sending.

        This is a simple, fast getter once the flat parameters have been
        lazy-initialized on the first access.  If a parameter's ``.data``
        has been replaced externally, call :meth:`relink_parameters` to
        re-synchronise the flat view.

        Returns:
            Tensor of shape ``(d,)`` sharing memory with all module
            parameters, in the iteration order of ``module.parameters()``.
        """
        if self._flat_parameters is None:
            self._flat_parameters = self._flatten(tuple(self._module.parameters()))
        return self._flat_parameters

    @parameters.setter
    def parameters(self, flat: Tensor) -> None:
        r"""Zero-copy relink of the module parameters to the given flat tensor.

        Relinking the module parameters does not affect their gradients.

        Args:
            flat: Tensor of shape :math:`(d,)` containing the module parameters,
                in the same flattening order as :attr:`parameters`.
        """
        self._flat_parameters = self._relink(tuple(self._module.parameters()), flat)

    def relink_parameters(self) -> Tensor:
        """Re-synchronise the flat parameter view after an external ``.data`` replacement.

        This method is necessary when a parameter's ``.data`` has been
        replaced since the flat parameters were built, restoring the zero-copy
        link between the cached flat tensor and every parameter so that the
        fast ``.parameters`` getter yields a consistent view again.

        Returns:
            The flat parameter tensor of shape ``(d,)`` sharing memory with
            all module parameters.
        """
        flat = self.parameters
        storage = flat.untyped_storage()
        with torch.no_grad():
            offset = 0
            for parameter in self._module.parameters():
                end = offset + parameter.numel()
                if parameter.data.untyped_storage() is not storage:
                    parameter.data = flat[offset:end].view(*parameter.shape).copy_(parameter.data)
                offset = end
        return flat

    def _gradients(self, *, empty: bool) -> Iterator[Tensor]:
        """Iterate over module parameter gradients, initializing missing ones.

        Iterates in the same order as ``_module.parameters()``, and lazily
        initializes gradients that have not been instantiated yet.

        Args:
            empty: If ``True``, allocate empty tensors instead of zero-filled
                ones. Used by the gradients setter, which then overwrites them
                with the caller's flat tensor.

        Yields:
            Gradient tensors of the module parameters.
        """
        for parameter in self._module.parameters():
            if parameter.grad is None:
                parameter.grad = torch.empty_like(parameter) if empty else torch.zeros_like(parameter)
            yield parameter.grad

    @property
    def gradients(self) -> Tensor:
        r"""Zero-copy flat view of module gradients.

        The returned tensor shares memory with each parameter's ``.grad``.
        If a parameter has no gradient yet, a zero-filled gradient is assigned.

        This is a simple, fast getter once the flat gradient has been
        lazy-initialized on the first access. If gradients are replaced or
        removed externally (e.g. via ``module.zero_grad(set_to_none=True)``),
        call :meth:`relink_gradients` to re-synchronise the flat view.

        Returns:
            Tensor of shape ``(d,)`` sharing memory with all module
            gradients, in the iteration order of ``module.parameters()``.
        """
        if self._flat_gradients is None:
            self._flat_gradients = self._flatten(tuple(self._gradients(empty=False)))
        return self._flat_gradients

    def relink_gradients(self) -> Tensor:
        """Re-synchronise the flat gradient view after an external ``.grad`` replacement.

        This method is necessary when a parameter's ``.grad`` attribute has
        been replaced or removed since the flat gradient was built, for
        instance after ``module.zero_grad(set_to_none=True)``. It restores the
        zero-copy link between the cached flat tensor and every ``.grad``
        so that the fast ``.gradients`` getter yields a consistent view again.

        Returns:
            The flat gradient tensor of shape ``(d,)`` sharing memory with
            all module gradients.
        """
        flat = self.gradients
        storage = flat.untyped_storage()
        offset = 0
        with torch.no_grad():
            for parameter in self._module.parameters():
                end = offset + parameter.numel()
                # The gradient was deleted
                if parameter.grad is None:
                    grad = flat[offset:end].view(*parameter.shape)
                    parameter.grad = grad.zero_()
                # The gradient was replaced by a tensor with a different storage
                elif parameter.grad.untyped_storage() is not storage:
                    parameter.grad.data = flat[offset:end].view(*parameter.shape).copy_(parameter.grad)
                offset = end
        return flat

    @gradients.setter
    def gradients(self, flat: Tensor) -> None:
        r"""Zero-copy relink of the module parameters' gradients.

        Relinks every ``.grad`` to share the ``flat`` buffer, so that accessing
        :attr:`gradients` afterwards returns a view of that same memory.
        Relinking the gradients does not affect the module parameters.

        Args:
            flat: Tensor of shape :math:`(d,)` containing the gradients to set,
                in the same flattening order as :attr:`gradients`.
        """
        self._flat_gradients = self._relink(tuple(self._gradients(empty=True)), flat)
