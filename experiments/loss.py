# coding: utf-8
###
 # @file   loss.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Loss/criterion wrappers/helpers.
###

"""
Loss and criterion wrappers for training and evaluation.

This module provides:

- :class:`Loss` — derivable loss functions (includes L1/L2 regularization
  and all PyTorch losses) with arithmetic composition support.
- :class:`Criterion` — non-derivable evaluation metrics (top-k accuracy,
  sigmoid accuracy).

Example
-------

>>> from experiments import Loss, Criterion
>>> loss = Loss("crossentropy") + 0.01 * Loss("l2")
>>> crit = Criterion("top-k", k=5)
"""

__all__ = ["Loss", "Criterion"]

import tools

import torch

# ---------------------------------------------------------------------------- #
# Loss wrapper class


class Loss:
    """
    Derivable loss function wrapper with composition support.

    Losses can be added (``loss1 + loss2``) and scaled (``0.5 * loss``).
    All standard PyTorch losses are available by lower-case name.
    Additionally, ``"l1"`` and ``"l2"`` provide parameter-norm
    regularization.

    Parameters
    ----------
    name_build : str or callable
        Loss name (e.g. ``"crossentropy"``, ``"mse"``) or a callable
        with signature ``(output, target, params) -> tensor``.
    *args : object
        Forwarded to the loss constructor when ``name_build`` is a string.
    **kwargs : object
        Forwarded to the loss constructor when ``name_build`` is a string.

    Raises
    ------
    tools.UnavailableException
        If ``name_build`` is an unknown string.
    """

    __reserved_init = object()

    @staticmethod
    def _l1loss(output, target, params):
        """
        L1 regularization on parameters.

        Parameters
        ----------
        output : torch.Tensor
            Ignored.
        target : torch.Tensor
            Ignored.
        params : torch.Tensor
            Flat parameter tensor.

        Returns
        -------
        torch.Tensor
            L1 norm of ``params``.
        """
        return params.norm(p=1)

    @staticmethod
    def _l2loss(output, target, params):
        """
        L2 regularization on parameters.

        Parameters
        ----------
        output : torch.Tensor
            Ignored.
        target : torch.Tensor
            Ignored.
        params : torch.Tensor
            Flat parameter tensor.

        Returns
        -------
        torch.Tensor
            L2 norm of ``params``.
        """
        return params.norm()

    @classmethod
    def _l1loss_builder(cls):
        """
        Build an L1 regularization loss instance.

        Returns
        -------
        Loss
            L1 loss wrapper.
        """
        return cls(cls.__reserved_init, cls._l1loss, None, "l1")

    @classmethod
    def _l2loss_builder(cls):
        """
        Build an L2 regularization loss instance.

        Returns
        -------
        Loss
            L2 loss wrapper.
        """
        return cls(cls.__reserved_init, cls._l2loss, None, "l2")

    # Map 'lower-case names' -> 'loss constructor' available in PyTorch
    __losses = None

    @staticmethod
    def _make_drop_params(builder):
        """
        Wrap a PyTorch loss builder to drop the ``params`` argument.

        Parameters
        ----------
        builder : callable
            Original loss constructor.

        Returns
        -------
        callable
            Wrapped builder returning a loss that ignores ``params``.
        """

        def drop_builder(*args, **kwargs):
            loss = builder(*args, **kwargs)

            def drop_loss(output, target, params):
                return loss(output, target)

            return drop_loss

        return drop_builder

    @classmethod
    def _get_losses(cls):
        """
        Lazily build the name-to-constructor mapping for losses.

        Returns
        -------
        dict[str, callable]
            Lower-case loss names mapped to builders.
        """
        # Fast-path already loaded
        if cls.__losses is not None:
            return cls.__losses
        # Initialize the dictionary
        cls.__losses = dict()
        # Populate with PyTorch losses
        for name in dir(torch.nn.modules.loss):
            if len(name) < 5 or name[0] == "_" or name[-4:] != "Loss":
                continue
            builder = getattr(torch.nn.modules.loss, name)
            if isinstance(builder, type):
                cls.__losses[name[:-4].lower()] = cls._make_drop_params(builder)
        # Add/replace the l1 and l2 losses
        cls.__losses["l1"] = cls._l1loss_builder
        cls.__losses["l2"] = cls._l2loss_builder
        return cls.__losses

    def __init__(self, name_build, *args, **kwargs):
        """
        Initialize the loss wrapper.

        Parameters
        ----------
        name_build : str or callable
            Loss name or constructor function.
        *args : object
            Forwarded to the constructor.
        **kwargs : object
            Forwarded to the constructor.
        """
        # Reserved custom initialization
        if name_build is type(self).__reserved_init:
            self._loss = args[0]
            self._fact = args[1]
            self._name = args[2]
            return
        # Recover name/constructor
        if callable(name_build):
            name = tools.fullqual(name_build)
            build = name_build
        else:
            losses = type(self)._get_losses()
            name = str(name_build)
            build = losses.get(name, None)
            if build is None:
                raise tools.UnavailableException(losses, name, what="loss name")
        # Build loss
        loss = build(*args, **kwargs)
        # Finalization
        self._loss = loss
        self._fact = None
        self._name = name

    def _str_make(self):
        """
        Build the formatted loss string.

        Returns
        -------
        str
            Human-readable loss description.
        """
        return self._name if self._fact is None else f"{self._fact} × {self._name}"

    def __str__(self):
        """
        Return a printable representation.

        Returns
        -------
        str
            Human-readable loss name.
        """
        return f"loss {self._str_make()}"

    def __call__(self, output, target, params):
        """
        Compute the loss.

        Parameters
        ----------
        output : torch.Tensor
            Model output.
        target : torch.Tensor
            Expected output.
        params : torch.Tensor
            Flat parameter tensor (for regularization losses).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        res = self._loss(output, target, params)
        if self._fact is not None:
            res *= self._fact
        return res

    def __add__(self, loss):
        """
        Sum two losses.

        Parameters
        ----------
        loss : Loss
            Loss to add.

        Returns
        -------
        Loss
            Composite loss.
        """

        def add(output, target, params):
            return self(output, target, params) + loss(output, target, params)

        return type(self)(
            type(self).__reserved_init,
            add,
            None,
            f"({self._str_make()} + {loss._str_make()})",
        )

    def __mul__(self, factor):
        """
        Scale the loss by a constant factor.

        Parameters
        ----------
        factor : float
            Scaling factor.

        Returns
        -------
        Loss
            Scaled loss.
        """

        def mul(output, target, params):
            return self(output, target, params) * factor

        return type(self)(
            type(self).__reserved_init,
            mul,
            factor * (1.0 if self._fact is None else self._fact),
            self._name,
        )

    def __rmul__(self, *args, **kwargs):
        """Forward to ``__mul__``."""
        return self.__mul__(*args, **kwargs)

    def __imul__(self, factor):
        """
        Scale the loss in place.

        Parameters
        ----------
        factor : float
            Scaling factor.

        Returns
        -------
        Loss
            Self.
        """
        self._fact = factor * (1.0 if self._fact is None else self._fact)
        return self


# ---------------------------------------------------------------------------- #
# Criterion wrapper class


class Criterion:
    """
    Non-derivable evaluation metric wrapper.

    Available criteria:

    - ``"top-k"`` — top-k classification accuracy.
    - ``"sigmoid"`` — binary accuracy with sigmoid threshold at 0.5.

    All criteria return a 1-D tensor ``[num_correct, batch_size]``.

    Parameters
    ----------
    name_build : str or callable
        Criterion name or constructor function.
    *args : object
        Forwarded to the criterion constructor.
    **kwargs : object
        Forwarded to the criterion constructor.

    Raises
    ------
    tools.UnavailableException
        If ``name_build`` is an unknown string.
    """

    class _TopkCriterion:
        """
        Top-k classification accuracy.
        """

        def __init__(self, k=1):
            """
            Initialize top-k criterion.

            Parameters
            ----------
            k : int, optional
                Number of top predictions to consider. Defaults to 1.
            """
            self.k = k

        def __call__(self, output, target):
            """
            Compute top-k accuracy.

            Parameters
            ----------
            output : torch.Tensor
                Batch × model logits.
            target : torch.Tensor
                Batch × target index.

            Returns
            -------
            torch.Tensor
                1-D tensor ``[num_correct, batch_size]``.
            """
            res = (
                (output.topk(self.k, dim=1)[1] == target.view(-1).unsqueeze(1))
                .any(dim=1)
                .sum()
            )
            return torch.cat(
                (
                    res.unsqueeze(0),
                    torch.tensor(
                        target.shape[0], dtype=res.dtype, device=res.device
                    ).unsqueeze(0),
                )
            )

    class _SigmoidCriterion:
        """
        Binary accuracy with 0.5 threshold.
        """

        def __call__(self, output, target):
            """
            Compute sigmoid accuracy.

            Parameters
            ----------
            output : torch.Tensor
                Batch × model logits (expected in ``[0, 1]``).
            target : torch.Tensor
                Batch × target index (expected in ``{0, 1}``).

            Returns
            -------
            torch.Tensor
                1-D tensor ``[num_correct, batch_size]``.
            """
            correct = target.sub(output).abs_() < 0.5
            res = torch.empty(2, dtype=output.dtype, device=output.device)
            res[0] = correct.sum()
            res[1] = len(correct)
            return res

    # Map 'lower-case names' -> 'criterion constructor'
    __criterions = None

    @classmethod
    def _get_criterions(cls):
        """
        Lazily build the name-to-constructor mapping.

        Returns
        -------
        dict[str, type]
            Lower-case criterion names mapped to classes.
        """
        # Fast-path already loaded
        if cls.__criterions is not None:
            return cls.__criterions
        # Initialize
        cls.__criterions = {
            "top-k": cls._TopkCriterion,
            "sigmoid": cls._SigmoidCriterion,
        }
        return cls.__criterions

    def __init__(self, name_build, *args, **kwargs):
        """
        Initialize the criterion wrapper.

        Parameters
        ----------
        name_build : str or callable
            Criterion name or constructor function.
        *args : object
            Forwarded to the constructor.
        **kwargs : object
            Forwarded to the constructor.
        """
        # Recover name/constructor
        if callable(name_build):
            name = tools.fullqual(name_build)
            build = name_build
        else:
            crits = type(self)._get_criterions()
            name = str(name_build)
            build = crits.get(name, None)
            if build is None:
                raise tools.UnavailableException(crits, name, what="criterion name")
        # Build criterion
        crit = build(*args, **kwargs)
        # Finalization
        self._crit = crit
        self._name = name

    def __str__(self):
        """
        Return a printable representation.

        Returns
        -------
        str
            Human-readable criterion name.
        """
        return f"criterion {self._name}"

    def __call__(self, output, target):
        """
        Compute the criterion.

        Parameters
        ----------
        output : torch.Tensor
            Model output.
        target : torch.Tensor
            Expected output.

        Returns
        -------
        torch.Tensor
            1-D tensor ``[num_correct, batch_size]``.
        """
        return self._crit(output, target)
