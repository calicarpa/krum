# coding: utf-8
###
 # @file   model.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Model wrappers/helpers.
###

"""
Model wrapper with name resolution, initialization, and gradient handling.

This module provides :class:`Model`, a unified interface that can instantiate
``torchvision`` models (by name), custom models from ``experiments/models/``,
or arbitrary callables. It also manages parameter flattening, gradient
extraction, and data-parallelism automatically.

Example
-------

>>> from experiments import Model, Configuration
>>> config = Configuration(device="cpu")
>>> model = Model("resnet18", config, num_classes=10)
>>> output = model.run(inputs)
>>> gradient, loss = model.backprop(dataset=dataset, loss=loss, outloss=True)
>>> model.update(gradient, optimizer=optimizer)
"""

__all__ = ["Model"]

import tools

import pathlib
import torch
import torchvision
import types

from .configuration import Configuration

# ---------------------------------------------------------------------------- #
# Model wrapper class


class Model:
  """
  Unified model wrapper with parameter and gradient management.

  Models are resolved by lower-case name from ``torchvision.models`` and
  from custom modules under ``experiments/models/``. Parameters are
  automatically flattened into a contiguous vector accessible via
  :meth:`get` and :meth:`set`.

  Data parallelism (``torch.nn.DataParallel``) is enabled automatically
  when the model is placed on a non-indexed CUDA device.

  Parameters
  ----------
  name_build : str or callable
      Model name (e.g. ``"resnet18"``, ``"torchvision-resnet18"``) or a
      constructor function.
  config : experiments.Configuration, optional
      Target device and dtype configuration.
  init_multi : str or callable or None, optional
      Weight initializer for tensors of dimension ``>= 2``.
  init_multi_args : dict or None, optional
      Keyword arguments for ``init_multi`` when it is a string.
  init_mono : str or callable or None, optional
      Weight initializer for tensors of dimension ``== 1``.
  init_mono_args : dict or None, optional
      Keyword arguments for ``init_mono`` when it is a string.
  *args : object
      Forwarded to the model constructor.
  **kwargs : object
      Forwarded to the model constructor.

  Raises
  ------
  tools.UnavailableException
      If ``name_build`` is an unknown string.
  tools.UserException
      If the built object is not a ``torch.nn.Module``.
  """

  # Map 'lower-case names' -> 'model constructor'
  __models = None

  # Map 'lower-case names' -> 'tensor initializer'
  __inits = None

  @classmethod
  def _get_models(cls):
    """
    Lazily build the name-to-constructor mapping for models.

    Returns
    -------
    dict[str, callable]
        Lower-case model names mapped to constructors.
    """
    # Fast-path already loaded
    if cls.__models is not None:
      return cls.__models
    # Initialize the dictionary
    cls.__models = dict()
    # Populate with TorchVision's models
    for name in dir(torchvision.models):
      if len(name) == 0 or name[0] == "_":
        continue
      builder = getattr(torchvision.models, name)
      if isinstance(builder, types.FunctionType):
        cls.__models[f"torchvision-{name.lower()}"] = builder
    # Dynamically add custom models from subdirectory 'models/'
    def add_custom_models(name, module, _):
      nonlocal cls
      exports = getattr(module, "__all__", None)
      if exports is None:
        tools.warning(
          f"Model module {name!r} does not provide '__all__'; "
          f"falling back to '__dict__' for name discovery"
        )
        exports = (
          n for n in dir(module) if len(n) > 0 and n[0] != "_"
        )
      exported = False
      for model in exports:
        if not isinstance(model, str):
          tools.warning(
            f"Model module {name!r} exports non-string name "
            f"{model!r}; ignored"
          )
          continue
        constructor = getattr(module, model, None)
        if not callable(constructor):
          continue
        exported = True
        fullname = f"{name}-{model}"
        if fullname in cls.__models:
          tools.warning(
            f"Unable to make available model {model!r} from module "
            f"{name!r}, as the name {fullname!r} already exists"
          )
          continue
        cls.__models[fullname] = constructor
      if not exported:
        tools.warning(
          f"Model module {name!r} does not export any valid "
          f"constructor name through '__all__'"
        )
    with tools.Context("models", None):
      tools.import_directory(
        pathlib.Path(__file__).parent / "models",
        {"__package__": f"{__package__}.models"},
        post=add_custom_models,
      )
    return cls.__models

  @classmethod
  def _get_inits(cls):
    """
    Lazily build the name-to-function mapping for initializers.

    Returns
    -------
    dict[str, callable]
        Lower-case initializer names mapped to functions.
    """
    # Fast-path already loaded
    if cls.__inits is not None:
      return cls.__inits
    # Initialize the dictionary
    cls.__inits = dict()
    # Populate with PyTorch's initialization functions
    for name in dir(torch.nn.init):
      if len(name) == 0 or name[0] == "_":
        continue
      if name[-1] != "_":
        continue
      func = getattr(torch.nn.init, name)
      if isinstance(func, types.FunctionType):
        cls.__inits[name[:-1]] = func
    return cls.__inits

  def __init__(
    self,
    name_build,
    config=Configuration(),
    init_multi=None,
    init_multi_args=None,
    init_mono=None,
    init_mono_args=None,
    *args,
    **kwargs,
  ):
    """
    Initialize the model wrapper.

    Parameters
    ----------
    name_build : str or callable
        Model name or constructor.
    config : experiments.Configuration, optional
        Tensor configuration.
    init_multi : str or callable or None, optional
        Multi-dimensional initializer.
    init_multi_args : dict or None, optional
        Arguments for multi-dimensional initializer.
    init_mono : str or callable or None, optional
        Mono-dimensional initializer.
    init_mono_args : dict or None, optional
        Arguments for mono-dimensional initializer.
    *args : object
        Forwarded to the model constructor.
    **kwargs : object
        Forwarded to the model constructor.
    """
    def make_init(name, args):
      inits = type(self)._get_inits()
      func = inits.get(name, None)
      if func is None:
        raise tools.UnavailableException(inits, name, what="initializer name")
      args = dict() if args is None else args
      def init(params):
        return func(params, **args)
      return init
    # Recover name/constructor
    if callable(name_build):
      name = tools.fullqual(name_build)
      build = name_build
    else:
      models = type(self)._get_models()
      name = str(name_build)
      build = models.get(name, None)
      if build is None:
        raise tools.UnavailableException(models, name, what="model name")
    # Recover initialization algorithms
    if isinstance(init_multi, str):
      init_multi = make_init(init_multi, init_multi_args)
    if isinstance(init_mono, str):
      init_mono = make_init(init_mono, init_mono_args)
    # Build model
    with torch.no_grad():
      model = build(*args, **kwargs)
      if not isinstance(model, torch.nn.Module):
        raise tools.UserException(
          f"Expected built model {name!r} to be an instance of "
          f"'torch.nn.Module', found "
          f"{getattr(type(model), '__name__', '<unknown>')!r} instead"
        )
      # Initialize parameters
      for param in model.parameters():
        if len(param.shape) > 1:
          if init_multi is not None:
            init_multi(param)
        else:
          if init_mono is not None:
            init_mono(param)
      # Move parameters to target device
      model = model.to(**config)
      device = config["device"]
      if (
        device.type == "cuda"
        and device.index is None
      ):
        model = torch.nn.DataParallel(model)
    params = tools.flatten(model.parameters())
    # Finalization
    self._model = model
    self._name = name
    self._config = config
    self._params = params
    self._gradient = None
    self._defaults = {
      "trainset": None,
      "testset": None,
      "loss": None,
      "criterion": None,
      "optimizer": None,
    }

  def __str__(self):
    """
    Return a printable representation.

    Returns
    -------
    str
        Human-readable model name.
    """
    return f"model {self._name}"

  @property
  def config(self):
    """
    Return the immutable configuration.

    Returns
    -------
    experiments.Configuration
        Model configuration.
    """
    return self._config

  def default(self, name, new=None, erase=False):
    """
    Get and/or set a named default.

    Parameters
    ----------
    name : str
        Default key (e.g. ``"trainset"``, ``"loss"``, ``"optimizer"``).
    new : object or None, optional
        New value to set. Ignored unless ``new is not None`` or
        ``erase`` is ``True``.
    erase : bool, optional
        Force the value to ``None``.

    Returns
    -------
    object
        Current (or old) value of the default.

    Raises
    ------
    tools.UnavailableException
        If ``name`` is not a known default.
    """
    if name not in self._defaults:
      raise tools.UnavailableException(
        self._defaults, name, what="model default"
      )
    old = self._defaults[name]
    if erase or new is not None:
      self._defaults[name] = new
    return old

  def _resolve_defaults(self, **kwargs):
    """
    Replace ``None`` values with registered defaults.

    Parameters
    ----------
    **kwargs : object
        Keyword arguments where ``None`` means "use the default".

    Returns
    -------
    list[object]
        Resolved values in argument order.

    Raises
    ------
    RuntimeError
        If a required default is missing.
    """
    res = list()
    for name, value in kwargs.items():
      if value is None:
        value = self.default(name)
        if value is None:
          raise RuntimeError(f"Missing default {name}")
      res.append(value)
    return res

  def run(self, data, training=False):
    """
    Forward pass through the model.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor.
    training : bool, optional
        Whether to use training mode (enables dropout, batch-norm
        updates, etc.). Defaults to evaluation mode.

    Returns
    -------
    torch.Tensor
        Model output.
    """
    if training:
      self._model.train()
    else:
      self._model.eval()
    return self._model(data)

  def __call__(self, *args, **kwargs):
    """Forward to :meth:`run`."""
    return self.run(*args, **kwargs)

  def get(self):
    """
    Get a reference to the flat parameter vector.

    Returns
    -------
    torch.Tensor
        Flat parameter tensor. Future calls to :meth:`set` will modify
        it in place.
    """
    return self._params

  def set(self, params, relink=None):
    """
    Overwrite parameters with the given flat vector.

    Parameters
    ----------
    params : torch.Tensor
        New flat parameter vector.
    relink : bool or None, optional
        Whether to relink instead of copying. ``None`` uses the
        configuration default.
    """
    # Fast path 'set(get())'-like
    if params is self._params:
      return
    # Assignment
    if (self._config.relink if relink is None else relink):
      tools.relink(self._model.parameters(), params)
      self._params = params
    else:
      self._params.copy_(
        params, non_blocking=self._config["non_blocking"]
      )

  def get_gradient(self):
    """
    Get (or create) the flat gradient vector.

    Returns
    -------
    torch.Tensor
        Flat gradient tensor. Future calls to :meth:`set_gradient` will
        modify it in place.
    """
    # Fast path
    if self._gradient is not None:
      return self._gradient
    # Flatten (make if necessary)
    gradient = tools.flatten(tools.grads_of(self._model.parameters()))
    self._gradient = gradient
    return gradient

  def set_gradient(self, gradient, relink=None):
    """
    Overwrite the gradient with the given flat vector.

    Parameters
    ----------
    gradient : torch.Tensor
        New flat gradient.
    relink : bool or None, optional
        Whether to relink instead of copying. ``None`` uses the
        configuration default.
    """
    # Fast path 'set(get())'-like
    if gradient is self._gradient:
      return
    # Assignment
    if (self._config.relink if relink is None else relink):
      tools.relink(tools.grads_of(self._model.parameters()), gradient)
      self._gradient = gradient
    else:
      self.get_gradient().copy_(
        gradient, non_blocking=self._config["non_blocking"]
      )

  def loss(self, dataset=None, loss=None, training=None):
    """
    Estimate loss on a batch from the given dataset.

    Parameters
    ----------
    dataset : experiments.Dataset or None, optional
        Dataset to sample from. ``None`` uses the default trainset.
    loss : experiments.Loss or None, optional
        Loss function. ``None`` uses the default loss.
    training : bool or None, optional
        Whether this is a training run. ``None`` guesses from
        ``torch.is_grad_enabled()``.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    dataset, loss = self._resolve_defaults(trainset=dataset, loss=loss)
    inputs, targets = dataset.sample(self._config)
    if training is None:
      training = torch.is_grad_enabled()
    return loss(self.run(inputs), targets, self._params)

  @torch.enable_grad()
  def backprop(self, dataset=None, loss=None, outloss=False, **kwargs):
    """
    Compute gradient on a batch from the given dataset.

    Parameters
    ----------
    dataset : experiments.Dataset or None, optional
        Dataset to sample from. ``None`` uses the default trainset.
    loss : experiments.Loss or None, optional
        Loss function. ``None`` uses the default loss.
    outloss : bool, optional
        Whether to also return the loss value.
    **kwargs : object
        Forwarded to ``loss.backward()``.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        Flat gradient, optionally paired with the loss value.
    """
    # Detach and zero the gradient
    for param in self._params.linked_tensors:
      grad = param.grad
      if grad is not None:
        grad.detach_()
        grad.zero_()
    # Forward and backward passes
    loss_val = self.loss(dataset=dataset, loss=loss)
    loss_val.backward(**kwargs)
    # Relink needed if graph of derivatives was created
    if "create_graph" in kwargs:
      self._gradient = None
    # Return the flat gradient (and the loss if requested)
    if outloss:
      return (self.get_gradient(), loss_val)
    return self.get_gradient()

  def update(self, gradient, optimizer=None, relink=None):
    """
    Update parameters using the given gradient and optimizer.

    Parameters
    ----------
    gradient : torch.Tensor
        Flat gradient to apply.
    optimizer : experiments.Optimizer or None, optional
        Optimizer wrapper. ``None`` uses the default optimizer.
    relink : bool or None, optional
        Whether to relink the gradient. ``None`` uses the configuration
        default.
    """
    optimizer = self._resolve_defaults(optimizer=optimizer)[0]
    self.set_gradient(
      gradient,
      relink=(self._config.relink if relink is None else relink),
    )
    optimizer.step()

  @torch.no_grad()
  def eval(self, dataset=None, criterion=None):
    """
    Evaluate the model on a batch from the given dataset.

    Parameters
    ----------
    dataset : experiments.Dataset or None, optional
        Dataset to sample from. ``None`` uses the default testset.
    criterion : experiments.Criterion or None, optional
        Criterion function. ``None`` uses the default criterion.

    Returns
    -------
    torch.Tensor
        Mean criterion value over the sampled batch.
    """
    dataset, criterion = self._resolve_defaults(
      testset=dataset, criterion=criterion
    )
    inputs, targets = dataset.sample(self._config)
    return criterion(self.run(inputs), targets)
