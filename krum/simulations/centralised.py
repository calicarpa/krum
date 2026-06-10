r"""Parameter-server distributed SGD simulation confronting Byzantine workers.

Each synchronous round:
    #. Honest workers compute a gradient on their local data shard.
    #. Byzantine workers craft adversarial gradients.
    #. The aggregator combines all :math:`n` gradients into a single update.
    #. The aggregated update is applied via an SGD step.

The :class:`CentralisedSimulation` implements the full lifecycle (model
initialisation, data sharding, training loop, evaluation). Protocol-specific
metric reporting is configured via the ``evaluate_fn`` constructor parameter
rather than subclassing — evaluation is delegated to a caller-supplied
callable, following composition over inheritance.
"""

from typing import Any, Callable, Literal, Sized, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from krum.primitives.aggregators import Aggregator
from krum.primitives.attacks import Attack
from krum.primitives.attacks.full_gradient_negation import FullGradientNegationAttack
from krum.primitives.model import Model


class CentralisedSimulation:
    r"""Parameter-server distributed SGD simulation with Byzantine workers.

    One instance = one (aggregator, attack, dataset, model) configuration run
    over ``rounds`` synchronous rounds. The training set is IID-sharded across
    :math:`n` workers, of which :math:`f` are Byzantine (up to the tolerance of the
    chosen aggregator).

    Evaluation follows composition over inheritance: the caller supplies
    an ``evaluate_fn`` callable (e.g. one of the built-in
    :meth:`evaluate_test_error_and_loss` or :meth:`evaluate_full`) that
    receives the simulation instance and returns protocol-specific metrics.

    Args:
        model_cls: ``nn.Module`` subclass to instantiate for training.
        train_set: Full training dataset (will be IID-sharded across workers).
        test_set: Test dataset (evaluated via full-batch loader).
        aggregator: Gradient aggregation rule class (e.g. ``Average``, ``Krum``).
            Pass the class itself — :meth:`~CentralisedSimulation.step` calls
            ``aggregator.aggregate(gradients, n=n, f=f, **aggregator_kwargs)``.
        aggregator_kwargs: Extra keyword arguments forwarded to
            ``aggregator.aggregate`` (e.g. ``{"m": 12}`` for ``MultiKrum``).
            ``n`` and ``f`` are automatically injected by the simulation.
        attack: Byzantine attack strategy class (e.g. ``GaussianAttack``).
            Pass the class itself — :meth:`~CentralisedSimulation.step` calls
            ``attack.generate(honest_gradients, f=f, **attack_kwargs)``.
        attack_kwargs: Extra keyword arguments forwarded to
            ``attack.generate`` (e.g. ``{"std": 200.0}`` for ``GaussianAttack``).
        n: Total number of workers.
        f: Number of Byzantine workers. Must be within the aggregator's
            Byzantine tolerance.
        rounds: Number of synchronous training rounds.
        batch_size: Mini-batch size per honest worker.
        lr: Initial learning rate for SGD. Used as :math:`η_0` by every
            supported scheduler.
        lr_schedule: Learning-rate schedule applied after each round.
            ``"exponential"`` uses multiplicative decay ``lr_decay``
            per round (the default for the ICML 2018 protocol).
            ``"robbins_monro"`` uses the
            :math:`η(t) = r_η · η_0 / (t + r_η)` schedule of El Mhamdi
            et al. (ICML 2018), with fading rate ``r_eta``. ``"none"``
            keeps a constant learning rate (the default for the NIPS
            2017 protocol).
        lr_decay: Multiplicative learning-rate decay per round, used
            when ``lr_schedule == "exponential"``. ``None`` disables
            the scheduler in that mode. Default: ``0.99``.
        r_eta: Fading rate for the Robbins-Monro schedule
            :math:`η(t) = r_η · η_0 / (t + r_η)`. Required when
            ``lr_schedule == "robbins_monro"``.
        weight_decay: :math:`ℓ_2` regularization coefficient applied
            directly on the flat parameter tensor. Set to ``0.0``
            (default) to disable. Section 5.1 of El Mhamdi et al. (ICML 2018)
            recommends ``1e-4``.
        xavier_init: When ``True``, apply Glorot/Xavier uniform
            initialization to every weight tensor of the model after
            construction, and zero-initialize biases. Matches Section
            5.1 of El Mhamdi et al. (ICML 2018).
        stop_attack_at: If set, the Byzantine attack is disabled at
            round ``t = stop_attack_at``: the ``f`` Byzantine workers
            then send zero gradients. Used by Experiment 1 / Figure 2
            of El Mhamdi et al. (ICML 2018), where the attack is
            maintained only up to round 50.
        loss_fn: Per-sample loss function. Default: ``cross_entropy``.
        device: Device for training and evaluation. Auto-detected if ``None``
            (CUDA → MPS → CPU).
        seed: Random seed for reproducibility.
        eval_every: Evaluate on the test set every ``eval_every`` rounds
            (and always on the last round).
        evaluate_fn: Callable that receives the simulation instance
            and returns evaluation metrics. Built-in options include
            :meth:`evaluate_test_error`, :meth:`evaluate_test_error_and_loss`,
            and :meth:`evaluate_full`. Custom evaluators must accept a
            single ``CentralisedSimulation`` argument.

    Raises:
        RuntimeError: If :meth:`run` is called more than once on the same
            instance.
        ValueError: If ``lr_schedule`` is invalid or required
            hyperparameters (:math:`r_eta`) are missing.
    """

    def __init__(
        self,
        *,
        model_cls: type[nn.Module],
        train_set: Dataset[Any],
        test_set: Dataset[Any],
        aggregator: type[Aggregator],
        aggregator_kwargs: dict[str, Any] | None = None,
        attack: type[Attack],
        attack_kwargs: dict[str, Any] | None = None,
        n: int,
        f: int,
        rounds: int,
        batch_size: int,
        lr: float,
        lr_schedule: Literal["exponential", "robbins_monro", "none"] = "exponential",
        lr_decay: float | None = 0.99,
        r_eta: float | None = None,
        weight_decay: float = 0.0,
        xavier_init: bool = False,
        stop_attack_at: int | None = None,
        aggregator_f: int | None = None,
        loss_fn: Callable[..., torch.Tensor] = nn.functional.cross_entropy,
        device: torch.device | None = None,
        seed: int = 42,
        eval_every: int = 10,
        evaluate_fn: Callable[["CentralisedSimulation"], Any] | None = None,
    ) -> None:
        """See the class docstring for the full parameter list."""
        if evaluate_fn is None:
            raise TypeError("CentralisedSimulation.__init__() missing required argument: 'evaluate_fn'")
        if lr_schedule not in {"exponential", "robbins_monro", "none"}:
            raise ValueError(
                f"Invalid lr_schedule, got {lr_schedule!r}, expected 'exponential', 'robbins_monro', or 'none'"
            )
        if lr_schedule == "robbins_monro" and r_eta is None:
            raise ValueError("lr_schedule='robbins_monro' requires the r_eta parameter")
        if lr_schedule == "exponential" and lr_decay is None:
            raise ValueError(
                "lr_schedule='exponential' with lr_decay=None is ambiguous; "
                "use lr_schedule='none' to disable scheduling"
            )
        if r_eta is not None and r_eta <= 0:
            raise ValueError(f"Invalid r_eta, got {r_eta!r}, expected r_eta > 0")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay, got {weight_decay!r}, expected weight_decay >= 0")
        if stop_attack_at is not None and stop_attack_at < 0:
            raise ValueError(f"Invalid stop_attack_at, got {stop_attack_at!r}, expected stop_attack_at >= 0")
        if aggregator_f is not None and aggregator_f < 0:
            raise ValueError(f"Invalid aggregator_f, got {aggregator_f!r}, expected aggregator_f >= 0")

        self.model_cls = model_cls
        self.train_set = train_set
        self.test_set = test_set
        self.aggregator = aggregator
        self._aggregator_kwargs = aggregator_kwargs or {}
        self.attack = attack
        self._attack_kwargs = attack_kwargs or {}
        self.n = n
        self.f = f
        self.aggregator_f = aggregator_f if aggregator_f is not None else f
        self.rounds = rounds
        self.batch_size = batch_size
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.lr_decay = lr_decay
        self.r_eta = r_eta
        self.weight_decay = weight_decay
        self.xavier_init = xavier_init
        self.stop_attack_at = stop_attack_at
        self.loss_fn = loss_fn
        self.device = device or self._detect_device()
        self.seed = seed
        self.eval_every = eval_every
        self._evaluate_fn = evaluate_fn

        self._model: Model | None = None
        self._current_lr: float = self.lr
        self._worker_loaders: list[DataLoader[Any]] = []
        self._full_loader: DataLoader[Any] | None = None
        self._test_loader: DataLoader[Any] | None = None
        self._has_run = False
        self._current_round = 0

    @property
    def model(self) -> Model:
        """The encapsulated :class:`~krum.primitives.model.Model`, available after :meth:`setup` or :meth:`run`.

        Returns:
            The wrapped ``nn.Module`` with zero-copy flat parameter/gradient views.

        Raises:
            RuntimeError: If the simulation has not been set up yet.
        """
        if self._model is None:
            raise RuntimeError("Simulation not set up. Call setup() or run() first.")
        return self._model

    def setup(self) -> None:
        """Initialise the model, learning rate, IID data shards, and dataloaders.

        The training set is evenly split into ``n`` shards (the remaining
        ``len(train_set) % n`` samples are dropped).  Each worker receives a
        dedicated :class:`~torch.utils.data.DataLoader` with its own RNG
        generator, so mini-batch sampling is reproducible across runs.

        The learning rate is initialised to ``self.lr``. The
        Robbins-Monro schedule updates it each round inside
        :meth:`step`; the exponential schedule decays it after every
        round via ``self._current_lr *= lr_decay``.

        When ``xavier_init`` is enabled, every weight tensor of the
        instantiated model is re-initialized with the Glorot/Xavier
        uniform rule, and every bias is reset to zero.

        Safe to call multiple times — each call resets all internal state.
        """
        self._set_seed()
        self._model = Model(self.model_cls().to(self.device))
        if self.xavier_init:
            self._xavier_init_(self._model.module)
        self._current_lr = self.lr

        train_size = len(cast(Sized, self.train_set))
        shard_size = train_size // self.n
        shard_indices = torch.randperm(train_size, generator=torch.Generator().manual_seed(self.seed))

        self._worker_loaders = []
        for w in range(self.n):
            indices = shard_indices[w * shard_size : (w + 1) * shard_size]
            worker_ds = Subset(self.train_set, indices.tolist())
            worker_gen = torch.Generator().manual_seed(self.seed + w)
            self._worker_loaders.append(
                DataLoader(worker_ds, batch_size=self.batch_size, shuffle=True, generator=worker_gen)
            )

        self._full_loader = DataLoader(self.train_set, batch_size=len(cast(Sized, self.train_set)), shuffle=False)
        self._test_loader = DataLoader(self.test_set, batch_size=len(cast(Sized, self.test_set)), shuffle=False)
        self._has_run = False
        self._current_round = 0

    def step(self) -> None:
        r"""Advance the simulation by one synchronous round.

        #. The learning rate is updated to the value of the current round
           (only for the Robbins-Monro schedule; the exponential schedule
           updates the rate after the optimizer step instead).
        #. Each of the :math:`n - f` honest workers computes a gradient on its
           local data shard via :meth:`_train_one_worker`.
        #. If :math:`f > 0` and the attack has not been stopped, Byzantine
           workers generate attack gradients. For
            :class:`~krum.primitives.attacks.full_gradient_negation.FullGradientNegationAttack`,
           the full-dataset honest gradient is computed first.
        #. The aggregator combines all :math:`n` gradients into a single update
           via ``self.aggregator.aggregate(...)``.
        #. The aggregated gradient is written to
           ``self._model.gradients`` and applied via an in-place SGD
           update on the flat parameter tensor.
        #. The learning rate (if scheduled) is decayed.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
        """
        if self._model is None:
            raise RuntimeError("Simulation not set up. Call setup() first.")

        if self.lr_schedule == "robbins_monro":
            self._apply_robbins_monro_lr(self._current_round)

        num_honest = self.n - self.f
        worker_gradients: list[torch.Tensor] = []

        for w in range(num_honest):
            g = self._train_one_worker(self._worker_loaders[w])
            worker_gradients.append(g)

        if self.f > 0:
            attack_stopped = self.stop_attack_at is not None and self._current_round >= self.stop_attack_at
            if attack_stopped:
                d = worker_gradients[0].numel()
                zero_grad = torch.zeros(
                    d,
                    device=worker_gradients[0].device,
                    dtype=worker_gradients[0].dtype,
                )
                for _ in range(self.f):
                    worker_gradients.append(zero_grad.clone())
            else:
                attack_kw = dict(self._attack_kwargs)
                if issubclass(self.attack, FullGradientNegationAttack):
                    attack_kw["full_gradient"] = self._compute_full_gradient()

                honest_gradients = torch.stack(worker_gradients)
                byz_gradients = self.attack.generate(honest_gradients, f=self.f, **attack_kw)
                for g in byz_gradients:
                    worker_gradients.append(g)

        all_gradients = torch.stack(worker_gradients)
        aggregated = self.aggregator.aggregate(all_gradients, n=self.n, f=self.aggregator_f, **self._aggregator_kwargs)
        self._model.gradients = aggregated
        with torch.no_grad():
            params = self._model.parameters
            if self.weight_decay > 0:
                params.mul_(1 - self._current_lr * self.weight_decay)
            params.add_(self._model.gradients, alpha=-self._current_lr)
        if self.lr_schedule == "exponential":
            assert self.lr_decay is not None
            self._current_lr *= self.lr_decay

        self._current_round += 1

    def evaluate(self) -> Any:
        """Compute evaluation metrics by delegating to ``evaluate_fn``.

        The evaluator callable was supplied at construction time. Built-in
        options include :meth:`evaluate_test_error`,
        :meth:`evaluate_test_error_and_loss`, and :meth:`evaluate_full`.

        Returns:
            Scalar or tuple of evaluation metrics, as defined by the evaluator.
        """
        return self._evaluate_fn(self)

    def run(self) -> list[tuple[int, Any]]:
        """Run the simulation to completion.

        Calls :meth:`setup`, then loops over :meth:`step` and :meth:`evaluate`
        every ``eval_every`` rounds (always evaluating on the last round).

        Returns:
            List of ``(round, ...)`` tuples where each tail is the return value
            of :meth:`evaluate` (a scalar, tuple, or dict).

        Raises:
            RuntimeError: If :meth:`run` has already been called.
        """
        if self._has_run:
            raise RuntimeError(
                "run() has already been called on this instance. Create a new simulation for a fresh run."
            )
        self.setup()
        self._has_run = True

        traces: list[tuple[int, Any]] = []
        log_every = max(1, self.rounds // 20)
        for t in range(self.rounds):
            self.step()

            if t % log_every == 0 or t == self.rounds - 1:
                print(f"round {t + 1}/{self.rounds} done", flush=True)

            if t % self.eval_every == 0 or t == self.rounds - 1:
                result = self.evaluate()
                traces.append(_format_trace(t, result))
                if t % log_every == 0 or t == self.rounds - 1:
                    print(f"  {result}", flush=True)

        return traces

    def _evaluate_batch(self, loader: DataLoader[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one full-batch inference pass and return logits and labels.

        Args:
            loader: DataLoader with ``batch_size == len(dataset)``.

        Returns:
            Tuple of (logits, labels) on :attr:`device`.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
        """
        if self._model is None:
            raise RuntimeError("Simulation not set up. Call setup() first.")
        self._model.module.eval()
        with torch.no_grad():
            x, y = next(iter(loader))
            x, y = x.to(self.device), y.to(self.device)
            logits = self._model.module(x)
        return logits, y

    def evaluate_test_error(self) -> float:
        """Compute misclassification error rate on the held-out test set.

        Returns:
            Error rate in ``[0, 1]``.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
        """
        if self._test_loader is None:
            raise RuntimeError("Simulation not set up. Call setup() first.")
        logits, y = self._evaluate_batch(self._test_loader)
        preds = logits.argmax(dim=1)
        error = (preds != y).float().mean()
        return error.item()

    def evaluate_test_error_and_loss(self) -> dict[str, float]:
        """Compute misclassification error rate and cross-entropy loss on the test set.

        Returns:
            Dict with ``"test_error"`` and ``"test_loss"``.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
        """
        if self._test_loader is None:
            raise RuntimeError("Simulation not set up. Call setup() first.")
        logits, y = self._evaluate_batch(self._test_loader)
        preds = logits.argmax(dim=1)
        error = (preds != y).float().mean().item()
        loss = self.loss_fn(logits, y).item()
        return {"test_error": error, "test_loss": loss}

    def evaluate_full(self) -> dict[str, float]:
        """Compute training loss and test accuracy/loss on full datasets.

        Returns:
            Dict with ``"train_loss"``, ``"test_accuracy"``, ``"test_loss"``.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
        """
        if self._full_loader is None or self._test_loader is None:
            raise RuntimeError("Simulation not set up. Call setup() first.")

        logits_train, y_train = self._evaluate_batch(self._full_loader)
        train_loss = self.loss_fn(logits_train, y_train).item()

        logits_test, y_test = self._evaluate_batch(self._test_loader)
        test_loss = self.loss_fn(logits_test, y_test).item()
        preds = logits_test.argmax(dim=1)
        test_acc = (preds == y_test).float().mean().item()

        return {"train_loss": train_loss, "test_accuracy": test_acc, "test_loss": test_loss}

    @staticmethod
    def _detect_device() -> torch.device:
        """Detect the best available torch device.

        Returns:
            CUDA if available, otherwise MPS, otherwise CPU.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _set_seed(self) -> None:
        """Set all RNG seeds for reproducible runs (PyTorch, CUDA, MPS)."""
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        elif self.device.type == "mps":
            torch.mps.manual_seed(self.seed)

    def _train_one_worker(self, loader: DataLoader[Any]) -> torch.Tensor:
        r"""Compute the gradient on one worker's data shard.

        In the distributed-SGD protocol reproduced by the papers, each worker
        performs **a single mini-batch SGD step per round**. The worker calls
        ``loss.backward()`` on one mini-batch drawn from its dedicated
        :class:`~torch.utils.data.DataLoader`, then clones the flat gradient
        tensor from the :class:`~krum.primitives.model.Model`.

        Args:
            loader: DataLoader yielding mini-batches from the worker's IID shard.

        Returns:
            Cloned flat gradient tensor of shape ``(d,)``.
        """
        assert self._model is not None
        self._model.module.train()
        x, y = next(iter(loader))
        x, y = x.to(self.device), y.to(self.device)
        self._model.module.zero_grad()
        loss = self.loss_fn(self._model.module(x), y)
        loss.backward()
        return self._model.gradients.clone()

    def _apply_robbins_monro_lr(self, t: int) -> None:
        """Set the learning rate to the Robbins-Monro value for round ``t``.

        The Robbins-Monro schedule of El Mhamdi et al. (ICML 2018, Section
        5.1) is :math:`η(t) = r_η · η_0 / (t + r_η)`. The computed rate
        is stored in ``self._current_lr``, which is then used by the flat
        SGD update in :meth:`step`.

        Args:
            t: Current round index (0-based).
        """
        assert self.r_eta is not None
        self._current_lr = self.r_eta * self.lr / (t + self.r_eta)

    @staticmethod
    def _xavier_init_(module: nn.Module) -> None:
        """Re-initialize every weight tensor of ``module`` with Xavier-uniform.

        Biases are reset to zero. Convolutional and linear layers are
        the only ones re-initialized: scalar/vector buffers (e.g.
        ``BatchNorm`` running statistics) are left untouched.

        Args:
            module: The module to re-initialize in place.
        """
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _compute_full_gradient(self) -> torch.Tensor:
        r"""Compute the full-dataset honest gradient for the full-gradient negation attack.

        Called automatically by :meth:`step` before Byzantine gradient
        generation when ``self.attack`` is an
        :class:`~krum.primitives.attacks.full_gradient_negation.FullGradientNegationAttack`.

        Returns:
            Full-dataset gradient of shape ``(d,)``.
        """
        assert self._model is not None and self._full_loader is not None
        x, y = next(iter(self._full_loader))
        x, y = x.to(self.device), y.to(self.device)
        self._model.module.zero_grad()
        loss = self.loss_fn(self._model.module(x), y)
        loss.backward()
        return self._model.gradients.clone()


def _format_trace(round: int, result: Any) -> tuple[int, Any]:
    """Pack a round number and an evaluation result into a trace tuple.

    If *result* is a dict it is appended directly (unchanged)::

        _format_trace(3, {"train_loss": 0.1}) → (3, {"train_loss": 0.1})

    If *result* is a tuple it is unpacked::

        _format_trace(3, (0.1, 0.9)) → (3, 0.1, 0.9)

    Scalars are appended directly::

        _format_trace(3, 0.05) → (3, 0.05)

    Args:
        round: Current round index (0-based).
        result: The value returned by :meth:`CentralisedSimulation.evaluate`
            (i.e. by the ``evaluate_fn`` callable).

    Returns:
        A trace tuple ``(round, ...)`` suitable for insertion into the trace list.
    """
    if isinstance(result, tuple):
        return (round, *result)
    return (round, result)
