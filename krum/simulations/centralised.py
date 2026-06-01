"""Parameter-server distributed SGD simulation confronting Byzantine workers.

Each synchronous round:
    #. Honest workers compute a gradient on their local data shard.
    #. Byzantine workers craft adversarial gradients.
    #. The aggregator combines all ``n`` gradients into a single update.
    #. The aggregated update is applied via an SGD step.

The :class:`CentralisedSimulation` base class implements the full lifecycle
(model initialisation, data sharding, training loop, evaluation, persistence).
Protocol-specific subclasses — :class:`~krum.simulations.krum-nips-2017.simulation.KrumSimulation`
(Blanchard et al., NIPS 2017) and :class:`~krum.simulations.hidden-vulnerability-icml-2018.simulation.Simulation`
(El Mhamdi et al., ICML 2018) — override only :meth:`~CentralisedSimulation.evaluate`
to provide their own metric reporting.
"""

import csv
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset, Subset

from krum.primitives.aggregators import Aggregator
from krum.primitives.attacks import Attack, OmniscientAttack
from krum.primitives.model import Model


class CentralisedSimulation:
    """Parameter-server distributed SGD simulation with Byzantine workers.

    One instance = one (aggregator, attack, dataset, model) configuration run
    over ``rounds`` synchronous rounds. The training set is IID-sharded across
    ``n`` workers, of which ``f`` are Byzantine (up to the tolerance of the
    chosen aggregator).

    Subclasses override :meth:`evaluate` to customise the evaluation protocol
    (e.g. NIPS 2017 reports a single error rate, ICML 2018 reports three
    metrics).  The hooks :meth:`_log_round` and :meth:`_save_traces` control
    per-round printing and persistence respectively.

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
        attack: Byzantine attack strategy (e.g. ``GaussianAttack``).
            If the attack is an :class:`~krum.primitives.attacks.omniscient.OmniscientAttack`,
            the full-dataset gradient is computed and set before each attack
            generation.
        n: Total number of workers.
        f: Number of Byzantine workers. Must be within the aggregator's
            Byzantine tolerance.
        rounds: Number of synchronous training rounds.
        batch_size: Mini-batch size per honest worker.
        lr: Initial learning rate for SGD.
        lr_decay: Multiplicative learning rate decay per round. ``None``
            disables the scheduler. Default: ``None``.
        loss_fn: Per-sample loss function. Default: ``cross_entropy``.
        device: Device for training and evaluation. Auto-detected if ``None``
            (CUDA → MPS → CPU).
        seed: Random seed for reproducibility.
        eval_every: Evaluate on the test set every ``eval_every`` rounds
            (and always on the last round).
        label: Human-readable name for logging and trace filenames.
        results_dir: Directory to save per-run traces (``.pt`` and optionally
            ``.csv``).  If ``None``, results are not saved to disk.

    Raises:
        RuntimeError: If :meth:`run` is called more than once on the same
            instance.
    """

    def __init__(
        self,
        *,
        model_cls: type[nn.Module],
        train_set: Dataset[Any],
        test_set: Dataset[Any],
        aggregator: type[Aggregator],
        aggregator_kwargs: dict[str, Any] | None = None,
        attack: Attack,
        n: int,
        f: int,
        rounds: int,
        batch_size: int,
        lr: float,
        lr_decay: float | None = None,
        loss_fn: Callable[..., torch.Tensor] = nn.functional.cross_entropy,
        device: torch.device | None = None,
        seed: int = 42,
        eval_every: int = 10,
        label: str = "",
        results_dir: Path | str | None = None,
    ) -> None:
        """See the class docstring for the full parameter list."""
        self.model_cls = model_cls
        self.train_set = train_set
        self.test_set = test_set
        self.aggregator = aggregator
        self._aggregator_kwargs = aggregator_kwargs or {}
        self.attack = attack
        self.n = n
        self.f = f
        self.rounds = rounds
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.loss_fn = loss_fn
        self.device = device or self._detect_device()
        self.seed = seed
        self.eval_every = eval_every
        self.label = label
        self.results_dir = Path(results_dir) if results_dir is not None else None

        self._model: Model | None = None
        self._opt: torch.optim.Optimizer | None = None
        self._scheduler: ExponentialLR | None = None
        self._worker_loaders: list[DataLoader[Any]] = []
        self._full_loader: DataLoader[Any] | None = None
        self._test_loader: DataLoader[Any] | None = None
        self._has_run = False

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
        """Initialise the model, SGD optimizer, IID data shards, and dataloaders.

        The training set is evenly split into ``n`` shards (the remaining
        ``len(train_set) % n`` samples are dropped).  Each worker receives a
        dedicated :class:`~torch.utils.data.DataLoader` with its own RNG
        generator, so mini-batch sampling is reproducible across runs.

        An :class:`~torch.optim.lr_scheduler.ExponentialLR` scheduler is
        created when ``lr_decay`` is set.

        Safe to call multiple times — each call resets all internal state.
        """
        self._set_seed()
        self._model = Model(self.model_cls().to(self.device))
        self._opt = torch.optim.SGD(self._model.module.parameters(), lr=self.lr)
        if self.lr_decay is not None:
            self._scheduler = ExponentialLR(self._opt, gamma=self.lr_decay)

        train_size = len(self.train_set)
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

        self._full_loader = DataLoader(self.train_set, batch_size=len(self.train_set), shuffle=False)
        self._test_loader = DataLoader(self.test_set, batch_size=len(self.test_set), shuffle=False)
        self._has_run = False

    def step(self) -> None:
        """Advance the simulation by one synchronous round.

        #. Each of the ``n - f`` honest workers computes a gradient on its
           local data shard via :meth:`_train_one_worker`.
        #. If :math:`f > 0`, Byzantine workers generate attack gradients.
           For :class:`~krum.primitives.attacks.omniscient.OmniscientAttack`,
           the full-dataset honest gradient is computed first.
        #. The aggregator combines all ``n`` gradients into a single update
           via ``self.aggregator.aggregate(...)``.
        #. The aggregated gradient is written to
           ``self._model.gradients`` and the optimizer takes an SGD step.
        #. The learning rate scheduler (if enabled) is stepped.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
        """
        if self._model is None or self._opt is None:
            raise RuntimeError("Simulation not set up. Call setup() first.")

        num_honest = self.n - self.f
        worker_gradients: list[torch.Tensor] = []

        for w in range(num_honest):
            g = self._train_one_worker(self._worker_loaders[w])
            worker_gradients.append(g)

        if self.f > 0:
            if isinstance(self.attack, OmniscientAttack):
                self._set_full_gradient_for_attack()

            honest_gradients = torch.stack(worker_gradients)
            byz_gradients = self.attack.generate(honest_gradients, self.f)
            for g in byz_gradients:
                worker_gradients.append(g)

        all_gradients = torch.stack(worker_gradients)
        aggregated = self.aggregator.aggregate(all_gradients, n=self.n, f=self.f, **self._aggregator_kwargs)
        self._model.gradients = aggregated
        self._opt.step()
        if self._scheduler is not None:
            self._scheduler.step()

    def evaluate(self) -> Any:
        """Compute evaluation metrics after a training round.

        Subclasses must override this to return protocol-specific metrics
        (e.g. a single error rate for NIPS 2017, or
        ``(train_loss, test_accuracy, test_loss)`` for ICML 2018).

        Returns:
            Scalar or tuple of evaluation metrics.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError

    def run(self) -> list[tuple[int, Any]]:
        """Run the simulation to completion.

        Calls :meth:`setup`, then loops over :meth:`step` and :meth:`evaluate`
        every ``eval_every`` rounds (always evaluating on the last round).
        On each evaluation, :meth:`_log_round` prints progress and
        :meth:`_save_traces` persists results to ``results_dir`` when
        configured.

        Returns:
            List of ``(round, ...)`` tuples. The tail is the output of
            :meth:`evaluate`: a scalar value appended directly (e.g.
            ``(42, 0.05)``) or a tuple unpacked (e.g.
            ``(42, 0.1, 0.95, 0.2)``).

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
        for t in range(self.rounds):
            self.step()

            if t % self.eval_every == 0 or t == self.rounds - 1:
                result = self.evaluate()
                self._log_round(t, result)
                traces.append(_pack(t, result))

        self._save_traces(traces)
        return traces

    def _log_round(self, t: int, result: Any) -> None:
        """Log evaluation metrics for a round. Override in subclass.

        Args:
            t: Current round index (0-based).
            result: The value returned by :meth:`evaluate`.
        """

    def _save_traces(self, traces: list[tuple[int, Any]]) -> None:
        """Save traces to disk. Override in subclass.

        Args:
            traces: List of ``(round, ...)`` tuples as produced by :meth:`run`.
        """

    def evaluate_test_error(self) -> float:
        """Compute misclassification error rate on the held-out test set.

        Returns:
            Error rate in :math:`[0, 1]`.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
        """
        if self._model is None or self._test_loader is None:
            raise RuntimeError("Simulation not set up. Call setup() first.")

        self._model.module.eval()
        with torch.no_grad():
            x, y = next(iter(self._test_loader))
            x, y = x.to(self.device), y.to(self.device)
            logits = self._model.module(x)
            preds = logits.argmax(dim=1)
            error = (preds != y).float().mean()
        return error.item()

    def evaluate_full(self) -> tuple[float, float, float]:
        """Compute training loss and test accuracy/loss on full datasets.

        Returns:
            Tuple of ``(train_loss, test_accuracy, test_loss)``.
            Accuracy is the fraction of correct predictions in :math:`[0, 1]`.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
        """
        if self._model is None or self._full_loader is None or self._test_loader is None:
            raise RuntimeError("Simulation not set up. Call setup() first.")

        self._model.module.eval()
        with torch.no_grad():
            x_train, y_train = next(iter(self._full_loader))
            x_train, y_train = x_train.to(self.device), y_train.to(self.device)
            logits_train = self._model.module(x_train)
            train_loss = self.loss_fn(logits_train, y_train).item()

            x_test, y_test = next(iter(self._test_loader))
            x_test, y_test = x_test.to(self.device), y_test.to(self.device)
            logits_test = self._model.module(x_test)
            test_loss = self.loss_fn(logits_test, y_test).item()
            preds = logits_test.argmax(dim=1)
            test_acc = (preds == y_test).float().mean().item()

        return (train_loss, test_acc, test_loss)

    def _save_pt(self, data: dict[str, Any]) -> None:
        """Persist per-run traces as a ``.pt`` file.

        Args:
            data: Dictionary to save via :func:`torch.save`. Must be a
                serialisable ``dict`` (e.g. ``{"errors": traces, "label": label}``).
        """
        if self.results_dir is None:
            return
        self.results_dir.mkdir(parents=True, exist_ok=True)
        torch.save(data, self.results_dir / f"{self.label}.pt")

    def _save_csv(self, traces: list[tuple[int, float, float, float]]) -> None:
        """Persist per-run traces as a ``.csv`` file.

        Columns: ``round, train_loss, test_accuracy, test_loss``.

        Args:
            traces: List of ``(round, train_loss, test_accuracy, test_loss)`` tuples.
        """
        if self.results_dir is None:
            return
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_dir / f"{self.label}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "train_loss", "test_accuracy", "test_loss"])
            for t, tr_loss, acc, te_loss in traces:
                writer.writerow([t, tr_loss, acc, te_loss])

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
        """Compute the gradient on one worker's data shard.

        The worker calls ``loss.backward()`` on a single mini-batch drawn from
        its dedicated :class:`~torch.utils.data.DataLoader`, then clones the
        flat gradient tensor from the :class:`~krum.primitives.model.Model`.

        Args:
            loader: DataLoader yielding mini-batches from the worker's IID shard.

        Returns:
            Cloned flat gradient tensor of shape ``(d,)``.
        """
        assert self._model is not None and self._opt is not None
        self._model.module.train()
        x, y = next(iter(loader))
        x, y = x.to(self.device), y.to(self.device)
        self._opt.zero_grad()
        loss = self.loss_fn(self._model.module(x), y)
        loss.backward()
        return self._model.gradients.clone()

    def _set_full_gradient_for_attack(self) -> None:
        """Compute the full-dataset honest gradient and pass it to an omniscient attack.

        Called automatically by :meth:`step` before Byzantine gradient
        generation when ``self.attack`` is an
        :class:`~krum.primitives.attacks.omniscient.OmniscientAttack`.

        Raises:
            TypeError: If ``self.attack`` is not an :class:`OmniscientAttack`.
        """
        if not isinstance(self.attack, OmniscientAttack):
            raise TypeError(f"Expected OmniscientAttack, got {type(self.attack).__name__}")
        assert self._model is not None and self._opt is not None and self._full_loader is not None
        x, y = next(iter(self._full_loader))
        x, y = x.to(self.device), y.to(self.device)
        self._opt.zero_grad()
        loss = self.loss_fn(self._model.module(x), y)
        loss.backward()
        self.attack.set_full_gradient(self._model.gradients.clone())


def _pack(round: int, result: Any) -> tuple[int, Any]:
    """Pack a round number and an evaluation result into a trace tuple.

    If *result* is already a tuple it is unpacked::

        _pack(3, (0.1, 0.9)) → (3, 0.1, 0.9)

    Scalars are appended directly::

        _pack(3, 0.05) → (3, 0.05)

    Args:
        round: Current round index (0-based).
        result: The value returned by :meth:`CentralisedSimulation.evaluate`.

    Returns:
        A trace tuple ``(round, ...)`` suitable for insertion into the trace list.
    """
    if isinstance(result, tuple):
        return (round, *result)
    return (round, result)
