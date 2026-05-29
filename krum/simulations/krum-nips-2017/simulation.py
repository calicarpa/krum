"""KrumSimulation — encapsulated distributed SGD simulation with Byzantine workers.

One ``KrumSimulation`` instance = one configuration (aggregator, attack, dataset,
model) run over multiple synchronous rounds following the parameter-server
architecture from Blanchard et al., NIPS 2017.
"""

from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from krum.primitives.aggregators import Aggregator
from krum.primitives.attacks import Attack, OmniscientAttack
from krum.primitives.model import Model


class KrumSimulation:
    """Distributed SGD simulation with Byzantine workers.

    Encapsulates the full lifecycle of one experiment: setup (model, optimizer,
    data sharding), training (honest + Byzantine gradients, aggregation, SGD
    step), and evaluation on a held-out test set.

    One instance = one (aggregator, attack, dataset, model) configuration.

    Args:
        model_cls: ``nn.Module`` subclass to instantiate for training.
        train_set: Full training dataset (will be IID-sharded across workers).
        test_set: Test dataset (used for evaluation via full-batch loader).
        aggregator: Gradient aggregation rule (e.g. ``Average``, ``Krum``).
        attack: Byzantine attack strategy (e.g. ``GaussianAttack``).
        n: Total number of workers. Must match ``aggregator.n``.
        f: Number of Byzantine workers. Must match ``aggregator.f``.
        rounds: Number of synchronous rounds.
        batch_size: Mini-batch size per honest worker.
        lr: Learning rate for SGD.
        loss_fn: Loss function (default: ``cross_entropy``).
        device: Device for training and evaluation. Auto-detected if ``None``.
        seed: Random seed for reproducibility.
        eval_every: Evaluate on the test set every ``eval_every`` rounds.
        label: Human-readable name for logging.
        results_dir: Directory to save per-run traces. If ``None``, results
            are not saved to disk.

    Raises:
        ValueError: If ``n``/``f`` do not match the aggregator's values.
    """

    def __init__(
        self,
        *,
        model_cls: type[nn.Module],
        train_set: Dataset[Any],
        test_set: Dataset[Any],
        aggregator: Aggregator,
        attack: Attack,
        n: int,
        f: int,
        rounds: int,
        batch_size: int,
        lr: float,
        loss_fn: Callable[..., torch.Tensor] = nn.functional.cross_entropy,
        device: torch.device | None = None,
        seed: int = 42,
        eval_every: int = 10,
        label: str = "",
        results_dir: Path | str | None = None,
    ) -> None:
        """Initialize the simulation with the given parameters."""
        if hasattr(aggregator, "n") and n != aggregator.n:
            raise ValueError(f"n={n} does not match aggregator.n={aggregator.n}")
        if hasattr(aggregator, "f") and f != aggregator.f:
            raise ValueError(f"f={f} does not match aggregator.f={aggregator.f}")

        self.model_cls = model_cls
        self.train_set = train_set
        self.test_set = test_set
        self.aggregator = aggregator
        self.attack = attack
        self.n = n
        self.f = f
        self.rounds = rounds
        self.batch_size = batch_size
        self.lr = lr
        self.loss_fn = loss_fn
        self.device = device or self._detect_device()
        self.seed = seed
        self.eval_every = eval_every
        self.label = label
        self.results_dir = Path(results_dir) if results_dir is not None else None

        self._model: Model | None = None
        self._opt: torch.optim.Optimizer | None = None
        self._worker_loaders: list[DataLoader[Any]] = []
        self._full_loader: DataLoader[Any] | None = None
        self._test_loader: DataLoader[Any] | None = None
        self._has_run = False

    @property
    def model(self) -> Model:
        """The encapsulated ``Model``, available after ``setup()`` or ``run()``.

        Returns:
            The wrapped ``nn.Module`` with flat parameter/gradient views.

        Raises:
            RuntimeError: If the simulation has not been set up yet.
        """
        if self._model is None:
            raise RuntimeError("Simulation not set up. Call setup() or run() first.")
        return self._model

    def setup(self) -> None:
        """Initialize the model, optimizer, worker dataloaders, and RNG.

        Must be called before ``step()`` or ``evaluate()``. Called
        automatically by ``run()``. Idempotent: can be called multiple
        times to reset state.
        """
        self._set_seed()
        self._model = Model(self.model_cls().to(self.device))
        self._opt = torch.optim.SGD(self._model.module.parameters(), lr=self.lr)

        train_size = len(self.train_set)
        # The last ``train_size % n`` samples may be dropped when the
        # dataset size is not evenly divisible by the number of workers.
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
        """Perform one synchronous round of the distributed SGD simulation.

        1. Each honest worker computes a gradient on its local data shard.
        2. Byzantine workers generate attack gradients.
        3. The aggregator combines all gradients into a single update.
        4. The aggregated gradient is applied via SGD step.

        Raises:
            RuntimeError: If ``setup()`` has not been called.
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
        aggregated = self.aggregator(all_gradients)
        self._model.gradients = aggregated
        self._opt.step()

    def evaluate(self) -> float:
        """Compute misclassification error rate on the test set.

        Returns:
            Error rate in ``[0, 1]``.

        Raises:
            RuntimeError: If ``setup()`` has not been called.
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

    def run(self) -> list[tuple[int, float]]:
        """Run the full simulation for ``self.rounds`` rounds.

        Calls ``setup()``, then loops over ``step()`` and ``evaluate()``
        every ``eval_every`` rounds. Saves per-run traces to
        ``results_dir`` if configured.

        Returns:
            List of ``(round, error)`` pairs recorded during training.

        Raises:
            RuntimeError: If ``run()`` has already been called on this instance.
        """
        if self._has_run:
            raise RuntimeError(
                "run() has already been called on this instance. Create a new KrumSimulation for a fresh run."
            )
        self.setup()
        self._has_run = True

        errors: list[tuple[int, float]] = []
        for t in range(self.rounds):
            self.step()

            if t % self.eval_every == 0 or t == self.rounds - 1:
                error = self.evaluate()
                errors.append((t, error))
                if self.label:
                    print(f"[{self.label}] round {t:3d}  error={error:.4f}")

        if self.results_dir is not None:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            path = self.results_dir / f"{self.label}.pt"
            torch.save({"errors": errors, "label": self.label, "seed": self.seed}, path)

        return errors

    @staticmethod
    def _detect_device() -> torch.device:
        """Detect the best available device.

        Returns:
            CUDA, MPS, or CPU device.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _set_seed(self) -> None:
        """Set all random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        elif self.device.type == "mps":
            torch.mps.manual_seed(self.seed)

    def _train_one_worker(self, loader: DataLoader[Any]) -> torch.Tensor:
        """Compute gradients on one worker's data shard.

        Args:
            loader: DataLoader yielding mini-batches from the worker's shard.

        Returns:
            Flat gradient tensor of shape ``(d,)``, cloned from the model.
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
        """Compute the full-dataset gradient and feed it to an omniscient attack."""
        if not isinstance(self.attack, OmniscientAttack):
            raise TypeError(f"Expected OmniscientAttack, got {type(self.attack).__name__}")
        assert self._model is not None and self._opt is not None and self._full_loader is not None
        x, y = next(iter(self._full_loader))
        x, y = x.to(self.device), y.to(self.device)
        self._opt.zero_grad()
        loss = self.loss_fn(self._model.module(x), y)
        loss.backward()
        self.attack.set_full_gradient(self._model.gradients.clone())
