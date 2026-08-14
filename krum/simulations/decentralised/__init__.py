"""Peer-to-peer decentralised learning simulation confronting Byzantine workers.

Each honest worker holds its own model. Each round runs two phases:

#. **Local optimisation** — each worker computes a gradient on its local batch
   and updates its own model.
#. **Model mixing** — each worker gathers models from other nodes and replaces
   its model with an aggregate of the received set.

Two abstract seams let protocols vary the local update rule (e.g. momentum-SGD)
and the communication topology (which models each worker receives).
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Generic, Sized, TypedDict, TypeVar, cast

import torch
from torch.utils.data import DataLoader, Dataset

from ...primitives.aggregators import Aggregator
from ...primitives.attacks import Attack
from ...primitives.models import Model

Batch = tuple[torch.Tensor, torch.Tensor]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class StepResult(TypedDict):
    """Snapshot returned by :meth:`DecentralisedSimulation.step` for one round.

    Holds only the fields common to every decentralised protocol. Subclasses
    whose local step keeps extra state (e.g. momentum) extend this TypedDict and
    bind it as the simulation's ``StepResultT``.
    """

    step: int
    parameters: torch.Tensor
    honest_gradients: torch.Tensor
    local_parameters: torch.Tensor
    byzantine_parameters: torch.Tensor
    mixed_parameters: torch.Tensor
    losses: torch.Tensor


StepResultT = TypeVar("StepResultT", bound=StepResult)


class DecentralisedSimulation(ABC, Generic[StepResultT]):
    """Base for decentralised simulations with per-worker model mixing.

    Each honest worker holds its own flat parameter vector — one row of
    :attr:`parameters`. A round runs a local optimisation step, then a mixing
    phase in which every worker replaces its model with an aggregate of the
    models it *received* this round.

    Two things vary between protocols, and each is an abstract seam:

    * **how a worker updates locally** — :meth:`local_update` maps the worker
      gradients to the post-local parameters ``theta_{t+1/2}``. The base is
      agnostic to the rule and its state, so a momentum-free or optimiser-based
      protocol keeps none of MoNNA's momentum machinery.
    * **which models a worker receives** — :meth:`gather_received_models` builds
      each worker's received set (the communication topology).

    The base owns everything else: gradient computation, the Byzantine
    generation hook, the mixing loop, the generic state commit, and the
    multi-round :meth:`run` driver. Subclasses also implement
    :meth:`build_step_result` so each protocol's snapshot can carry its own
    fields (the base ``StepResult`` plus, e.g., momentum).

    :meth:`run` may be called repeatedly to continue training: all state
    (:attr:`parameters`, :attr:`step_index`, subclass optimiser state, and the
    worker dataloaders) lives on the instance and persists across calls. Each
    worker's :class:`~torch.utils.data.DataLoader` is automatically
    re-iterated (a fresh epoch) once exhausted, so :meth:`run` may be called
    for more rounds than one epoch provides.

    Evaluation is defined by subclasses overriding :meth:`evaluate` — it is
    never called from :meth:`step`, so a caller decides when a test-set
    snapshot is worth its cost (e.g. every few rounds).
    """

    def __init__(
        self,
        *,
        model: Model,
        train_datasets: Sequence[Dataset[Any]],
        train_batch_size: int,
        test_set: Dataset[Any],
        test_batch_size: int,
        loss_fn: LossFn,
        n: int,
        f: int,
        attack: type[Attack] | None = None,
        attack_kwargs: dict[str, Any] | None = None,
        aggregator: type[Aggregator],
        aggregator_kwargs: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the shared decentralised state.

        Args:
            model: Model wrapper whose flat parameters seed every worker.
            train_datasets: One dataset per worker (honest and Byzantine);
                ``len(train_datasets)`` must equal ``n``. Only the first
                ``n - f`` (the honest workers) are ever wrapped into a
                ``DataLoader`` and trained on — Byzantine workers craft their
                models from the honest models via ``attack``, not from local
                data — but the full ``n``-length sequence is required so a
                future data-consuming attack (e.g. label-flipping) has
                something to read. Each honest worker's loader is
                automatically re-iterated (a fresh epoch) once exhausted, so
                it need not provide as many batches as :meth:`run` will
                request rounds.
            train_batch_size: Mini-batch size for every honest worker's
                ``DataLoader``.
            test_set: Shared test dataset, evaluated per-worker by
                :meth:`evaluate`.
            test_batch_size: Mini-batch size for the test ``DataLoader``.
            loss_fn: Callable mapping ``(predictions, targets)`` to a scalar loss.
            n: Total number of workers; must exceed ``2 * f`` so that honest
                workers outnumber Byzantine ones.
            f: Number of Byzantine workers; must be non-negative.
            attack: :class:`~krum.primitives.attacks.Attack` subclass whose
                ``generate`` classmethod maps the honest models and ``f`` to the
                Byzantine models. Required when ``f > 0``.
            attack_kwargs: Extra keyword arguments forwarded to
                ``attack.generate`` (e.g. ``{"std": 200.0}``). ``f`` is injected
                by the simulation.
            aggregator: :class:`~krum.primitives.aggregators.Aggregator` subclass
                whose ``aggregate`` classmethod mixes each worker's received
                models. Subclasses typically resolve a protocol default before
                passing it here.
            aggregator_kwargs: Extra keyword arguments forwarded to
                ``aggregator.aggregate`` (e.g. ``{"num_closest": ...}``). The
                ``pivot`` is injected per worker.
            seed: Optional integer seed for responder sampling and per-worker
                ``DataLoader`` shuffling (worker ``w``'s loader is seeded with
                ``seed + w``). ``None`` disables both.

        Raises:
            ValueError: If a worker count or dataset count is out of range,
                any honest worker's dataset is empty, or an attack is
                missing while ``f > 0``.
            TypeError: If ``model`` or ``loss_fn`` has the wrong type, ``attack``
                is not an :class:`~krum.primitives.attacks.Attack` subclass,
                ``aggregator`` is not an
                :class:`~krum.primitives.aggregators.Aggregator` subclass, or
                ``seed`` is not an int or ``None``.
        """
        if f < 0:
            raise ValueError(f"Expected non-negative Byzantine worker count, got f={f!r}")
        if n - f < 1:
            raise ValueError(f"Expected at least one honest worker, got n={n!r} and f={f!r}")
        if n - f <= f:
            raise ValueError(f"Expected enough honest workers for decentralised model mixing, got n={n!r} and f={f!r}")
        if not isinstance(model, Model):
            raise TypeError(f"Expected model to be a Model, got {type(model).__name__}")
        if not callable(loss_fn):
            raise TypeError("Expected loss_fn to be callable")
        if len(train_datasets) != n:
            raise ValueError(f"Expected {n} train datasets, got {len(train_datasets)!r}")
        for w in range(n - f):
            if len(cast(Sized, train_datasets[w])) == 0:
                raise ValueError(
                    f"Worker {w} has an empty train_dataset (0 samples) and cannot train; "
                    f"check n against the partitioner's configuration (e.g. alpha)."
                )
        if f and attack is None:
            raise ValueError("An attack is required when f > 0")
        if attack is not None and not (isinstance(attack, type) and issubclass(attack, Attack)):
            raise TypeError("Expected attack to be an Attack subclass")
        if not (isinstance(aggregator, type) and issubclass(aggregator, Aggregator)):
            raise TypeError("Expected aggregator to be an Aggregator subclass")
        if seed is not None and not isinstance(seed, int):
            raise TypeError(f"Expected seed to be an int or None, got {type(seed).__name__}")

        self.model = model
        self.train_datasets: list[Dataset[Any]] = list(train_datasets)
        self.train_batch_size = train_batch_size
        self.test_set = test_set
        self.test_batch_size = test_batch_size
        self.loss_fn = loss_fn
        self.n = n
        self.f = f
        self.num_honest = n - f
        self.attack = attack
        self.attack_kwargs = attack_kwargs or {}
        self.aggregator = aggregator
        self.aggregator_kwargs = dict(aggregator_kwargs or {})
        self.parameters = model.parameters.detach().clone().repeat(self.num_honest, 1)
        self.generator = None if seed is None else torch.Generator(device=self.parameters.device).manual_seed(seed)
        self.step_index = 0

        self.worker_data: list[DataLoader[Any]] = []
        for w in range(self.num_honest):
            worker_dataset = self.train_datasets[w]
            worker_generator = None if seed is None else torch.Generator().manual_seed(seed + w)
            self.worker_data.append(
                DataLoader(worker_dataset, batch_size=train_batch_size, shuffle=True, generator=worker_generator)
            )
        self.worker_data_iterators = [iter(loader) for loader in self.worker_data]
        self.test_loader: DataLoader[Any] = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)

    def step(self) -> StepResultT:
        """Execute one decentralised training round.

        Runs one local optimisation phase (:meth:`local_update`) followed by one
        model-mixing phase over the per-worker received sets, then commits the
        resulting state and builds the snapshot.

        Returns:
            A snapshot dict of the round, as built by :meth:`build_step_result`.
        """
        batches = self.collect_worker_batches()
        gradients, losses = self.compute_honest_worker_gradients(batches)
        local_parameters = self.local_update(gradients)
        byzantine_parameters = self.generate_byzantine_models(local_parameters)
        mixed_parameters = self.aggregate_over_received_nodes(local_parameters, byzantine_parameters)
        self.commit_state(mixed_parameters)
        return self.build_step_result(
            honest_gradients=gradients,
            local_parameters=local_parameters,
            byzantine_parameters=byzantine_parameters,
            mixed_parameters=mixed_parameters,
            losses=losses,
        )

    def run(self, rounds: int) -> list[StepResultT]:
        """Execute several rounds and collect their snapshots.

        State persists on the instance, so successive calls continue training
        from where the previous call left off.

        Args:
            rounds: Number of rounds to run; must be non-negative.

        Returns:
            One snapshot per round, in execution order.

        Raises:
            ValueError: If ``rounds`` is negative.
        """
        if rounds < 0:
            raise ValueError(f"Expected non-negative rounds, got {rounds!r}")
        return [self.step() for _ in range(rounds)]

    @abstractmethod
    def evaluate(self) -> tuple[float, ...]:
        """Evaluate every honest worker's current local model on the test set.

        This is the evaluation seam: each protocol reports its own set of
        metrics, the way :class:`~krum.simulations.centralised.CentralisedSimulation`
        subclasses do. Never called automatically — invoke it explicitly
        whenever a test-set snapshot is wanted (e.g. every few rounds), since
        it is typically far more expensive than one :meth:`step`.

        Implementations typically loop over :attr:`parameters` (one row per
        honest worker), load each into :attr:`model` via
        :meth:`copy_parameters_to_model`, evaluate against :attr:`test_loader`,
        and average the per-worker metrics.

        Returns:
            The protocol's test metrics, averaged across honest workers.
        """

    def collect_worker_batches(self) -> list[Batch]:
        """Pull one local batch from every honest worker's ``DataLoader``.

        When a worker's loader is exhausted (its current epoch ends), a
        fresh iterator is created automatically and the first batch of the
        new epoch is used instead, so callers do not need to pre-cycle
        their loaders to run more rounds than one epoch provides.

        Returns:
            One batch per honest worker, in worker order.
        """
        batches = []
        for worker_index, iterator in enumerate(self.worker_data_iterators):
            try:
                batch = next(iterator)
            except StopIteration:
                # __init__ already rejects any honest worker with an empty
                # train_dataset, so a StopIteration here can only mean "this
                # epoch just ended" -- never "this loader was never going to
                # yield anything." Without that guarantee, the retry below
                # would raise StopIteration again immediately and propagate
                # uncaught, since it isn't wrapped in a further try/except.
                iterator = iter(self.worker_data[worker_index])
                self.worker_data_iterators[worker_index] = iterator
                batch = next(iterator)
            batches.append(batch)
        return batches

    def compute_honest_worker_gradients(self, batches: Sequence[Batch]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients at each honest worker's current parameters.

        Batches are moved to the model's device (wherever :attr:`parameters`
        already lives) before the forward pass, so callers may freely mix a
        CPU-only ``DataLoader`` with a model placed on an accelerator.

        Args:
            batches: One batch per honest worker, in worker order.

        Returns:
            A tuple ``(gradients, losses)`` of stacked tensors, each with one row
            per honest worker.
        """
        device = self.parameters.device
        gradients: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []
        for worker_parameters, (inputs, targets) in zip(self.parameters, batches, strict=True):
            inputs, targets = inputs.to(device), targets.to(device)
            self.copy_parameters_to_model(worker_parameters)
            self.model.module.train()
            self.model.module.zero_grad(set_to_none=True)
            loss = self.loss_fn(self.model.module(inputs), targets)
            loss.backward()
            gradients.append(self.model.relink_gradients().detach().clone())
            losses.append(loss.detach())

        return torch.stack(gradients), torch.stack(losses)

    def copy_parameters_to_model(self, parameters: torch.Tensor) -> None:
        """Copy one flat parameter vector into the shared model wrapper.

        Args:
            parameters: Flat parameter vector of shape ``(d,)`` to load.
        """
        with torch.no_grad():
            self.model.parameters.copy_(parameters)

    @abstractmethod
    def local_update(self, gradients: torch.Tensor) -> torch.Tensor:
        """Compute the post-local-update parameters ``theta_{t+1/2}``.

        This is the optimisation seam: each protocol defines how a worker turns
        its gradient into its pre-mixing parameters, and owns any optimiser
        state (momentum, moments, ...). Implementations should update that state
        in place so :meth:`build_step_result` can snapshot it.

        Args:
            gradients: Stacked honest gradients, one row per worker.

        Returns:
            The post-local-update parameters, one row per honest worker.
        """

    def generate_byzantine_models(self, local_parameters: torch.Tensor) -> torch.Tensor:
        """Generate the Byzantine model vectors injected into the mixing phase.

        Called once per round before the per-worker mixing loop. The same
        Byzantine models are then placed into received sets by
        :meth:`gather_received_models`. Protocols whose Byzantine replies depend
        on the recipient (e.g. recipient-specific attacks) override this to
        produce them inside ``gather_received_models`` instead.

        Args:
            local_parameters: Post-local-update honest models, one row per
                worker, passed to the attack.

        Returns:
            The Byzantine models, shape ``(f, d)``; empty when ``f == 0``.
        """
        if self.f == 0:
            return local_parameters.new_empty((0, local_parameters.shape[1]))
        assert self.attack is not None  # guaranteed by __init__ when f > 0
        return self.attack.generate(local_parameters, f=self.f, **self.attack_kwargs)

    def aggregate_over_received_nodes(
        self, local_parameters: torch.Tensor, byzantine_parameters: torch.Tensor
    ) -> torch.Tensor:
        """Run the model-mixing phase over post-local-update parameter vectors.

        For each worker, builds its received set via the protocol-specific
        :meth:`gather_received_models`, then aggregates it.

        Args:
            local_parameters: Post-local-update honest models, one row per
                worker.
            byzantine_parameters: Byzantine models, shape ``(f, d)``.

        Returns:
            The mixed models, one row per honest worker.
        """
        mixed_parameters = []
        for worker_index, pivot in enumerate(local_parameters):
            candidates = self.gather_received_models(
                local_parameters,
                byzantine_parameters,
                worker_index=worker_index,
            )
            mixed_parameters.append(self.aggregate_received_models(candidates, pivot=pivot))
        return torch.stack(mixed_parameters)

    @abstractmethod
    def gather_received_models(
        self,
        honest_vectors: torch.Tensor,
        byzantine_parameters: torch.Tensor,
        *,
        worker_index: int,
    ) -> torch.Tensor:
        """Build the set of models received by one honest worker this round.

        This is the communication-topology seam: each decentralised protocol
        decides which models (honest and Byzantine) land in a worker's received
        set. Implementations should lead the set with the worker's own model so
        a pivot-anchored aggregator can rely on its position.

        Args:
            honest_vectors: Post-local-update honest models, one row per worker.
            byzantine_parameters: Byzantine models, shape ``(f, d)``, as produced
                by :meth:`generate_byzantine_models` (may be empty or ignored by
                protocols that generate Byzantine replies per recipient).
            worker_index: Index of the receiving honest worker.

        Returns:
            The received models for the worker, with its own model first.
        """

    def aggregate_received_models(self, candidates: torch.Tensor, *, pivot: torch.Tensor) -> torch.Tensor:
        """Aggregate the set of models one worker received.

        ``NearestNeighborAverage`` anchors on the worker's own model via ``pivot``;
        pivot-free aggregators (e.g. Krum, Median) absorb it through ``**specialized``.

        Args:
            candidates: The received models for one worker.
            pivot: The worker's own model, used as the distance reference.

        Returns:
            The single mixed model for the worker, shape ``(d,)``.
        """
        return self.aggregator.aggregate(candidates, pivot=pivot, **self.aggregator_kwargs)

    def commit_state(self, parameters: torch.Tensor) -> None:
        """Persist the next parameters and advance the round counter.

        Subclasses commit their own optimiser state inside :meth:`local_update`;
        this only handles the parameter state shared by every protocol.

        Args:
            parameters: The mixed models to persist as the next parameters.
        """
        self.step_index += 1
        self.parameters = parameters.detach().clone()

    @abstractmethod
    def build_step_result(
        self,
        *,
        honest_gradients: torch.Tensor,
        local_parameters: torch.Tensor,
        byzantine_parameters: torch.Tensor,
        mixed_parameters: torch.Tensor,
        losses: torch.Tensor,
    ) -> StepResultT:
        """Build the round snapshot, called after the state has been committed.

        Implementations return the protocol's :class:`StepResult` subtype,
        adding any optimiser-state fields (e.g. momentum). The committed
        :attr:`step_index` and :attr:`parameters` are available via ``self``.

        Args:
            honest_gradients: Stacked honest gradients this round.
            local_parameters: Post-local-update honest models.
            byzantine_parameters: Byzantine models injected this round.
            mixed_parameters: Mixed models (equal to the committed parameters).
            losses: Per-worker losses.

        Returns:
            The protocol snapshot, with a detached clone of each tensor.
        """
