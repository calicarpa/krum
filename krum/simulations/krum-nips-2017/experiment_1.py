"""First Experiment for the Krum-NIPS-2017 simulation (Resilience to Byzantine processes).

Reproduces Figure 4: compares Average and Krum under 0% and 33% Gaussian
Byzantine workers on Spambase.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datasets import spambase_dataset
from models import MLPSpambase
from torch.utils.data import DataLoader, Dataset, Subset

from krum.primitives.aggregators import Aggregator, Average, Krum
from krum.primitives.attacks import Attack, GaussianAttack
from krum.primitives.model import Model


def _device() -> torch.device:
    """Detect the best available device.

    Returns:
        CUDA, MPS, or CPU device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int, device: torch.device) -> torch.Generator:
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed for all RNGs.
        device: Target device for device-specific seeds.

    Returns:
        A ``torch.Generator`` seeded for use with DataLoader.
    """
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif device.type == "mps":
        torch.mps.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def _train_one_round(
    model: Model,
    loader: DataLoader[Any],
    opt: torch.optim.Optimizer,
    device: torch.device,
) -> torch.Tensor:
    """Perform one SGD step on a worker's data shard.

    Args:
        model: Wrapped nn.Module exposing flat parameter and gradient views.
        loader: DataLoader yielding mini-batches from the worker's shard.
        opt: Optimizer linked to the model's parameters.
        device: Device on which to run the computation.

    Returns:
        Flat gradient tensor of shape (d,), cloned from the model.
    """
    model.module.train()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    opt.zero_grad()
    loss = nn.functional.cross_entropy(model.module(x), y)
    loss.backward()
    return model.gradients.clone()


def _evaluate(model: Model, loader: DataLoader[Any], device: torch.device) -> float:
    """Compute misclassification error rate on a held-out set.

    Args:
        model: Wrapped nn.Module.
        loader: DataLoader for the evaluation set (single full-batch loader).
        device: Device on which to run the computation.

    Returns:
        Misclassification error rate in [0, 1].
    """
    model.module.eval()
    with torch.no_grad():
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        logits = model.module(x)
        preds = logits.argmax(dim=1)
        error = (preds != y).float().mean()
    return error.item()


def run_simulation(
    label: str,
    aggregator: Aggregator,
    f: int,
    *,
    n: int,
    rounds: int,
    batch_size: int,
    lr: float,
    model_cls: type[nn.Module],
    train_set: Dataset[Any],
    test_loader: DataLoader[Any],
    attack: Attack,
    device: torch.device,
    seed: int,
    generator: torch.Generator,
) -> list[tuple[int, float]]:
    """Run a full synchronous distributed SGD simulation with Byzantine workers.

    The simulation follows the parameter-server architecture from the Krum
    NIPS 2017 paper:

    1. Partition the training set into ``n`` IID shards.
    2. Each round: broadcast model → honest workers compute gradients →
       Byzantine workers generate attack gradients → aggregate → SGD step.
    3. Track misclassification error on the test set every 10 rounds.

    Args:
        label: Human-readable name for logging.
        aggregator: Gradient aggregation rule (e.g. Average, Krum).
        f: Number of Byzantine workers.
        n: Total number of workers.
        rounds: Number of synchronous rounds.
        batch_size: Mini-batch size per honest worker.
        lr: Initial learning rate for SGD.
        lr_decay: Multiplicative learning rate decay applied each round.
        model_cls: ``nn.Module`` subclass to instantiate for training.
        train_set: Full training dataset (will be sharded across workers).
        test_loader: DataLoader for the test set (single full-batch loader).
        attack: Attack strategy for Byzantine workers.
        device: Device for training and evaluation.
        seed: Random seed for reproducibility.
        generator: Seeded ``torch.Generator`` for deterministic data loading.

    Returns:
        List of (round, error) pairs recorded during training.
    """
    model = Model(model_cls().to(device))
    opt = torch.optim.SGD(model.module.parameters(), lr=lr)
    num_honest = n - f

    train_size = len(train_set)
    shard_size = train_size // n
    shard_indices = torch.randperm(train_size, generator=torch.Generator().manual_seed(seed))
    worker_loaders: list[DataLoader[Any]] = []
    for w in range(n):
        indices = shard_indices[w * shard_size : (w + 1) * shard_size]
        worker_ds = Subset(train_set, indices.tolist())
        worker_gen = torch.Generator().manual_seed(seed + w)
        worker_loaders.append(DataLoader(worker_ds, batch_size=batch_size, shuffle=True, generator=worker_gen))

    errors: list[tuple[int, float]] = []

    for t in range(rounds):
        worker_gradients: list[torch.Tensor] = []
        honest_gradients = torch.zeros(num_honest, model.numel, device=device)

        for w in range(num_honest):
            g = _train_one_round(model, worker_loaders[w], opt, device)
            honest_gradients[w] = g
            worker_gradients.append(g)

        if f > 0:
            byz_gradients = attack.generate(honest_gradients, f)
            for g in byz_gradients:
                worker_gradients.append(g)

        all_gradients = torch.stack(worker_gradients)
        aggregated = aggregator(all_gradients)

        model.set_gradients(aggregated)
        opt.step()

        if t % 10 == 0 or t == rounds - 1:
            error = _evaluate(model, test_loader, device)
            errors.append((t, error))
            print(f"[{label}] round {t:3d}  error={error:.4f}")

    return errors


def main() -> None:
    """Run Experiment 1: Resilience to Byzantine processes (Figure 4).

    Compares Average and Krum under 0% and 33% Gaussian Byzantine workers
    (n=20, batch_size=3, 500 rounds on Spambase). Saves per-run traces as
    ``.pt`` files and produces ``figure_4.png``.
    """
    device = _device()
    seed = 42
    generator = _set_seed(seed, device)

    rounds = 500
    n = 20
    batch_size = 3
    lr = 0.01
    f_list = [0, int(n * 0.33)]

    train_set, test_set = spambase_dataset()
    test_loader: DataLoader[Any] = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    attack = GaussianAttack(std=200.0)
    results_dir = Path("results/experiment_1")
    results_dir.mkdir(parents=True, exist_ok=True)

    for f in f_list:
        for agg_cls, agg_label in [(Average, "Average"), (Krum, "Krum")]:
            agg = agg_cls(n=n, f=f)
            label = f"{agg_label}_f{f}"
            print(f"\n=== {label} ===")
            _set_seed(seed, device)
            errors = run_simulation(
                label=label,
                aggregator=agg,
                f=f,
                n=n,
                rounds=rounds,
                batch_size=batch_size,
                lr=lr,
                model_cls=MLPSpambase,
                train_set=train_set,
                test_loader=test_loader,
                attack=attack,
                device=device,
                seed=seed,
                generator=generator,
            )
            torch.save({"errors": errors, "label": label, "seed": seed}, results_dir / f"{label}.pt")

    _plot_results(results_dir)
    print("\nDone.")


def _plot_results(results_dir: Path) -> None:
    """Plot cross-validation error vs rounds (Figure 4).

    Produces two side-by-side subplots: left with 0% Byzantine, right with
    33% Byzantine. Each subplot compares Average and Krum.

    Args:
        results_dir: Directory containing the per-run trace files.
    """
    traces: dict[str, tuple[list[int], list[float]]] = {}
    for path in sorted(results_dir.glob("*.pt")):
        data: dict[str, Any] = torch.load(path, weights_only=True)
        if "errors" not in data:
            continue
        label: str = data["label"]
        rounds, errors = zip(*data["errors"], strict=True)
        traces[label] = (list(rounds), list(errors))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, f_val in [(ax0, 0), (ax1, 6)]:
        avg_key = f"Average_f{f_val}"
        krum_key = f"Krum_f{f_val}"

        ax.plot(traces[avg_key][0], traces[avg_key][1], "b-", label="Average", linewidth=1.5)
        ax.plot(traces[krum_key][0], traces[krum_key][1], "r-", label="Krum", linewidth=1.5)

        byz_pct = "0%" if f_val == 0 else "33%"
        ax.set_title(f"{byz_pct} Byzantine workers")
        ax.set_xlabel("Round")
        ax.set_ylabel("Cross-validation error")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    fig.suptitle("Krum NIPS 2017 — Experiment 1: Resilience to Byzantine processes")
    fig.tight_layout()
    fig.savefig(results_dir / "figure_4.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
