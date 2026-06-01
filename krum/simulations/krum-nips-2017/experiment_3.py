"""Third Experiment for the Krum-NIPS-2017 simulation (Multi-Krum).

Reproduces Figure 6: compares Average (0% Byzantine), Krum (33%),
and Multi-Krum (33%) under Gaussian Byzantine workers on Spambase.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from datasets import spambase_dataset
from models import MLPSpambase
from simulation import KrumSimulation

from krum.primitives.aggregators import Aggregator
from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.krum import Krum
from krum.primitives.aggregators.multikrum import MultiKrum
from krum.primitives.attacks import GaussianAttack


def main() -> None:
    """Run Experiment 3: Multi-Krum (Figure 6).

    Compares Average (0%), Krum (33%), and Multi-Krum with m=n-f (33%)
    under Gaussian Byzantine workers on Spambase (n=20, batch_size=3,
    500 rounds). Saves per-run traces and produces ``figure_6.png``.
    """
    seed = 42
    rounds = 500
    n = 20
    f = int(n * 0.33)
    batch_size = 3
    lr = 0.01

    train_set, test_set = spambase_dataset()
    attack = GaussianAttack(std=200.0)
    results_dir = Path("results/experiment_3")
    results_dir.mkdir(parents=True, exist_ok=True)

    configs: list[tuple[type[Aggregator], str, int, dict[str, Any] | None]] = [
        (Average, "Average_f0", 0, None),
        (Krum, "Krum_f6", f, None),
        (MultiKrum, "MultiKrum_f6", f, {"m": n - f - 2}),
    ]

    for agg, label, f_val, agg_kw in configs:
        print(f"\n=== {label} ===")
        sim = KrumSimulation(
            model_cls=MLPSpambase,
            train_set=train_set,
            test_set=test_set,
            aggregator=agg,
            aggregator_kwargs=agg_kw,
            attack=attack,
            n=n,
            f=f_val,
            rounds=rounds,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            label=label,
            results_dir=results_dir,
        )
        sim.run()

    _plot_results(results_dir)
    print("\nExperiment 3 done.")


def _plot_results(results_dir: Path) -> None:
    """Plot cross-validation error vs rounds (Figure 6).

    Produces a single plot with three curves: Average (0% Byzantine),
    Krum (33%), and Multi-Krum (33%).

    Args:
        results_dir: Directory containing the per-run trace files.
    """
    traces: dict[str, tuple[list[int], list[float]]] = {}
    for path in sorted(results_dir.glob("*.pt")):
        data: dict[str, Any] = torch.load(path, weights_only=True)
        if "traces" not in data:
            continue
        label: str = data["label"]
        rounds, errors, _losses = zip(*data["traces"], strict=True)
        traces[label] = (list(rounds), list(errors))

    fig, ax = plt.subplots(figsize=(8, 5))

    config = [
        ("Average_f0", "b-", "Average (0%)"),
        ("Krum_f6", "r--", "Krum (33%)"),
        ("MultiKrum_f6", "g-.", "Multi-Krum (33%)"),
    ]
    for key, style, legend_label in config:
        if key in traces:
            ax.plot(traces[key][0], traces[key][1], style, label=legend_label, linewidth=1.5)

    ax.set_title("Gaussian Byzantine (33%)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cross-validation error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    fig.suptitle("Krum NIPS 2017 — Experiment 3: Multi-Krum")
    fig.tight_layout()
    fig.savefig(results_dir / "figure_6.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
