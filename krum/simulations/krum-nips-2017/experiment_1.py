"""First Experiment for the Krum-NIPS-2017 simulation (Resilience to Byzantine processes).

Reproduces Figure 4: compares Average and Krum under 0% and 33% Gaussian
Byzantine workers on Spambase.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from datasets import spambase_dataset
from models import MLPSpambase
from simulation import KrumSimulation

from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.krum import Krum
from krum.primitives.attacks import GaussianAttack


def main() -> None:
    """Run Experiment 1: Resilience to Byzantine processes (Figure 4).

    Compares Average and Krum under 0% and 33% Gaussian Byzantine workers
    (n=20, batch_size=3, 500 rounds on Spambase). Saves per-run traces as
    ``.pt`` files and produces ``figure_4.png``.
    """
    seed = 42
    rounds = 500
    n = 20
    batch_size = 3
    lr = 0.01
    f_list = [0, int(n * 0.33)]

    train_set, test_set = spambase_dataset()
    attack = GaussianAttack(std=200.0)
    results_dir = Path("results/experiment_1")
    results_dir.mkdir(parents=True, exist_ok=True)

    for f in f_list:
        for agg, agg_label in [
            (Average, "Average"),
            (Krum, "Krum"),
        ]:
            label = f"{agg_label}_f{f}"
            print(f"\n=== {label} ===")

            sim = KrumSimulation(
                model_cls=MLPSpambase,
                train_set=train_set,
                test_set=test_set,
                aggregator=agg,
                attack=attack,
                n=n,
                f=f,
                rounds=rounds,
                batch_size=batch_size,
                lr=lr,
                seed=seed,
                label=label,
                results_dir=results_dir,
            )
            sim.run()

    _plot_results(results_dir)
    print("\nExperiment 1 done.")


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
        if "traces" not in data:
            continue
        label: str = data["label"]
        rounds, errors, _losses = zip(*data["traces"], strict=True)
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
