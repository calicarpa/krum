"""Second Experiment for the Krum-NIPS-2017 simulation (Cost of Resilience).

Reproduces Figure 5: compares cross-validation error at round 500 for
Average and Krum under 0% and 45% Byzantine workers across different
mini-batch sizes on both Spambase and MNIST.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datasets import mnist_dataset, spambase_dataset
from models import MLPMnist, MLPSpambase
from simulation import KrumSimulation
from torch.utils.data import Dataset

from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.krum import Krum
from krum.primitives.attacks import OmniscientAttack


def main() -> None:
    """Run Experiment 2: Cost of Resilience (Figure 5).

    Sweeps mini-batch sizes from 3 to 160 on both Spambase and MNIST,
    comparing Average and Krum under 0% and 45% Byzantine workers.
    Records error at round 500 for each configuration. Saves per-run
    results and produces ``figure_5.png``.
    """
    rounds = 500
    n = 20
    f_list = [0, 9]
    batch_sizes = [3, 5, 10, 20, 40, 80, 160]
    lr = 0.01
    seed = 42

    datasets: list[tuple[str, tuple[Dataset[Any], Dataset[Any]], type[nn.Module]]] = [
        ("spambase", spambase_dataset(), MLPSpambase),
        ("mnist", mnist_dataset(), MLPMnist),
    ]

    results_dir = Path("results/experiment_2")
    results_dir.mkdir(parents=True, exist_ok=True)

    for ds_name, (train_set, test_set), model_cls in datasets:
        for f in f_list:
            for bs in batch_sizes:
                for agg, agg_label in [
                    (Average, "Average"),
                    (Krum, "Krum"),
                ]:
                    attack = OmniscientAttack(kappa=100.0)
                    label = f"{ds_name}_{agg_label}_f{f}_bs{bs}"
                    print(f"\n=== {label} ===")

                    sim = KrumSimulation(
                        model_cls=model_cls,
                        train_set=train_set,
                        test_set=test_set,
                        aggregator=agg,
                        attack=attack,
                        n=n,
                        f=f,
                        rounds=rounds,
                        batch_size=bs,
                        lr=lr,
                        seed=seed,
                        label=label,
                    )
                    errors = sim.run()
                    final_error = errors[-1][1]

                    torch.save(
                        {
                            "error": final_error,
                            "label": label,
                            "dataset": ds_name,
                            "batch_size": bs,
                            "f": f,
                            "seed": seed,
                        },
                        results_dir / f"{label}.pt",
                    )

    _plot_results(results_dir)
    print("\nExperiment 2 done.")


def _plot_results(results_dir: Path) -> None:
    """Plot cross-validation error at round 500 vs batch size (Figure 5).

    Produces two side-by-side subplots: Spambase (left) and MNIST (right).
    Each subplot compares Average and Krum under 0% and 45% Byzantine workers.

    Args:
        results_dir: Directory containing the per-run trace files.
    """
    dataset_data: dict[str, dict[str, dict[int, float]]] = {}
    for path in sorted(results_dir.glob("*.pt")):
        entry: dict[str, Any] = torch.load(path, weights_only=True)
        ds = entry.get("dataset", "spambase")
        key = f"{entry['label'].split('_', 2)[1]}_f{entry['f']}"
        bs: int = entry["batch_size"]
        error: float = entry["error"]
        dataset_data.setdefault(ds, {}).setdefault(key, {})[bs] = error

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5))

    plot_config = [
        ("Average_f0", "b-o", "Average (0%)"),
        ("Average_f9", "b--s", "Average (45%)"),
        ("Krum_f0", "r-o", "Krum (0%)"),
        ("Krum_f9", "r--s", "Krum (45%)"),
    ]

    for ax, ds_name in [(ax0, "spambase"), (ax1, "mnist")]:
        if ds_name not in dataset_data:
            continue
        data = dataset_data[ds_name]
        for key, style, legend_label in plot_config:
            if key in data:
                points = data[key]
                xs = sorted(points.keys())
                ys = [points[x] for x in xs]
                ax.plot(xs, ys, style, label=legend_label, linewidth=1.5)

        ax.set_title(ds_name.capitalize())
        ax.set_xlabel("Mini-batch size")
        ax.set_ylabel("Cross-validation error at round 500")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1)

    fig.suptitle("Krum NIPS 2017 — Experiment 2: Cost of Resilience")
    fig.tight_layout()
    fig.savefig(results_dir / "figure_5.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
