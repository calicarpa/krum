"""Small MoNNA vs Mean proof of concept, on MNIST under a sign-flip Byzantine worker.

A small, fast decentralised run driven by the orchestrator: honest workers
train on Dirichlet-sharded MNIST and mix their models either by
nearest-neighbor averaging (MoNNA) or by a plain (non-robust) mean, while
``f`` Byzantine workers apply the sign-flip attack. Sweeps ``f in F_VALUES``
against ``AGGREGATORS`` to illustrate the phenomenon behind #42: NNA resists
the attack and keeps training, while Mean collapses to chance-level accuracy.

This is a proof-of-concept demo, not a paper reproduction: it uses a small
MLP and a fast, illustrative configuration rather than the paper's own model
and hyperparameters.

Edit the constants below to reconfigure.
"""

import matplotlib.pyplot as plt

from krum.orchestration import Orchestrator
from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.nearest_neighbor_average import NearestNeighborAverage
from krum.primitives.attacks.sign_flip import SignFlipAttack
from krum.primitives.models.mlp import Monna2023SmallMnist

from .run import monna_experiment

# --- Configurable parameters ---
DATASET = "mnist"  # "mnist" or "fake"
DATA_DIR = "data"
ROUNDS = 200
EVAL_EVERY = 10
N = 16
F_VALUES = [0, 5]  # 5 ~= n/3
BATCH_SIZE = 32
TRAIN_SIZE = 4096
TEST_SIZE = 1024
LEARNING_RATE = 0.5
BETA = 0.99
PARTITION = "dirichlet"  # "iid" or "dirichlet"
DIRICHLET_ALPHA = 1.0
SEED = 0
NUM_WORKERS = 0

# Aggregators to compare: NNA is MoNNA's default (num_closest = n - 2f,
# injected by MonnaSimulation), Average is the non-robust baseline.
AGGREGATORS = [
    (NearestNeighborAverage, "NNA"),
    (Average, "Mean"),
]
# --------------------------------------------


def main() -> None:
    """Run the small MoNNA vs Mean proof of concept."""
    orchestrator = Orchestrator("monna_icml_2023_experiment")

    for f in F_VALUES:
        attack = None if f == 0 else SignFlipAttack
        for agg, agg_label in AGGREGATORS:
            label = f"{agg_label}_f{f}"
            # NNA is MonnaSimulation's own default (num_closest injected
            # automatically); pass it explicitly only when overriding to Mean.
            aggregator = None if agg is NearestNeighborAverage else agg

            orchestrator.run(
                monna_experiment,
                label=label,
                dataset=DATASET,
                data_dir=DATA_DIR,
                model_cls=Monna2023SmallMnist,
                n=N,
                f=f,
                learning_rate=LEARNING_RATE,
                beta=BETA,
                attack=attack,
                aggregator=aggregator,
                rounds=ROUNDS,
                eval_every=EVAL_EVERY,
                batch_size=BATCH_SIZE,
                train_size=TRAIN_SIZE,
                test_size=TEST_SIZE,
                partition=PARTITION,
                dirichlet_alpha=DIRICHLET_ALPHA,
                num_workers=NUM_WORKERS,
                seed=SEED,
            )

    metrics = ["train_loss", "test_loss", "test_accuracy"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    for ax, name in zip(axes, metrics, strict=True):
        frame = orchestrator.get(name).to_pandas()
        for run_label, group in frame.groupby("label", sort=False):
            group = group.sort_values("step")
            ax.plot(group["step"], group["value"], label=run_label)
        ax.set_xlabel("round")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[-1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    plt.show()

    print("\nSmall experiment done.")


if __name__ == "__main__":
    main()
