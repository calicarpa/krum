"""MoNNA vs Mean on MNIST, reproducing Table 2 (Appendix D.2) of the ICML 2023 paper.

Uses the paper's own CNN (``Monna2023CNNMnist``), learning rate, batch size,
momentum, weight decay, worker/Byzantine counts, and round count, sweeping the
paper's tested data-heterogeneity values (``ALPHA_VALUES``) against ``f = 0``
(baseline) and the two attacks this library implements an exact match for —
ALIE ("little" in the paper) and sign-flip ("signflipping") — compared under
MoNNA's nearest-neighbor averaging and a plain (non-robust) mean.

Not included: the paper's "empire" (Fall of Empires) attack, whose strength is
chosen by a line search against the specific defense rather than a fixed
factor, and "labelflipping", which requires flipping training labels rather
than transforming gradients — neither fits this library's current
``Attack.generate(honest_gradients, f=...)`` primitive shape.

Edit the constants below to reconfigure.
"""

import matplotlib.pyplot as plt

from krum.orchestration import Orchestrator
from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.nearest_neighbor_average import NearestNeighborAverage
from krum.primitives.attacks.alie import ALIEAttack
from krum.primitives.attacks.sign_flip import SignFlipAttack
from krum.primitives.models.cnn import Monna2023CNNMnist

from .run import monna_experiment

# --- Configurable parameters (Table 2, Appendix D.2 — MNIST column) ---
DATASET = "mnist"
DATA_DIR = "data"
ROUNDS = 600  # T
EVAL_EVERY = 20
N = 26
F = 5
BATCH_SIZE = 25  # b
TRAIN_SIZE = 0  # 0 = full dataset
TEST_SIZE = 0  # 0 = full dataset
LEARNING_RATE = 0.75  # gamma
BETA = 0.99
WEIGHT_DECAY = 1e-4  # l2-regularization
PARTITION = "dirichlet"
ALPHA_VALUES = [0.5, 1, 5]  # data heterogeneity, all three tested in the paper
SEED = 0
NUM_WORKERS = 0

# Attacks to compare: None is the f=0 baseline (aggregator choice doesn't
# matter without an adversary); ALIE and sign-flip are exact matches for the
# paper's "little" and "signflipping" attacks.
ATTACKS = [
    (None, "None", 0),
    (ALIEAttack, "ALIE", F),
    (SignFlipAttack, "SignFlip", F),
]

# Aggregators to compare: NNA is MoNNA's default (num_closest = n - 2f,
# injected by MonnaSimulation), Average is the non-robust baseline.
AGGREGATORS = [
    (NearestNeighborAverage, "NNA"),
    (Average, "Mean"),
]
# --------------------------------------------


def main() -> None:
    """Run the MNIST paper-reproduction experiment."""
    orchestrator = Orchestrator("monna_icml_2023_experiment_mnist")

    for alpha in ALPHA_VALUES:
        for attack, attack_label, f in ATTACKS:
            for agg, agg_label in AGGREGATORS:
                label = f"{agg_label}_{attack_label}_f{f}_alpha{alpha}"
                # NNA is MonnaSimulation's own default (num_closest injected
                # automatically); pass it explicitly only when overriding to Mean.
                aggregator = None if agg is NearestNeighborAverage else agg

                orchestrator.run(
                    monna_experiment,
                    label=label,
                    dataset=DATASET,
                    data_dir=DATA_DIR,
                    model_cls=Monna2023CNNMnist,
                    n=N,
                    f=f,
                    learning_rate=LEARNING_RATE,
                    beta=BETA,
                    weight_decay=WEIGHT_DECAY,
                    attack=attack,
                    aggregator=aggregator,
                    rounds=ROUNDS,
                    eval_every=EVAL_EVERY,
                    batch_size=BATCH_SIZE,
                    train_size=TRAIN_SIZE,
                    test_size=TEST_SIZE,
                    partition=PARTITION,
                    dirichlet_alpha=alpha,
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
    axes[-1].legend(fontsize=6, loc="best")
    fig.tight_layout()
    plt.show()

    print("\nMNIST experiment done.")


if __name__ == "__main__":
    main()
