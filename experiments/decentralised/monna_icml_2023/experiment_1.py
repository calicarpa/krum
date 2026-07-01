"""Experiment 1 — MoNNA on MNIST with a sign-flip Byzantine worker.

A small decentralised run driven by the orchestrator: honest workers train on
Dirichlet-sharded MNIST and mix their models by nearest-neighbor averaging,
while one Byzantine worker applies a sign-flip attack. Edit the constants below
to reconfigure, or wrap ``orchestrator.run`` in loops to sweep parameters.
"""

import matplotlib.pyplot as plt

from krum.orchestration import Orchestrator
from krum.primitives.attacks.sign_flip import SignFlipAttack

from ..models import SmallMnistNet
from .run import monna_experiment

# --- Configurable parameters ---
DATASET = "mnist"  # "mnist" or "fake"
DATA_DIR = "data"
ROUNDS = 50
EVAL_EVERY = 10
NUM_HONEST = 8
NUM_BYZANTINE = 1
BATCH_SIZE = 32
TRAIN_SIZE = 4096
TEST_SIZE = 1024
LEARNING_RATE = 0.5
BETA = 0.99
PARTITION = "dirichlet"  # "iid" or "dirichlet"
DIRICHLET_ALPHA = 1.0
ATTACK = "sign-flip"  # "sign-flip" or "none"
SIGN_FLIP_SCALE = 1.0
SEED = 0
NUM_WORKERS = 0
# --------------------------------------------


def main() -> None:
    """Run Experiment 1."""
    attack = None if ATTACK == "none" else SignFlipAttack
    attack_kwargs = None if ATTACK == "none" else {"scale": SIGN_FLIP_SCALE}

    orchestrator = Orchestrator("monna_icml_2023_experiment_1")
    orchestrator.run(
        monna_experiment,
        dataset=DATASET,
        data_dir=DATA_DIR,
        model_cls=SmallMnistNet,
        n=NUM_HONEST + NUM_BYZANTINE,
        f=NUM_BYZANTINE,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        attack=attack,
        attack_kwargs=attack_kwargs,
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
        ax.plot(frame["step"], frame["value"])
        ax.set_xlabel("round")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig("experiment_1.png", dpi=150)
    plt.close(fig)

    print("\nExperiment 1 done.")


if __name__ == "__main__":
    main()
