"""Experiment 1 — MoNNA on MNIST with a sign-flip Byzantine worker.

A small decentralised run: honest workers train on Dirichlet-sharded MNIST and
mix their models by nearest-neighbor averaging, while one Byzantine worker
applies a sign-flip attack. Edit the constants below to reconfigure.
"""

from krum.primitives.attacks.sign_flip import SignFlipAttack

from ..datasets import make_datasets
from ..models import SmallMnistNet
from .run import run_monna_simulation

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
    train_set, test_set = make_datasets(
        dataset=DATASET,
        data_dir=DATA_DIR,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        num_honest=NUM_HONEST,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )
    attack = None if ATTACK == "none" else SignFlipAttack
    attack_kwargs = None if ATTACK == "none" else {"scale": SIGN_FLIP_SCALE}

    metrics = run_monna_simulation(
        model_cls=SmallMnistNet,
        train_set=train_set,
        test_set=test_set,
        n=NUM_HONEST + NUM_BYZANTINE,
        f=NUM_BYZANTINE,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        attack=attack,
        attack_kwargs=attack_kwargs,
        rounds=ROUNDS,
        eval_every=EVAL_EVERY,
        batch_size=BATCH_SIZE,
        partition=PARTITION,
        dirichlet_alpha=DIRICHLET_ALPHA,
        num_workers=NUM_WORKERS,
        seed=SEED,
    )

    print("round,train_loss_mean,test_loss_mean,test_accuracy_mean")
    for row in metrics:
        print(
            f"{int(row['round'])},{row['train_loss']:.6f},"
            f"{row['test_loss']:.6f},{row['test_accuracy']:.4f}"
        )

    print("\nExperiment 1 done.")


if __name__ == "__main__":
    main()
