"""Experiment 1 — MoNNA on MNIST with a sign-flip Byzantine worker.

A small decentralised run driven by the orchestrator: honest workers train on
Dirichlet-sharded MNIST and mix their models by nearest-neighbor averaging,
while one Byzantine worker applies a sign-flip attack. Edit the constants below
to reconfigure, or wrap ``orchestrator.run`` in loops to sweep parameters.
"""

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

    # The three channels share the same run parameters and rounds. Keep both in
    # the merge key so this also works when the orchestrator drives a sweep.
    def column(metric: str, name: str):
        frame = orchestrator.get(metric).to_pandas()
        return frame.rename(columns={"step": "round", "value": name})

    train = column("train_loss", "train_loss")
    merge_keys = [column for column in train.columns if column != "train_loss"]
    report = train.merge(column("test_loss", "test_loss"), on=merge_keys).merge(
        column("test_accuracy", "test_accuracy"), on=merge_keys
    )
    print(report.to_string(index=False))

    print("\nExperiment 1 done.")


if __name__ == "__main__":
    main()
