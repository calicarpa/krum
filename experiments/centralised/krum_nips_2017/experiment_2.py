"""Experiment 2 — NIPS 2017 (Blanchard et al.).

Cost of Resilience: sweep batch sizes on Spambase and MNIST.
"""

from krum.orchestration import Orchestrator
from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.krum import Krum
from krum.primitives.attacks.full_gradient_negation import FullGradientNegationAttack

from ..models import MLP as MLPMnist
from .models import MLPSpambase
from .plot import plot_error_vs_batch_size
from .run import krum_experiment

# --- Configurable parameters ---
ROUNDS = 500
N = 20
LR = 0.01
SEED = 42

# Attack configuration
ATTACK_KAPPA = 100.0

# Batch sizes to sweep
BATCH_SIZES = [3, 5, 10, 20, 40, 80, 160]

# Byzantine worker counts to test
F_VALUES = [0, 8]  # 0% and 45%

# Aggregators to compare
AGGREGATORS = [
    (Average, "Average"),
    (Krum, "Krum"),
]

# Datasets to test
DATASETS = [
    ("spambase", MLPSpambase),
    ("mnist", MLPMnist),
]
# --------------------------------------------


def main() -> None:
    """Run Experiment 2."""
    attack_kw = {"kappa": ATTACK_KAPPA}

    orchestrator = Orchestrator("krum_nips_2017_experiment_2")

    for ds_name, model_cls in DATASETS:
        for f in F_VALUES:
            for bs in BATCH_SIZES:
                for agg, agg_label in AGGREGATORS:
                    label = f"{ds_name}_{agg_label}_f{f}_bs{bs}"

                    orchestrator.run(
                        krum_experiment,
                        label=label,
                        dataset=ds_name,
                        model_cls=model_cls,
                        aggregator=agg,
                        attack=FullGradientNegationAttack,
                        attack_kwargs=attack_kw,
                        n=N,
                        f=f,
                        rounds=ROUNDS,
                        batch_size=bs,
                        lr=LR,
                        seed=SEED,
                    )

    print("\nExperiment 2 done.")
    loss_data = orchestrator.get("loss")
    error_data = orchestrator.get("error")
    print(len(loss_data))
    print(len(error_data))
    plot_error_vs_batch_size(error_data)


if __name__ == "__main__":
    main()
