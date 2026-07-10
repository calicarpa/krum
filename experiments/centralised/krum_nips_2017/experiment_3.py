"""Experiment 3 — NIPS 2017 (Blanchard et al.).

Multi-Krum vs Krum and Average under Gaussian attack on Spambase.
"""

from krum.orchestration import Orchestrator
from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.krum import Krum
from krum.primitives.aggregators.multikrum import MultiKrum
from krum.primitives.attacks.gaussian import GaussianAttack
from krum.primitives.models import Krum2017MLPSpambase

from .plot import plot_error_curves_multi_krum
from .run import krum_experiment

# --- Configurable parameters ---
ROUNDS = 500
N = 20
BATCH_SIZE = 3
LR = 0.01
SEED = 42

# Byzantine fraction (33%)
BYZ_FRACTION = 0.33

# Attack configuration
ATTACK_STD = 200.0

# MultiKrum parameter
MULTIKRUM_M = None  # None = auto (N - F - 2)
# --------------------------------------------


def main() -> None:
    """Run Experiment 3."""
    attack_kw = {"std": ATTACK_STD}

    f = int(N * BYZ_FRACTION)
    m = MULTIKRUM_M if MULTIKRUM_M is not None else (N - f - 2)

    configs = [
        (Average, "Average_f0", 0, None),
        (Krum, f"Krum_f{f}", f, None),
        (MultiKrum, f"MultiKrum_f{f}", f, {"m": m}),
    ]

    orchestrator = Orchestrator("krum_nips_2017_experiment_3")

    for agg, label, f_val, agg_kw in configs:
        orchestrator.run(
            krum_experiment,
            label=label,
            dataset="spambase",
            model_cls=Krum2017MLPSpambase,
            aggregator=agg,
            aggregator_kwargs=agg_kw,
            attack=GaussianAttack,
            attack_kwargs=attack_kw,
            n=N,
            f=f_val,
            rounds=ROUNDS,
            batch_size=BATCH_SIZE,
            lr=LR,
            seed=SEED,
        )

    print("\nExperiment 3 done.")
    loss_data = orchestrator.get("loss")
    error_data = orchestrator.get("error")
    print(len(loss_data))
    print(len(error_data))
    plot_error_curves_multi_krum(error_data)


if __name__ == "__main__":
    main()
