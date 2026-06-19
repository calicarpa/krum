"""Exemple complet d'utilisation de krum.orchestration."""

from krum.orchestration import Metric, Orchestrator
from krum.simulation import KrumSimulation
from torch import float

from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.bulyan import Bulyan
from krum.primitives.aggregators.krum import Krum
from krum.primitives.attacks.alie import ALIEAttack
from krum.primitives.attacks.sign_flip import SignFlipAttack


def my_experiment(n: int, f: int, aggregator, attack, n_steps: int):
    """Une expérience de simulation de fédéré basée sur krum.simulation."""
    krum_simulation = KrumSimulation(n=n, f=f, aggregator=aggregator, attack=attack)

    loss = Metric("loss", dtype=float)
    accuracy = Metric("accuracy", dtype=float)

    for step in range(n_steps):
        krum_simulation.step()

        current_loss = krum_simulation.loss()
        current_acc = krum_simulation.accuracy()

        loss.push(step, current_loss, skip_if_exists=True)
        accuracy.push(step, current_acc, skip_if_exists=True)


orchestrator = Orchestrator("krum_byzantine_study")

for n in [10, 20]:
    for f in [2, 3]:
        for agg in [Krum, Bulyan, Average]:
            for atk in [ALIEAttack, SignFlipAttack, None]:
                orchestrator.run(
                    my_experiment,
                    n=n,
                    f=f,
                    aggregator=agg,
                    attack=atk,
                    n_steps=100,
                )


loss_df = orchestrator.get("loss")
acc_df = orchestrator.get("accuracy")

# Colonnes : [step, value, n, f, aggregator, attack]
print(loss_df.head())

# Filtrage pandas classique
krum_alie = loss_df[(loss_df["aggregator"] == "Krum") & (loss_df["attack"] == "Alie")]

# Agrégation : loss moyenne par step pour chaque combo (aggregator, attack)
mean_loss = loss_df.groupby(["aggregator", "attack", "step"])["value"].mean().reset_index()

# Comparaison finale : loss moyenne sur les 10 derniers steps
final_loss = loss_df[loss_df["step"] >= 90].groupby(["aggregator", "attack"])["value"].mean().sort_values()
print("\nMeilleurs agrégateurs (loss finale moyenne) :")
print(final_loss)

# Merge des deux métriques sur les colonnes communes (step + paramètres du run)
loss_renamed = loss_df.rename(columns={"value": "loss"})
acc_renamed = acc_df.rename(columns={"value": "accuracy"})
merged = loss_renamed.merge(acc_renamed, on=["step", "n", "f", "aggregator", "attack"])
print("\nLoss et Accuracy combinées :")
print(merged.head())
