"""Exemple complet d'utilisation de krum.orchestration."""

from torchvision import datasets, transforms

from krum.orchestration import Metric, Orchestrator
from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.bulyan import Bulyan
from krum.primitives.aggregators.krum import Krum
from krum.primitives.attacks.alie import ALIEAttack
from krum.primitives.attacks.sign_flip import SignFlipAttack
from krum.primitives.data_partitioners.iid import IidPartitioner
from krum.primitives.models.mlp import Krum2017MLPMnist
from krum.simulations.centralised.krum_nips_2017 import KrumSimulation


def make_datasets(root="./data"):
    """Charge MNIST et le normalise comme dans le tutoriel centralisé."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return (
        datasets.MNIST(root=root, train=True, download=True, transform=transform),
        datasets.MNIST(root=root, train=False, download=True, transform=transform),
    )


def my_experiment(n: int, f: int, aggregator, attack, rounds: int, seed: int = 42):
    """Une expérience de simulation fédérée basée sur KrumSimulation."""
    train_set, test_set = make_datasets()
    train_datasets = IidPartitioner.partition(train_set, n=n, seed=seed)
    krum_simulation = KrumSimulation(
        model_cls=Krum2017MLPMnist,
        train_datasets=train_datasets,
        test_set=test_set,
        aggregator=aggregator,
        attack=attack,
        n=n,
        f=f,
        rounds=rounds,
        batch_size=64,
        lr=0.01,
        seed=seed,
    )
    krum_simulation.setup()

    loss = Metric("loss", dtype=float)
    accuracy = Metric("accuracy", dtype=float)

    for step in range(rounds):
        krum_simulation.step()

        current_loss, current_acc = krum_simulation.evaluate()

        loss.push(step, current_loss)
        accuracy.push(step, current_acc)


def main() -> None:
    """Balaye les configurations et analyse les métriques lues."""
    with Orchestrator("krum_byzantine_study") as orchestrator:
        for n in [10, 20]:
            for f in [0, 2, 3]:
                for agg in [Krum, Bulyan, Average]:
                    if agg is Bulyan and n < 4 * f + 3:
                        continue
                    attacks = [None] if f == 0 else [ALIEAttack, SignFlipAttack]
                    for atk in attacks:
                        orchestrator.run(
                            my_experiment,
                            n=n,
                            f=f,
                            aggregator=agg,
                            attack=atk,
                            rounds=100,
                        )

        print(f"\n{orchestrator.drain().report()}")

        loss_df = orchestrator.get("loss").to_pandas()
        acc_df = orchestrator.get("accuracy").to_pandas()

        # Colonnes : [step, value, n, f, aggregator, attack, rounds, seed, job_key]
        print(loss_df.head())

        # Filtrage pandas classique
        krum_alie = loss_df[(loss_df["aggregator"] == "Krum") & (loss_df["attack"] == "ALIEAttack")]
        print(krum_alie.head())

        # Agrégation : loss moyenne par step pour chaque combo (aggregator, attack)
        mean_loss = loss_df.groupby(["aggregator", "attack", "step"], dropna=False)["value"].mean().reset_index()
        print(mean_loss.head())

        # Comparaison finale : loss moyenne sur les 10 derniers steps
        final_loss = (
            loss_df[loss_df["step"] >= 90].groupby(["aggregator", "attack"], dropna=False)["value"].mean().sort_values()
        )
        print("\nMeilleurs agrégateurs (loss finale moyenne) :")
        print(final_loss)

        # Merge des deux métriques sur les colonnes communes (step + paramètres du run)
        loss_renamed = loss_df.rename(columns={"value": "loss"})
        acc_renamed = acc_df.rename(columns={"value": "accuracy"})
        merged = loss_renamed.merge(acc_renamed, on=["step", "n", "f", "aggregator", "attack"])
        print("\nLoss et Accuracy combinées :")
        print(merged.head())


if __name__ == "__main__":
    main()
