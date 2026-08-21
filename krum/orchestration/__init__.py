"""Run experiments over parameter ranges and collect their metrics.

The user writes an experiment as a single run, sweeps it with ordinary Python
loops, and reads the results back per metric.

Metric values are collected in memory and not persisted. Execution is
synchronous and fail-fast in this version; the near-term plan is multi-process,
one process per run.

Example::

    from krum.orchestration import Metric, Orchestrator
    from krum.primitives.aggregators.average import Average
    from krum.primitives.aggregators.krum import Krum
    from krum.primitives.aggregators.bulyan import Bulyan
    from krum.primitives.attacks.alie import ALIEAttack
    from krum.primitives.attacks.sign_flip import SignFlipAttack
    from krum.primitives.data_partitioners.iid import IidPartitioner
    from krum.simulations.centralised.krum_nips_2017 import KrumSimulation

    def my_experiment(n, f, aggregator, attack, seed):
        train_set, test_set = ...  # e.g. torchvision datasets
        worker_datasets = IidPartitioner.partition(train_set, n=n, seed=seed)
        simulation = KrumSimulation(
            model_cls=..., train_datasets=worker_datasets, test_set=test_set,
            aggregator=aggregator, attack=attack,
            n=n, f=f, rounds=100, batch_size=32, lr=0.1, seed=seed,
        )
        simulation.setup()
        loss = Metric("loss", dtype=float)
        for step in range(100):
            simulation.step()
            if step % 10 == 0:
                test_loss, _test_accuracy = simulation.evaluate()
                loss.push(step, test_loss)

    orch = Orchestrator("byzantine_study")
    for n, f in [(10, 2), (20, 3)]:
        for aggregator in [Average, Krum, Bulyan]:
            for attack in [ALIEAttack, SignFlipAttack]:
                orch.run(
                    my_experiment,
                    n=n, f=f, aggregator=aggregator, attack=attack, seed=42,
                )

    loss = orch.get("loss")               # MetricDataFrame
    krum_alie = loss.filter(aggregator=Krum, attack=ALIEAttack)  # narrowed MetricDataFrame
    frame = krum_alie.to_pandas()         # pandas.DataFrame for plotting/analysis
"""

from .dataframe import MetricDataFrame
from .metric import Metric
from .orchestrator import Orchestrator

__all__ = ["Metric", "MetricDataFrame", "Orchestrator"]
