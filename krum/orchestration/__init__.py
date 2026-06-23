"""Run experiments over parameter ranges and collect their metrics.

The user writes an experiment as a single run, sweeps it with ordinary Python
loops, and reads the results back per metric.

It exposes three classes:

* :class:`~krum.orchestration.orchestrator.Orchestrator` -- drives the runs and
  owns all collected data.
* :class:`~krum.orchestration.metric.Metric` -- a named channel the user pushes
  values into; a write handle whose data lives on the orchestrator.
* :class:`~krum.orchestration.dataframe.MetricDataFrame` -- a metric's collected
  samples, viewable as a pandas frame and sliceable by parameter values.

Metric values are collected in memory and not persisted. Execution is
synchronous and fail-fast in this version; the near-term plan is multi-process,
one process per run.

Example::

    from krum.orchestration import Metric, Orchestrator

    def my_experiment(n, f, aggregator, attack, n_steps):
        simulation = KrumSimulation(n=n, f=f, aggregator=aggregator, attack=attack)
        loss = Metric("loss", dtype=float)
        for step in range(n_steps):
            simulation.step()
            loss.push(step, simulation.loss())

    orch = Orchestrator("byzantine_study")
    for n in [10, 20]:
        for f in [2, 3]:
            for aggregator in [Average, Krum, Bulyan]:
                for attack in [ALIEAttack, SignFlipAttack, None]:
                    orch.run(
                        my_experiment,
                        n=n, f=f, aggregator=aggregator, attack=attack,
                        n_steps=100,
                    )

    loss = orch.get("loss")               # MetricDataFrame
    krum_alie = loss(aggregator=Krum, attack=ALIEAttack)   # narrowed MetricDataFrame
    frame = krum_alie.to_pandas()         # pandas.DataFrame for plotting/analysis
"""

from .dataframe import MetricDataFrame
from .metric import Metric
from .orchestrator import Orchestrator

__all__ = ["Metric", "MetricDataFrame", "Orchestrator"]
