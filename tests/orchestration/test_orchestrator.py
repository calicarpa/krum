"""Tests for the Orchestrator."""

import unittest

from krum.orchestration.dataframe import MetricDataFrame
from krum.orchestration.metric import Metric
from krum.orchestration.orchestrator import Orchestrator


class OrchestratorTest(unittest.TestCase):
    """Test Orchestrator run, retrieval and run-identity behaviour."""

    def test_get_returns_metric_dataframe(self) -> None:
        """get() returns the channel's MetricDataFrame."""
        orchestrator = Orchestrator("t")
        orchestrator.run(
            lambda n, n_steps: Metric("loss", float).push(0, float(n)),
            n=1,
            n_steps=1,
        )
        self.assertIsInstance(orchestrator.get("loss"), MetricDataFrame)

    def test_get_unknown_metric_raises(self) -> None:
        """get() on an undeclared channel raises KeyError."""
        with self.assertRaises(KeyError):
            Orchestrator("t").get("nope")

    def test_metrics_lists_declared_channels(self) -> None:
        """metrics() lists every declared channel name."""
        orchestrator = Orchestrator("t")

        def experiment(n_steps: int) -> None:
            Metric("loss", float)
            Metric("accuracy", float)

        orchestrator.run(experiment, n_steps=1)
        self.assertEqual(set(orchestrator.metrics()), {"loss", "accuracy"})

    def test_sweep_accumulates_across_runs(self) -> None:
        """Samples from every run accumulate under the same channel."""
        orchestrator = Orchestrator("t")

        def experiment(n: int, n_steps: int) -> None:
            loss = Metric("loss", float)
            for step in range(n_steps):
                loss.push(step, float(n + step))

        for n in (10, 20):
            orchestrator.run(experiment, n=n, n_steps=3)
        self.assertEqual(len(orchestrator.get("loss")), 6)

    def test_same_experiment_across_runs_is_allowed(self) -> None:
        """Running the same experiment many times is fine."""
        orchestrator = Orchestrator("t")

        def experiment(n: int, n_steps: int) -> None:
            Metric("loss", float).push(0, float(n))

        orchestrator.run(experiment, n=10, n_steps=1)
        orchestrator.run(experiment, n=20, n_steps=1)  # must not raise
        self.assertEqual(len(orchestrator.get("loss")), 2)

    def test_different_experiment_raises(self) -> None:
        """Running a different experiment on the same orchestrator raises ValueError."""
        orchestrator = Orchestrator("t")

        def experiment_a(n_steps: int) -> None:
            Metric("loss", float).push(0, 1.0)

        def experiment_b(n_steps: int) -> None:
            Metric("loss", float).push(0, 2.0)

        orchestrator.run(experiment_a, n_steps=1)
        with self.assertRaises(ValueError):
            orchestrator.run(experiment_b, n_steps=1)

    def test_explicit_reserved_param_name_raises(self) -> None:
        """A parameter named like a reserved column raises ValueError."""

        def experiment(step: int, n_steps: int) -> None:
            pass

        with self.assertRaises(ValueError):
            Orchestrator("t").run(experiment, step=5, n_steps=1)

    def test_defaulted_reserved_param_name_raises(self) -> None:
        """A defaulted parameter named like a reserved column also raises."""

        def experiment(n_steps: int, step: int = 0) -> None:
            pass

        with self.assertRaises(ValueError):
            Orchestrator("t").run(experiment, n_steps=1)

    def test_failure_wraps_in_runtime_error_naming_the_run(self) -> None:
        """An experiment error becomes a RuntimeError naming the run, chaining the cause."""
        orchestrator = Orchestrator("t")

        def boom(n: int, n_steps: int) -> None:
            raise ValueError("diverged")

        with self.assertRaises(RuntimeError) as context:
            orchestrator.run(boom, n=10, n_steps=1)
        self.assertIsInstance(context.exception.__cause__, ValueError)
        self.assertIn("n", str(context.exception))

    def test_context_cleared_after_failure(self) -> None:
        """After a failed run, no run is active (the context was cleared)."""
        orchestrator = Orchestrator("t")

        def boom(n_steps: int) -> None:
            raise ValueError("diverged")

        with self.assertRaises(RuntimeError):
            orchestrator.run(boom, n_steps=1)
        with self.assertRaises(RuntimeError):
            Metric("loss")  # no active run

    def test_dtype_mismatch_surfaces_as_chained_value_error(self) -> None:
        """Re-declaring a channel with a clashing dtype fails the run."""
        orchestrator = Orchestrator("t")

        def experiment(n_steps: int) -> None:
            Metric("loss", float)
            Metric("loss", int)

        with self.assertRaises(RuntimeError) as context:
            orchestrator.run(experiment, n_steps=1)
        self.assertIsInstance(context.exception.__cause__, ValueError)

    def test_defaults_are_recorded(self) -> None:
        """A defaulted argument is recorded as a parameter of the run."""
        orchestrator = Orchestrator("t")

        def experiment(n: int, n_steps: int, lr: float = 0.1) -> None:
            Metric("loss", float).push(0, float(n))

        orchestrator.run(experiment, n=10, n_steps=1)  # lr left to default
        self.assertIn("lr", orchestrator.get("loss").dataframe().index.names)

    def test_explicit_and_defaulted_value_collapse_to_one_run(self) -> None:
        """Passing a default explicitly yields the same run identity as omitting it."""
        orchestrator = Orchestrator("t")

        def experiment(n: int, n_steps: int, lr: float = 0.1) -> None:
            Metric("loss", float).push(0, float(n))

        orchestrator.run(experiment, n=10, n_steps=1)  # lr defaulted to 0.1
        orchestrator.run(experiment, n=10, n_steps=1, lr=0.1)  # lr = 0.1 explicitly
        self.assertEqual(len(orchestrator.get("loss")), 1)


if __name__ == "__main__":
    unittest.main()
