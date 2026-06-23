"""Tests for the Metric write handle.

Metric only works while a run is active, so these tests establish the run state
directly via :func:`begin_run` / :func:`end_run` to exercise Metric in isolation
(without the orchestrator's fail-fast wrapping around experiment exceptions).
"""

import unittest

from krum.orchestration._context import begin_run, end_run
from krum.orchestration.metric import Metric
from krum.orchestration.orchestrator import Orchestrator


class MetricTest(unittest.TestCase):
    """Test Metric creation and push behaviour."""

    def _activate(self, params: dict | None = None) -> Orchestrator:
        """Make a run active for the rest of the test and return its orchestrator."""
        orchestrator = Orchestrator("test")
        begin_run(orchestrator, {"n": 1} if params is None else params)
        self.addCleanup(end_run)
        return orchestrator

    def test_create_outside_run_raises(self) -> None:
        """Creating a Metric with no active run raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            Metric("loss")

    def test_push_outside_run_raises(self) -> None:
        """Pushing after the run has ended raises RuntimeError."""
        self._activate()
        metric = Metric("loss", float)
        end_run()
        with self.assertRaises(RuntimeError):
            metric.push(0, 1.0)

    def test_name_with_space_raises(self) -> None:
        """A metric name containing a space raises ValueError."""
        self._activate()
        with self.assertRaises(ValueError):
            Metric("my loss")

    def test_name_and_dtype_properties(self) -> None:
        """The name and dtype are exposed as properties."""
        self._activate()
        metric = Metric("loss", float)
        self.assertEqual(metric.name, "loss")
        self.assertEqual(metric.dtype, float)

    def test_push_wrong_type_raises(self) -> None:
        """Pushing a value not matching the declared dtype raises TypeError."""
        self._activate()
        metric = Metric("loss", float)
        with self.assertRaises(TypeError):
            metric.push(0, "not a float")

    def test_push_records_value(self) -> None:
        """A pushed value is stored against the metric's channel."""
        orchestrator = self._activate({"n": 5})
        Metric("loss", float).push(0, 2.5)
        self.assertEqual(orchestrator.get("loss").to_pandas().iloc[0]["value"], 2.5)

    def test_redeclare_same_dtype_is_noop(self) -> None:
        """Re-declaring a channel with the same dtype is allowed."""
        self._activate()
        Metric("loss", float)
        Metric("loss", float)  # must not raise

    def test_redeclare_different_dtype_raises(self) -> None:
        """Re-declaring a channel with a different dtype raises ValueError."""
        self._activate()
        Metric("loss", float)
        with self.assertRaises(ValueError):
            Metric("loss", int)


if __name__ == "__main__":
    unittest.main()
