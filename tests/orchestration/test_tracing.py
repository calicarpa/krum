"""Tests for recording what a job actually called."""

import importlib
import shutil
import sys
import unittest
from pathlib import Path
from sys import monitoring
from tempfile import TemporaryDirectory
from typing import Any

from krum.orchestration import Metric, Orchestrator
from krum.orchestration.tracing import TOOL_IDS, DependencyTracker, TracingUnavailable, verify_called

OWNED = ("orchestration", "krum")


def helper(value):
    """A function the tracked experiment calls directly."""
    return value * 2


def nested_caller(value):
    """A function whose inner function cannot be fetched back by name."""

    def inner(x):
        return x + 1

    return inner(value)


def tracked(value):
    """Call a helper, so that the call is there to be recorded."""
    return helper(value)


class TrackerTest(unittest.TestCase):
    """Test what the tracker records and what it leaves out."""

    def test_records_an_owned_callee(self) -> None:
        """A function in an owned module is recorded with its code hash."""
        with DependencyTracker(OWNED) as tracker:
            tracked(1)
        called = tracker.called()
        self.assertIn(f"{__name__}:tracked", called)
        self.assertIn(f"{__name__}:helper", called)
        self.assertEqual(len(called[f"{__name__}:helper"]), 32)

    def test_leaves_out_unowned_modules(self) -> None:
        """A call into a module nobody owns is not recorded."""
        with DependencyTracker(("nothing_at_all",)) as tracker:
            tracked(1)
        self.assertEqual(tracker.called(), {})

    def test_leaves_out_what_cannot_be_fetched_back(self) -> None:
        """A nested function is skipped: there would be nothing to re-hash."""
        with DependencyTracker(OWNED) as tracker:
            nested_caller(1)
        called = tracker.called()
        self.assertIn(f"{__name__}:nested_caller", called)
        self.assertFalse([name for name in called if "<" in name])

    def test_a_function_reports_once_however_often_it_runs(self) -> None:
        """Monitoring is disabled per function after its first call.

        This is what makes watching a long simulation affordable: the cost is
        one callback per distinct function, not one per call.
        """
        with DependencyTracker(OWNED) as tracker:
            seen = tracker._codes
            for _ in range(500):
                helper(1)
            recorded = len(seen)
        self.assertLess(recorded, 10)
        self.assertIn(f"{__name__}:helper", tracker.called())

    def test_every_tracker_sees_its_own_calls(self) -> None:
        """A second traced job records its callees just as the first did.

        Regression guard: disabling an event is recorded against the code
        object and outlives the tracker that asked for it, so without
        restarting events only the first job of a process would see anything.
        """
        with DependencyTracker(OWNED) as first:
            tracked(1)
        with DependencyTracker(OWNED) as second:
            tracked(1)
        self.assertIn(f"{__name__}:helper", first.called())
        self.assertEqual(first.called(), second.called())

    def test_the_tool_id_is_released(self) -> None:
        """Leaving the tracker frees monitoring for the next one."""
        with DependencyTracker(OWNED) as tracker:
            self.assertTrue(tracker.active)
        self.assertFalse(tracker.active)
        with DependencyTracker(OWNED) as again:
            self.assertTrue(again.active)

    def test_the_tool_id_is_released_after_a_failure(self) -> None:
        """A body that raises still leaves monitoring free."""
        with self.assertRaises(ValueError), DependencyTracker(OWNED):
            raise ValueError("the experiment failed")
        with DependencyTracker(OWNED) as again:
            self.assertTrue(again.active)

    def test_reentering_is_refused(self) -> None:
        """One tracker cannot be active twice over."""
        with DependencyTracker(OWNED) as tracker, self.assertRaises(RuntimeError):
            tracker.__enter__()

    def test_no_free_tool_id_is_reported(self) -> None:
        """With monitoring fully taken, tracing says so rather than recording nothing."""
        for identifier in TOOL_IDS:
            monitoring.use_tool_id(identifier, "a tool that got there first")
        try:
            with self.assertRaises(TracingUnavailable):
                DependencyTracker(OWNED).__enter__()
        finally:
            for identifier in TOOL_IDS:
                monitoring.free_tool_id(identifier)


class VerifyTest(unittest.TestCase):
    """Test how a recorded callee is re-checked."""

    def test_unchanged_callees_give_no_reason(self) -> None:
        """Left alone, recorded callees still match."""
        with DependencyTracker(OWNED) as tracker:
            tracked(1)
        self.assertEqual(verify_called(tracker.called()), [])

    def test_a_changed_callee_is_named(self) -> None:
        """A callee whose code no longer hashes the same is named."""
        with DependencyTracker(OWNED) as tracker:
            tracked(1)
        tampered = dict.fromkeys(tracker.called(), "00" * 16)
        reasons = verify_called(tampered)
        self.assertEqual(len(reasons), len(tampered))
        self.assertTrue(all(reason.endswith("changed") for reason in reasons))

    def test_a_missing_callee_is_named(self) -> None:
        """A callee that is no longer there is reported, not raised over."""
        self.assertEqual(
            verify_called({"krum.nowhere:Absent.method": "00" * 16}),
            ["krum.nowhere:Absent.method is no longer there"],
        )

    def test_nothing_recorded_gives_no_reason(self) -> None:
        """An empty record is nothing to disagree with."""
        self.assertEqual(verify_called({}), [])


class RuntimeDependencyTest(unittest.TestCase):
    """Test the gap tracing closes, against a dependency on disk.

    The experiment imports its helper inside its own body, so the helper is a
    local rather than a global and a static read of the bytecode cannot reach
    it. Without tracing the edit goes unnoticed; with it, the job re-runs.
    """

    MODULE = "krum_tracing_fixture"

    def setUp(self) -> None:
        """Put a rewritable helper module on the path."""
        self._directory = TemporaryDirectory()
        self.home = Path(self._directory.name)
        self.root = self.home / "study"
        sys.path.insert(0, str(self.home))
        self.write("return value * 2")

    def tearDown(self) -> None:
        """Take the helper module off the path again."""
        sys.path.remove(str(self.home))
        sys.modules.pop(self.MODULE, None)
        importlib.invalidate_caches()
        self._directory.cleanup()

    def write(self, body: str, scale: int = 2) -> None:
        """Rewrite the helper, making sure the new bytecode is what loads.

        Python validates a cached `.pyc` on size and modification time, so two
        same-length edits within one second can leave the old bytecode in
        place. The cache is cleared rather than relied upon.
        """
        (self.home / f"{self.MODULE}.py").write_text(
            f'"""A helper."""\n\nSCALE = {scale}\n\n\ndef compute(value):\n    {body}\n'
        )
        shutil.rmtree(self.home / "__pycache__", ignore_errors=True)
        sys.modules.pop(self.MODULE, None)
        importlib.invalidate_caches()

    def orchestrator(self, trace: bool) -> Orchestrator:
        """Build an orchestrator owning the fixture module."""
        return Orchestrator(
            self.root, source=self.home, owned=("__main__", "krum", self.MODULE, "orchestration"), trace=trace
        )

    def sweep(self, trace: bool, experiment: Any = None):
        """Enqueue the experiment and return the decision taken."""
        orch = self.orchestrator(trace)
        orch.run(experiment or experiment_importing_at_runtime, amount=21)
        decision = orch.plan()[0]
        orch.drain()
        return decision, orch.get("value")["value"]

    def test_untraced_a_runtime_dependency_change_is_missed(self) -> None:
        """Without tracing, editing the helper leaves a stale result standing.

        This is the false negative the design accepts for an untraced sweep,
        pinned here so that the next test means something.
        """
        self.sweep(trace=False)
        self.write("return value * 3")
        decision, values = self.sweep(trace=False)
        self.assertFalse(decision.runs)
        self.assertEqual(values, [42])

    def test_traced_a_runtime_dependency_change_is_caught(self) -> None:
        """With tracing, editing the helper re-runs the job and says why."""
        decision, values = self.sweep(trace=True)
        self.assertEqual(values, [42])
        self.write("return value * 3")
        decision, values = self.sweep(trace=True)
        self.assertTrue(decision.runs)
        self.assertEqual(decision.reasons, (f"{self.MODULE}:compute changed",))
        self.assertEqual(values, [63])

    def test_traced_a_member_read_at_runtime_is_recorded(self) -> None:
        """A constant read off the helper, with nothing of it called, is recorded by name.

        No function of the helper runs, so there is no callee to record; the
        member is found because the experiment's own code names it.
        """
        orch = self.orchestrator(trace=True)
        key = orch.run(experiment_reading_at_runtime, amount=21)
        orch.drain()
        called = orch.store.folder_for(key).called()
        assert called is not None, "the job is traced"
        self.assertIn(f"{self.MODULE}:SCALE", called)
        self.assertNotIn(f"{self.MODULE}:compute", called, "a member the code does not name is left out")

    def test_traced_a_member_change_is_caught(self) -> None:
        """Editing the constant re-runs the job and names the member."""
        decision, values = self.sweep(trace=True, experiment=experiment_reading_at_runtime)
        self.assertEqual(values, [42])
        self.write("return value * 2", scale=3)
        decision, values = self.sweep(trace=True, experiment=experiment_reading_at_runtime)
        self.assertEqual(decision.reasons, (f"{self.MODULE}:SCALE changed",))
        self.assertEqual(values, [63])

    def test_traced_an_unchanged_dependency_is_still_skipped(self) -> None:
        """Tracing does not make every job re-run."""
        self.sweep(trace=True)
        decision, _ = self.sweep(trace=True)
        self.assertFalse(decision.runs)

    def test_tracing_is_on_by_default(self) -> None:
        """A sweep records its callees without being asked to.

        The cost is one callback per distinct function, and the alternative is
        keeping a result whose dependency has since changed.
        """
        orch = Orchestrator(self.root, source=self.home, owned=("__main__", self.MODULE, "orchestration"))
        key = orch.run(experiment_importing_at_runtime, amount=21)
        orch.drain()
        called = orch.store.folder_for(key).called()
        assert called is not None, "tracing is on by default"
        self.assertIn(f"{self.MODULE}:compute", called)

    def test_callees_are_recorded_only_when_traced(self) -> None:
        """An untraced job records no callees, and is told apart from one that called none."""
        orch = self.orchestrator(trace=False)
        key = orch.run(experiment_importing_at_runtime, amount=21)
        orch.drain()
        self.assertIsNone(orch.store.folder_for(key).called())

        traced = Orchestrator(
            self.home / "traced", source=self.home, owned=("__main__", self.MODULE, "orchestration"), trace=True
        )
        traced.run(experiment_importing_at_runtime, amount=21)
        traced.drain()
        called = traced.store.folder_for(key).called()
        assert called is not None, "a traced job records its callees"
        self.assertIn(f"{self.MODULE}:compute", called)


def experiment_importing_at_runtime(amount) -> None:
    """Reach a helper by importing it inside the body, where no static read can see it."""
    import krum_tracing_fixture  # ty: ignore[unresolved-import]

    Metric("value", dtype=int).push(0, krum_tracing_fixture.compute(amount))


def experiment_reading_at_runtime(amount) -> None:
    """Read a constant off a helper imported inside the body, calling nothing of it."""
    import krum_tracing_fixture  # ty: ignore[unresolved-import]

    Metric("value", dtype=int).push(0, amount * krum_tracing_fixture.SCALE)


if __name__ == "__main__":
    unittest.main()
