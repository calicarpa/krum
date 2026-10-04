"""Tests for the orchestrator: identity, skipping, persistence and failure."""

import os
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

from krum.orchestration import Metric, Modules, NoActiveJob, Orchestrator, RunFailed
from krum.orchestration.metrics import collect

ROUNDS = 3
# Set by a test to make the same experiment fail on demand. This lives in the
# environment rather than in a module global on purpose: a global the
# experiment reads is part of its identity, so mutating one would change the
# job key and defeat any test of re-running the *same* job.
EXPLODE = "KRUM_TEST_EXPLODE"
# Read by `scaled_experiment`, and mutated to show a dependency change
SCALE = 1


class Average:
    """A stand-in aggregator passed as a parameter."""


class Krum:
    """A second stand-in aggregator passed as a parameter."""


def experiment(n, f, aggregator, seed=42) -> None:
    """Record two metrics, and fail for a marked worker count."""
    if os.environ.get(EXPLODE) == str(n):
        raise ValueError(f"n={n} is not supported")
    loss = Metric("loss", dtype=float)
    accuracy = Metric("test accuracy", dtype=float)
    for step in range(ROUNDS):
        loss.push(step, 1.0 / (step + 1) + f)
        accuracy.push(step, step / ROUNDS)


def recording_experiment(recorded) -> None:
    """Record a single metric value."""
    Metric("value", dtype=int).push(0, recorded)


def scaled_experiment(n) -> None:
    """Record a value scaled by a module-level constant."""
    Metric("value", dtype=int).push(0, n * SCALE)


def interrupted_experiment(recorded) -> None:
    """Record a value, then interrupt the sweep."""
    Metric("value", dtype=int).push(0, recorded)
    raise KeyboardInterrupt


class OrchestratorTestCase(unittest.TestCase):
    """Shared fixture: a store in a temporary directory, outside any repository."""

    def setUp(self) -> None:
        """Open an orchestrator whose provenance is read outside a repository."""
        os.environ.pop(EXPLODE, None)
        self._directory = TemporaryDirectory()
        self.home = Path(self._directory.name)
        self.root = self.home / "study"

    def tearDown(self) -> None:
        """Remove the temporary directory."""
        os.environ.pop(EXPLODE, None)
        self._directory.cleanup()

    def orchestrator(self) -> Orchestrator:
        """Build an orchestrator with deterministic, repository-free provenance."""
        return Orchestrator(self.root, source=self.home)


class IdentityTest(OrchestratorTestCase):
    """Test the keys runs are enqueued under."""

    def test_same_run_gives_the_same_key(self) -> None:
        """Equal parameters give one key, hence one folder."""
        orch = self.orchestrator()
        first = orch.run(experiment, n=10, f=2, aggregator=Krum)
        second = orch.run(experiment, n=10, f=2, aggregator=Krum)
        self.assertEqual(first, second)

    def test_different_parameters_give_different_keys(self) -> None:
        """A changed parameter is a different job."""
        orch = self.orchestrator()
        self.assertNotEqual(
            orch.run(experiment, n=10, f=2, aggregator=Krum),
            orch.run(experiment, n=10, f=2, aggregator=Average),
        )

    def test_unknown_parameter_is_rejected_at_the_call_site(self) -> None:
        """A parameter the experiment does not accept raises on `run`, not later."""
        orch = self.orchestrator()
        with self.assertRaises(TypeError):
            orch.run(experiment, n=10, f=2, aggregator=Krum, nope=1)
        self.assertEqual(orch.queued, 0)

    def test_an_ownership_test_may_be_given_instead_of_prefixes(self) -> None:
        """Ownership may be given as a `Modules`, which carries exclusions.

        Prefixes alone cannot own `krum` while disowning a subpackage of it, so
        the richer form is accepted wherever prefixes are.
        """
        owned = Modules(("orchestration", "krum"), exclude=("krum.primitives",))
        orch = Orchestrator(self.root, source=self.home, owned=owned)
        key = orch.run(recording_experiment, recorded=1)
        orch.drain()
        self.assertEqual(orch.store.folder_for(key).manifest()["owned"], ["krum", "orchestration"])

    def test_enqueueing_runs_nothing(self) -> None:
        """`run` only enqueues; nothing is executed until the queue is drained."""
        orch = self.orchestrator()
        orch.run(experiment, n=10, f=2, aggregator=Krum)
        self.assertEqual(orch.queued, 1)
        self.assertEqual(list(orch.store), [])


class DrainTest(OrchestratorTestCase):
    """Test execution, persistence and skipping."""

    def test_drain_executes_and_persists(self) -> None:
        """A drained run leaves a completed folder holding its metrics."""
        orch = self.orchestrator()
        key = orch.run(experiment, n=10, f=2, aggregator=Krum)
        summary = orch.drain()
        self.assertEqual(summary.count("done"), 1)
        folder = orch.store.folder_for(key)
        self.assertTrue(folder.done)
        self.assertEqual(folder.metric_names(), ["loss", "test accuracy"])

    def test_manifest_records_defaults_and_provenance(self) -> None:
        """The manifest records the parameters the key was computed from."""
        orch = self.orchestrator()
        key = orch.run(experiment, n=10, f=2, aggregator=Krum)
        orch.drain()
        manifest = orch.store.folder_for(key).manifest()
        self.assertEqual(manifest["params"]["seed"], 42)
        self.assertEqual(manifest["params"]["aggregator"]["name"], "Krum")
        self.assertEqual(manifest["callable"]["qualname"], "experiment")
        self.assertEqual(manifest["job_key"], orch.store.name_for(key))

    def test_second_drain_skips_completed_runs(self) -> None:
        """A completed job is skipped rather than re-executed."""
        orch = self.orchestrator()
        orch.run(experiment, n=10, f=2, aggregator=Krum)
        orch.drain()
        again = self.orchestrator()
        again.run(experiment, n=10, f=2, aggregator=Krum)
        summary = again.drain()
        self.assertEqual(summary.count("skipped"), 1)
        self.assertEqual(summary.count("done"), 0)

    def test_skipping_does_not_re_execute_the_body(self) -> None:
        """A skipped job's output is the one already on disk."""
        orch = self.orchestrator()
        orch.run(recording_experiment, recorded=1)
        orch.drain()
        self.assertEqual(orch.get("value")["value"], [1])
        # Same parameters, so the same job: the recorded value stands
        again = self.orchestrator()
        again.run(recording_experiment, recorded=1)
        self.assertEqual(again.drain().count("skipped"), 1)
        self.assertEqual(again.get("value")["value"], [1])

    def test_queue_is_emptied(self) -> None:
        """Draining consumes the queue."""
        orch = self.orchestrator()
        orch.run(experiment, n=10, f=2, aggregator=Krum)
        orch.drain()
        self.assertEqual(orch.queued, 0)

    def test_context_manager_drains_on_exit(self) -> None:
        """Leaving the block cleanly runs the sweep."""
        with self.orchestrator() as orch:
            orch.run(experiment, n=10, f=2, aggregator=Krum)
        self.assertEqual(len([folder for folder in orch.store if folder.done]), 1)

    def test_context_manager_does_not_drain_while_unwinding(self) -> None:
        """A failure inside the block is not followed by a sweep."""
        orch = self.orchestrator()
        with self.assertRaises(RuntimeError), orch:
            orch.run(experiment, n=10, f=2, aggregator=Krum)
            raise RuntimeError("the user's own failure")
        self.assertEqual(list(orch.store), [])
        self.assertEqual(orch.queued, 1)


class ReadBackTest(OrchestratorTestCase):
    """Test the frame read back from completed jobs."""

    def test_frame_is_tidy_and_carries_the_parameters(self) -> None:
        """One row per step, with the sweep dimensions as columns."""
        orch = self.orchestrator()
        for f in (2, 3):
            orch.run(experiment, n=10, f=f, aggregator=Krum)
        table = orch.get("loss")
        self.assertEqual(table.columns, ("step", "value", "n", "f", "aggregator", "seed", "job_key"))
        self.assertEqual(len(table), 2 * ROUNDS)
        self.assertEqual(sorted(set(table["f"])), [2, 3])
        self.assertEqual(set(table["aggregator"]), {"Krum"})

    def test_get_drains_first(self) -> None:
        """Reading a metric runs whatever is still queued."""
        orch = self.orchestrator()
        orch.run(experiment, n=10, f=2, aggregator=Krum)
        self.assertEqual(len(orch.get("loss")), ROUNDS)

    def test_frame_supports_grouping(self) -> None:
        """The frame is shaped for the comparisons a sweep is run for."""
        orch = self.orchestrator()
        for aggregator in (Krum, Average):
            orch.run(experiment, n=10, f=2, aggregator=aggregator)
        means = orch.get("loss").to_pandas().groupby("aggregator")["value"].mean()
        self.assertEqual(sorted(means.index), ["Average", "Krum"])

    def test_metric_name_with_a_space_round_trips(self) -> None:
        """An unrestricted metric name is readable back under its own name."""
        orch = self.orchestrator()
        orch.run(experiment, n=10, f=2, aggregator=Krum)
        orch.drain()
        self.assertIn("test accuracy", orch.metrics())
        self.assertEqual(len(orch.get("test accuracy")), ROUNDS)

    def test_unknown_metric_lists_what_is_available(self) -> None:
        """Asking for a metric nothing recorded says what was recorded."""
        orch = self.orchestrator()
        orch.run(experiment, n=10, f=2, aggregator=Krum)
        orch.drain()
        with self.assertRaises(KeyError) as caught:
            orch.get("nope")
        self.assertIn("loss", str(caught.exception))

    def test_dtype_is_applied(self) -> None:
        """Recorded values are coerced to the declared dtype."""
        orch = self.orchestrator()
        orch.run(recording_experiment, recorded=3.7)
        self.assertEqual(orch.get("value")["value"], [3])


class SweepScopeTest(OrchestratorTestCase):
    """A reading covers the sweep that asked for it, not the whole store."""

    def test_only_this_sweep_is_read(self) -> None:
        """A job recorded by an earlier sweep is not read by a later one.

        A store accumulates a folder per version of an experiment, so reading
        everything in it mixes code versions. Two of them can carry the same
        label and be drawn over each other, which is silent and wrong.
        """
        first = self.orchestrator()
        first.run(recording_experiment, recorded=1)
        first.drain()
        second = self.orchestrator()
        second.run(recording_experiment, recorded=2)
        self.assertEqual(second.get("value")["value"], [2])
        self.assertEqual(len(list(second.store)), 2)

    def test_the_whole_store_is_still_readable(self) -> None:
        """Reading everything a store holds stays available, explicitly."""
        orch = self.orchestrator()
        for recorded in (1, 2):
            orch.run(recording_experiment, recorded=recorded)
        orch.drain()
        later = self.orchestrator()
        later.run(recording_experiment, recorded=3)
        later.drain()
        self.assertEqual(later.get("value")["value"], [3])
        self.assertEqual(sorted(collect(later.store, "value")["value"]), [1, 2, 3])

    def test_reading_follows_the_enqueued_order(self) -> None:
        """Rows come back in the order the sweep was defined.

        Plots group with `sort=False` and assign colours in order, so the
        order a sweep was written in is the order it should read back in.
        """
        orch = self.orchestrator()
        for recorded in (30, 10, 20):
            orch.run(recording_experiment, recorded=recorded)
        self.assertEqual(orch.get("value")["value"], [30, 10, 20])

    def test_enqueued_keys_are_reported_without_repeats(self) -> None:
        """The same run enqueued twice is one job of the sweep."""
        orch = self.orchestrator()
        first = orch.run(recording_experiment, recorded=1)
        second = orch.run(recording_experiment, recorded=1)
        self.assertEqual(first, second)
        self.assertEqual(orch.enqueued, (orch.store.name_for(first),))

    def test_an_empty_sweep_reads_the_store(self) -> None:
        """With nothing enqueued, a reading falls back on the whole store."""
        orch = self.orchestrator()
        orch.run(recording_experiment, recorded=1)
        orch.drain()
        inspector = self.orchestrator()
        self.assertEqual(inspector.get("value")["value"], [1])


class MetricTest(OrchestratorTestCase):
    """Test the metric writer's contract."""

    def test_metric_outside_a_job_is_reported(self) -> None:
        """Creating a metric with no job running raises rather than losing data."""
        with self.assertRaises(NoActiveJob):
            Metric("loss")

    def test_same_metric_twice_appends(self) -> None:
        """Re-creating a metric inside one job keeps appending to one file."""

        def twice(recorded) -> None:
            Metric("value", dtype=int).push(0, recorded)
            Metric("value", dtype=int).push(1, recorded + 1)

        orch = self.orchestrator()
        orch.run(twice, recorded=5)
        self.assertEqual(orch.get("value")["value"], [5, 6])

    def test_skip_if_exists_keeps_the_first_value(self) -> None:
        """A step already recorded is left untouched."""
        pushed = []

        def repeated(recorded) -> None:
            metric = Metric("value", dtype=int)
            pushed.append(metric.push(0, recorded, skip_if_exists=True))
            pushed.append(metric.push(0, recorded + 100, skip_if_exists=True))

        orch = self.orchestrator()
        orch.run(repeated, recorded=7)
        self.assertEqual(orch.get("value")["value"], [7])
        self.assertEqual(pushed, [True, False])


class ReservedNameTest(OrchestratorTestCase):
    """A sweep dimension must not shadow a metric frame's own columns."""

    def test_reserved_parameter_is_rejected_at_the_call_site(self) -> None:
        """A parameter named after a metric column is refused when enqueued.

        Regression guard: such a parameter used to be written over the metric's
        own column on the way out, silently replacing every recorded value.
        """

        def takes_step(step) -> None:
            """A fixture whose parameter shadows the step column."""

        def takes_value(value) -> None:
            """A fixture whose parameter shadows the value column."""

        def takes_job_key(job_key) -> None:
            """A fixture whose parameter shadows the job key column."""

        for function, name in ((takes_step, "step"), (takes_value, "value"), (takes_job_key, "job_key")):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as caught:
                    self.orchestrator().run(function, **{name: 1})
                self.assertIn("reserved", str(caught.exception))

    def test_reserved_name_from_a_default_is_rejected(self) -> None:
        """A reserved name is caught even when it only appears as a default."""

        def defaulted(n, step=1) -> None:
            Metric("value", dtype=int).push(0, n)

        with self.assertRaises(ValueError):
            self.orchestrator().run(defaulted, n=1)

    def test_ordinary_names_are_accepted(self) -> None:
        """Names that do not collide are unaffected."""
        self.assertIsNotNone(self.orchestrator().run(recording_experiment, recorded=1))


class DependencyTest(OrchestratorTestCase):
    """A job's identity follows the code it depends on, not just its parameters."""

    def test_changing_a_dependency_is_a_different_job(self) -> None:
        """An experiment reading a changed constant is a different job.

        This is step 1 and step 2 composed: the key covers the code reached
        from the experiment, so a changed constant lands in its own folder
        rather than silently reusing the earlier result.
        """
        global SCALE
        orch = self.orchestrator()
        first = orch.run(scaled_experiment, n=2)
        orch.drain()
        try:
            SCALE = 10
            again = self.orchestrator()
            second = again.run(scaled_experiment, n=2)
            self.assertNotEqual(first, second)
            self.assertEqual(again.drain().count("done"), 1)
        finally:
            SCALE = 1
        values = sorted(self.orchestrator().get("value")["value"])
        self.assertEqual(values, [2, 20])

    def test_unchanged_dependency_is_the_same_job(self) -> None:
        """Left alone, the same experiment is the same job."""
        orch = self.orchestrator()
        orch.run(scaled_experiment, n=2)
        orch.drain()
        again = self.orchestrator()
        again.run(scaled_experiment, n=2)
        self.assertEqual(again.drain().count("skipped"), 1)


class StalenessTest(OrchestratorTestCase):
    """Test what decides that a recorded result no longer stands."""

    def plan_one(self, orch: Orchestrator):
        """The single decision a one-run sweep would make."""
        decisions = orch.plan()
        self.assertEqual(len(decisions), 1)
        return decisions[0]

    def test_a_new_job_runs(self) -> None:
        """With nothing recorded, the job runs."""
        orch = self.orchestrator()
        orch.run(recording_experiment, recorded=1)
        decision = self.plan_one(orch)
        self.assertTrue(decision.runs)
        self.assertEqual(decision.reasons, ("no recorded result",))

    def test_planning_runs_nothing(self) -> None:
        """Inspecting a sweep does not execute it."""
        orch = self.orchestrator()
        orch.run(recording_experiment, recorded=1)
        orch.plan()
        self.assertEqual(list(orch.store), [])
        self.assertEqual(orch.queued, 1)

    def test_a_recorded_job_is_skipped(self) -> None:
        """An unchanged environment leaves a recorded result standing."""
        orch = self.orchestrator()
        orch.run(recording_experiment, recorded=1)
        orch.drain()
        again = self.orchestrator()
        again.run(recording_experiment, recorded=1)
        decision = self.plan_one(again)
        self.assertFalse(decision.runs)
        self.assertEqual(decision.reasons, ())

    def test_a_failed_job_runs_again(self) -> None:
        """A failed attempt is no reason to keep its folder."""
        os.environ[EXPLODE] = "20"
        orch = self.orchestrator()
        orch.run(experiment, n=20, f=2, aggregator=Krum)
        with self.assertRaises(RunFailed):
            orch.drain()
        again = self.orchestrator()
        again.run(experiment, n=20, f=2, aggregator=Krum)
        self.assertEqual(self.plan_one(again).reasons, ("previous attempt failed",))

    def test_a_changed_environment_runs_again(self) -> None:
        """A dependency bump invalidates a recorded result.

        This is the witness the job key cannot cover: third-party code is
        folded into a key as a location only, so a version change is invisible
        to it by design.
        """
        lock = self.home / "uv.lock"
        lock.write_text("version = 1\n")
        orch = Orchestrator(self.root, source=self.home, lock=lock)
        orch.run(recording_experiment, recorded=1)
        orch.drain()

        lock.write_text("version = 2\n")
        again = Orchestrator(self.root, source=self.home, lock=lock)
        again.run(recording_experiment, recorded=1)
        decision = self.plan_one(again)
        self.assertTrue(decision.runs)
        self.assertIn("uv_lock changed", decision.reasons[0])
        self.assertEqual(again.drain().count("done"), 1)

    def test_an_unverifiable_job_runs_again(self) -> None:
        """A recorded result with no fingerprint is treated as stale."""
        orch = self.orchestrator()
        key = orch.run(recording_experiment, recorded=1)
        orch.drain()
        (orch.store.folder_for(key).path / "deps.json").unlink()
        again = self.orchestrator()
        again.run(recording_experiment, recorded=1)
        self.assertIn("fingerprint missing", self.plan_one(again).reasons[0])

    def test_force_runs_everything(self) -> None:
        """Forcing overrides a result that would otherwise stand."""
        orch = self.orchestrator()
        orch.run(recording_experiment, recorded=1)
        orch.drain()
        again = Orchestrator(self.root, source=self.home, force=True)
        again.run(recording_experiment, recorded=1)
        self.assertEqual(self.plan_one(again).reasons, ("forced",))
        self.assertEqual(again.drain().count("done"), 1)

    def test_force_can_be_overridden_per_drain(self) -> None:
        """A drain may force, or decline to, whatever the orchestrator says."""
        orch = self.orchestrator()
        orch.run(recording_experiment, recorded=1)
        orch.drain()
        again = self.orchestrator()
        again.run(recording_experiment, recorded=1)
        self.assertTrue(again.plan(force=True)[0].runs)
        self.assertEqual(again.drain(force=True).count("done"), 1)

    def test_re_running_replaces_the_earlier_output(self) -> None:
        """A stale job's folder is replaced, not merged into."""
        orch = self.orchestrator()
        key = orch.run(recording_experiment, recorded=1)
        orch.drain()
        stale = orch.store.folder_for(key)
        (stale.path / "metrics" / "left-over.csv").write_text("step,value\n0,1\n")
        self.assertIn("left-over", stale.metric_names())

        again = Orchestrator(self.root, source=self.home, force=True)
        again.run(recording_experiment, recorded=1)
        again.drain()
        self.assertEqual(again.store.folder_for(key).metric_names(), ["value"])

    def test_reasons_reach_the_summary(self) -> None:
        """A drain says why each job ran, not only that it did."""
        orch = self.orchestrator()
        orch.run(recording_experiment, recorded=1)
        self.assertIn("no recorded result", orch.drain().report())

    def test_witnesses_are_computed_once(self) -> None:
        """A sweep is one environment, read once rather than per job."""
        orch = self.orchestrator()
        self.assertIs(orch.witnesses(), orch.witnesses())


class FailureTest(OrchestratorTestCase):
    """Test fail-fast, and what is left on disk afterwards."""

    def test_failure_stops_the_sweep(self) -> None:
        """The first failure stops the drain and reports what was left."""
        os.environ[EXPLODE] = "20"
        orch = self.orchestrator()
        orch.run(experiment, n=10, f=2, aggregator=Krum)
        orch.run(experiment, n=20, f=2, aggregator=Krum)
        orch.run(experiment, n=30, f=2, aggregator=Krum)
        with self.assertRaises(RunFailed) as caught:
            orch.drain()
        summary = caught.exception.summary
        self.assertEqual(summary.count("done"), 1)
        self.assertEqual(summary.count("failed"), 1)
        self.assertEqual(summary.pending, 1)

    def test_failed_job_records_its_traceback(self) -> None:
        """A failed job is promoted with its traceback, for inspection."""
        os.environ[EXPLODE] = "20"
        orch = self.orchestrator()
        key = orch.run(experiment, n=20, f=2, aggregator=Krum)
        with self.assertRaises(RunFailed):
            orch.drain()
        folder = orch.store.folder_for(key)
        self.assertEqual(folder.status, "failed")
        recorded = folder.traceback()
        assert recorded is not None, "a failed job records its traceback"
        self.assertIn("n=20 is not supported", recorded)
        self.assertEqual(folder.manifest()["status"], "failed")

    def test_failed_job_is_excluded_from_the_read_path(self) -> None:
        """A failed job's partial metrics do not reach the frame."""
        os.environ[EXPLODE] = "20"
        orch = self.orchestrator()
        orch.run(recording_experiment, recorded=1)
        orch.drain()
        orch.run(experiment, n=20, f=2, aggregator=Krum)
        with self.assertRaises(RunFailed):
            orch.drain()
        self.assertEqual(len(orch.get("value")), 1)

    def test_failure_abandons_the_queue(self) -> None:
        """A failure empties the queue, so completed results stay readable.

        Were the failing job left queued, every later `get` would re-run it and
        re-raise, making the results that did complete unreachable.
        """
        os.environ[EXPLODE] = "20"
        orch = self.orchestrator()
        orch.run(recording_experiment, recorded=1)
        orch.run(experiment, n=20, f=2, aggregator=Krum)
        orch.run(experiment, n=30, f=2, aggregator=Krum)
        with self.assertRaises(RunFailed):
            orch.drain()
        self.assertEqual(orch.queued, 0)
        self.assertEqual(orch.get("value")["value"], [1])

    def test_failed_job_is_retried_when_enqueued_again(self) -> None:
        """Once its cause is fixed, re-enqueueing runs the failed job."""
        os.environ[EXPLODE] = "20"
        orch = self.orchestrator()
        key = orch.run(experiment, n=20, f=2, aggregator=Krum)
        with self.assertRaises(RunFailed):
            orch.drain()
        self.assertEqual(orch.store.folder_for(key).status, "failed")
        os.environ.pop(EXPLODE, None)
        again = self.orchestrator()
        again.run(experiment, n=20, f=2, aggregator=Krum)
        self.assertEqual(again.drain().count("done"), 1)
        self.assertTrue(again.store.folder_for(key).done)
        self.assertIsNone(again.store.folder_for(key).traceback())

    def test_interrupt_promotes_nothing(self) -> None:
        """An interrupted job leaves no marker, so it is run again next time."""
        orch = self.orchestrator()
        key = orch.run(interrupted_experiment, recorded=1)
        with self.assertRaises(KeyboardInterrupt):
            orch.drain()
        self.assertEqual(orch.store.folder_for(key).status, "absent")
        self.assertEqual(orch.store.prune_staging(), 0)

    def test_summary_reports_each_run(self) -> None:
        """The summary renders a line per run for a human to read."""
        orch = self.orchestrator()
        orch.run(experiment, n=10, f=2, aggregator=Krum)
        report = orch.drain().report()
        self.assertIn("1 done", report)
        self.assertIn("done", report.splitlines()[1])


class ProvenanceTest(OrchestratorTestCase):
    """Test what is recorded about the code that produced a result."""

    def test_commit_is_recorded_when_run_from_a_repository(self) -> None:
        """Run from a repository, a job records the commit it ran at."""
        repository = Path(__file__).resolve().parents[2]
        orch = Orchestrator(self.root, source=repository)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            key = orch.run(experiment, n=10, f=2, aggregator=Krum)
            orch.drain()
        git = orch.store.folder_for(key).manifest()["git"]
        self.assertEqual(len(git["commit"]), 40)
        self.assertIn("dirty", git)

    def test_dirty_tree_warns_once(self) -> None:
        """Results produced from a modified tree are flagged, once per sweep."""
        repository = Path(__file__).resolve().parents[2]
        orch = Orchestrator(self.root, source=repository)
        orch.run(experiment, n=10, f=2, aggregator=Krum)
        orch.run(experiment, n=20, f=2, aggregator=Krum)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            orch.drain()
        dirty = [entry for entry in caught if "dirty working tree" in str(entry.message)]
        # The repository is clean in CI and dirty while developing; either way,
        # the warning must not be emitted more than once.
        self.assertLessEqual(len(dirty), 1)

    def test_no_commit_outside_a_repository(self) -> None:
        """Run outside a repository, a job records no commit and does not warn."""
        orch = self.orchestrator()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            key = orch.run(experiment, n=10, f=2, aggregator=Krum)
            orch.drain()
        self.assertIsNone(orch.store.folder_for(key).manifest()["git"])


if __name__ == "__main__":
    unittest.main()
