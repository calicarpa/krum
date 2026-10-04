"""Tests for running each job in an interpreter of its own.

These spawn real processes, so they are deliberately few: the contract worth
pinning is that a job runs elsewhere, that nothing it leaves behind reaches the
next job, and that a child failing or dying is reported rather than lost.
"""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from krum.orchestration import InlineRunner, Orchestrator, RunFailed, SubprocessRunner
from orchestration._isolation_fixture import expire, explode, leak, record_pid


class RunnerTest(unittest.TestCase):
    """Test how a runner is chosen and configured."""

    def test_inline_by_default(self) -> None:
        """Without asking for isolation, jobs run in this process."""
        with TemporaryDirectory() as directory:
            orch = Orchestrator(Path(directory) / "study", source=directory)
            self.assertIsInstance(orch.runner(), InlineRunner)

    def test_isolation_selects_a_subprocess_runner(self) -> None:
        """Asking for isolation runs jobs out of process."""
        with TemporaryDirectory() as directory:
            orch = Orchestrator(Path(directory) / "study", source=directory, isolate=True)
            self.assertIsInstance(orch.runner(), SubprocessRunner)

    def test_worker_count_is_validated(self) -> None:
        """A runner with no workers could never run anything."""
        with self.assertRaises(ValueError):
            SubprocessRunner(workers=0)

    def test_pool_starts_only_when_used(self) -> None:
        """Building a runner spawns nothing."""
        runner = SubprocessRunner()
        self.assertIsNone(runner._executor)
        runner.close()


class IsolationTestCase(unittest.TestCase):
    """Shared fixture: an isolating orchestrator in a temporary directory."""

    def setUp(self) -> None:
        """Open a temporary store outside any repository."""
        self._directory = TemporaryDirectory()
        self.home = Path(self._directory.name)
        self.root = self.home / "study"

    def tearDown(self) -> None:
        """Remove the temporary directory."""
        self._directory.cleanup()

    def orchestrator(self, isolate: bool = True) -> Orchestrator:
        """Build an orchestrator with deterministic provenance."""
        return Orchestrator(self.root, source=self.home, isolate=isolate)


class SubprocessExecutionTest(IsolationTestCase):
    """Test that an isolated job really runs elsewhere, and reports back."""

    def test_job_runs_in_another_process(self) -> None:
        """An isolated job does not run in the orchestrator's process."""
        orch = self.orchestrator()
        orch.run(record_pid, n=1)
        orch.drain()
        self.assertNotEqual(orch.get("pid")["value"], [os.getpid()])

    def test_metrics_recorded_in_the_child_are_promoted(self) -> None:
        """What a child writes ends up in the job's folder and its manifest."""
        orch = self.orchestrator()
        key = orch.run(record_pid, n=1)
        orch.drain()
        folder = orch.store.folder_for(key)
        self.assertTrue(folder.done)
        self.assertEqual(folder.metric_names(), ["pid"])
        self.assertIn("pid", folder.manifest()["metrics"])

    def test_each_job_gets_a_fresh_interpreter(self) -> None:
        """State one job leaves behind does not reach the next.

        Run in one process these counters would read 1, 2, 3; each job having
        its own interpreter, they all read 1.
        """
        orch = self.orchestrator()
        for n in (10, 20, 30):
            orch.run(leak, n=n)
        orch.drain()
        self.assertEqual(sorted(orch.get("seen")["value"]), [1, 1, 1])

    def test_inline_shares_one_interpreter(self) -> None:
        """The contrast: run in process, the same jobs do see each other."""
        orch = self.orchestrator(isolate=False)
        for n in (10, 20, 30):
            orch.run(leak, n=n)
        orch.drain()
        self.assertEqual(sorted(orch.get("seen")["value"]), [1, 2, 3])

    def test_recorded_result_is_still_skipped(self) -> None:
        """Isolation does not change what counts as already done."""
        orch = self.orchestrator()
        orch.run(record_pid, n=1)
        orch.drain()
        again = self.orchestrator()
        again.run(record_pid, n=1)
        self.assertEqual(again.drain().count("skipped"), 1)

    def test_identity_does_not_depend_on_isolation(self) -> None:
        """The same experiment is the same job however it is executed."""
        inline = self.orchestrator(isolate=False).run(record_pid, n=1)
        isolated = self.orchestrator().run(record_pid, n=1)
        self.assertEqual(inline, isolated)


class SubprocessFailureTest(IsolationTestCase):
    """Test what happens when a child fails or never returns."""

    def test_failure_in_a_child_is_recorded(self) -> None:
        """A child's traceback crosses back and lands in the FAILED marker."""
        orch = self.orchestrator()
        key = orch.run(explode, n=20)
        with self.assertRaises(RunFailed):
            orch.drain()
        folder = orch.store.folder_for(key)
        self.assertEqual(folder.status, "failed")
        recorded = folder.traceback()
        assert recorded is not None, "a failed job records its traceback"
        self.assertIn("n=20 is not supported", recorded)

    def test_a_child_that_dies_is_reported_as_a_failure(self) -> None:
        """A child leaving without returning fails its job rather than the sweep."""
        orch = self.orchestrator()
        key = orch.run(expire, n=1)
        with self.assertRaises(RunFailed) as caught:
            orch.drain()
        self.assertEqual(orch.store.folder_for(key).status, "failed")
        self.assertEqual(caught.exception.summary.count("failed"), 1)

    def test_the_pool_recovers_after_a_child_dies(self) -> None:
        """A dead child does not take the rest of the sweep with it."""
        orch = self.orchestrator()
        orch.run(expire, n=1)
        with self.assertRaises(RunFailed):
            orch.drain()
        again = self.orchestrator()
        again.run(record_pid, n=1)
        self.assertEqual(again.drain().count("done"), 1)

    def test_an_unsendable_experiment_is_explained(self) -> None:
        """An experiment a child cannot import says so, naming the requirement."""

        def local_experiment(n) -> None:
            """Defined inside a method, so it cannot be pickled by name."""

        orch = self.orchestrator()
        orch.run(local_experiment, n=1)
        with self.assertRaises(RuntimeError) as caught:
            orch.drain()
        self.assertIn("picklable", str(caught.exception))
        self.assertEqual(list(orch.store), [])


if __name__ == "__main__":
    unittest.main()
