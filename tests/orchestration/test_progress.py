"""Tests for the live progress display."""

import io
import tempfile
import unittest
from pathlib import Path

from krum.orchestration import Metric, Orchestrator
from krum.orchestration.progress import Failure, NullProgress, PlainProgress, make_progress, progress_stream


class FakeTTY(io.StringIO):
    """A string stream that claims to be an interactive terminal."""

    def isatty(self) -> bool:
        """Claim an interactive terminal so the plain renderer is exercised."""
        return True


def record(seed: int) -> None:
    """Write one metric, so the job has something to store."""
    Metric("loss", dtype=float).push(seed, float(seed))


class ProgressTest(unittest.TestCase):
    """Exercise the progress display builders and the plain renderer."""

    def test_no_stream_returns_null(self) -> None:
        """Without a stream there is nowhere to draw, so nothing is drawn."""
        self.assertIsInstance(make_progress(10, 10), NullProgress)

    def test_disabled_stream_returns_none(self) -> None:
        """Turning progress off yields no stream at all."""
        self.assertIsNone(progress_stream(False))

    def test_explicit_stream_is_kept_even_when_not_a_terminal(self) -> None:
        """A stream the caller passed is drawn on, terminal or not."""
        stream = io.StringIO()
        self.assertIs(progress_stream(stream), stream)
        self.assertNotIsInstance(make_progress(10, 10, stream=stream), NullProgress)

    def test_plain_progress_renders_a_line(self) -> None:
        """The plain renderer draws the count, ETA and description."""
        stream = FakeTTY()
        progress = PlainProgress(2, 2, stream=stream)
        progress.advance(ran=True, duration=1.0, description="abc")
        text = stream.getvalue()
        self.assertIn("1/2", text)
        self.assertIn("ETA", text)
        self.assertIn("abc", text)

    def test_plain_progress_closes_with_a_newline(self) -> None:
        """Closing leaves the terminal on a fresh line."""
        stream = FakeTTY()
        progress = PlainProgress(1, 1, stream=stream)
        progress.advance(ran=True, duration=1.0)
        progress.close()
        self.assertTrue(stream.getvalue().endswith("\n"))

    def test_eta_ignores_cached_jobs(self) -> None:
        """A cached job must not drag the estimate down.

        The ETA is built from the durations of the jobs that actually executed,
        so a burst of instant cached jobs leaves it unchanged.
        """
        progress = PlainProgress(4, 2, stream=FakeTTY())
        progress.advance(ran=False, duration=0.0)
        progress.advance(ran=True, duration=10.0)
        self.assertEqual(progress.runs_done, 1)
        eta = progress.eta()
        assert eta is not None
        self.assertAlmostEqual(eta, 10.0)

    def test_rerun_is_tracked_separately(self) -> None:
        """A retried job counts as both run and rerun."""
        progress = PlainProgress(3, 2, stream=FakeTTY())
        progress.advance(ran=True, rerun=True, duration=1.0)
        progress.advance(ran=True, duration=1.0)
        progress.advance(ran=False)
        self.assertEqual(progress.runs_done, 2)
        self.assertEqual(progress.reruns, 1)
        self.assertEqual(progress.cached, 1)

    def test_failure_reports_the_job_that_stopped_the_drain(self) -> None:
        """The closing report names the job, its parameters and its traceback."""
        stream = FakeTTY()
        progress = PlainProgress(3, 1, stream=stream)
        failure = Failure(
            key="abc123",
            params={"seed": 2, "lr": 0.1},
            error="Traceback (most recent call last):\nRuntimeError: boom",
            path=Path("/tmp/store/abc123"),
        )
        progress.fail(failure)
        progress.close()
        text = stream.getvalue()
        self.assertIn("job abc123 failed", text)
        self.assertIn("seed=2 · lr=0.1", text)
        self.assertIn("/tmp/store/abc123", text)
        self.assertIn("RuntimeError: boom", text)
        self.assertEqual(progress.failed, 1)

    def test_an_injected_renderer_receives_the_drain(self) -> None:
        """A renderer passed to the orchestrator is the one that is driven."""
        renderer = PlainProgress(1, 1, stream=FakeTTY())
        orchestrator = Orchestrator(Path(tempfile.mkdtemp(prefix="krum-progress-inject-")), progress=renderer)
        orchestrator.run(record, seed=1)
        orchestrator.drain()
        self.assertEqual(renderer.runs_done, 1)
        self.assertEqual(renderer.completed, 1)


if __name__ == "__main__":
    unittest.main()
