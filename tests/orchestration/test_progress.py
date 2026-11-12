"""Tests for the live progress display."""

import io
import unittest

from krum.orchestration.progress import NullProgress, PlainProgress, make_progress


class FakeTTY(io.StringIO):
    """A string stream that claims to be an interactive terminal."""

    def isatty(self) -> bool:
        """Claim an interactive terminal so the plain renderer is exercised."""
        return True


class ProgressTest(unittest.TestCase):
    """Exercise the progress display builders and the plain renderer."""

    def test_disabled_returns_null(self) -> None:
        """A disabled display is a no-op."""
        self.assertIsInstance(make_progress(10, 10, enabled=False), NullProgress)

    def test_non_terminal_returns_null(self) -> None:
        """A non-terminal stream gets a no-op, not escape codes."""
        self.assertIsInstance(make_progress(10, 10, stream=io.StringIO()), NullProgress)

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


if __name__ == "__main__":
    unittest.main()
