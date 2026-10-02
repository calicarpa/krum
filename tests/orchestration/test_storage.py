"""Tests for the on-disk job folder protocol."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from krum.orchestration.storage import (
    DONE,
    FAILED,
    STAGING,
    JobFolder,
    JobStore,
    bind_job,
    build_manifest,
    current_job,
    display_value,
    encode_params,
    encode_value,
    environment_fingerprint,
    find_lock,
    git_provenance,
    hash_file,
    metric_filename,
    metric_name,
    read_manifest_params,
)

REPO = Path(__file__).resolve().parents[2]


class Aggregator:
    """A stand-in for a parameter passed as a class."""


def helper() -> None:
    """A stand-in for a parameter passed as a function."""


class EncodingTest(unittest.TestCase):
    """Test how parameter values are recorded and displayed."""

    def test_scalars_pass_through(self) -> None:
        """Scalars are recorded as themselves."""
        for value in (None, True, 10, 1.5, "krum"):
            self.assertEqual(encode_value(value), value)

    def test_class_keeps_location_and_name(self) -> None:
        """A class keeps its full location for tracing and a short name to read."""
        encoded = encode_value(Aggregator)
        self.assertEqual(encoded["$kind"], "class")
        self.assertEqual(encoded["name"], "Aggregator")
        self.assertTrue(encoded["location"].endswith(".Aggregator"))
        self.assertEqual(display_value(encoded), "Aggregator")

    def test_callable_keeps_location_and_name(self) -> None:
        """A function is recorded like a class."""
        encoded = encode_value(helper)
        self.assertEqual(encoded["$kind"], "callable")
        self.assertEqual(display_value(encoded), "helper")

    def test_containers_recurse(self) -> None:
        """Lists and mappings are encoded element-wise."""
        self.assertEqual(encode_value([1, Aggregator])[1]["name"], "Aggregator")
        self.assertEqual(encode_value({"a": Aggregator})["a"]["name"], "Aggregator")

    def test_opaque_value_is_recorded_by_repr(self) -> None:
        """A value with no better encoding keeps its repr."""
        encoded = encode_value(object())
        self.assertEqual(encoded["$kind"], "opaque")
        self.assertIn("object", encoded["name"])

    def test_encoding_is_json_serializable(self) -> None:
        """Whatever the parameters, the encoding can be written as JSON."""
        encoded = encode_params({"n": 10, "aggregator": Aggregator, "opaque": object()})
        self.assertIsInstance(json.dumps(encoded), str)

    def test_encoded_params_keep_their_order(self) -> None:
        """Parameters keep signature order, which is the column order read back."""
        encoded = encode_params({"n": 1, "f": 2, "seed": 3})
        self.assertEqual(list(encoded), ["n", "f", "seed"])

    def test_manifest_params_render_one_cell_each(self) -> None:
        """Reading a manifest's parameters yields one value per parameter."""
        manifest = {"params": encode_params({"n": 10, "aggregator": Aggregator})}
        self.assertEqual(read_manifest_params(manifest), {"n": 10, "aggregator": "Aggregator"})


class MetricFilenameTest(unittest.TestCase):
    """Test the metric name to file name mapping."""

    def test_round_trips_unrestricted_names(self) -> None:
        """Names with spaces and separators survive the round trip."""
        for name in ("loss", "test accuracy", "loss/train", "a.b", "100%", "éval"):
            self.assertEqual(metric_name(metric_filename(name)), name)

    def test_distinct_names_give_distinct_files(self) -> None:
        """Encoding is injective, so two names never share a file."""
        names = ("a b", "a%20b", "a/b", "a%2Fb")
        self.assertEqual(len({metric_filename(name) for name in names}), len(names))


class EnvironmentTest(unittest.TestCase):
    """Test the environment fingerprint and the git provenance."""

    def test_find_lock_walks_up_to_the_project(self) -> None:
        """The lock file is found from a directory inside the project."""
        self.assertEqual(find_lock(Path(__file__).parent), REPO / "uv.lock")

    def test_find_lock_returns_none_outside_a_project(self) -> None:
        """A directory with no lock above it yields None."""
        with TemporaryDirectory() as directory:
            self.assertIsNone(find_lock(directory))

    def test_hash_file_is_deterministic_and_content_sensitive(self) -> None:
        """Hashing a file depends on its contents, not its name."""
        with TemporaryDirectory() as directory:
            first = Path(directory) / "a"
            second = Path(directory) / "b"
            first.write_text("same")
            second.write_text("same")
            self.assertEqual(hash_file(first), hash_file(second))
            second.write_text("different")
            self.assertNotEqual(hash_file(first), hash_file(second))

    def test_fingerprint_records_interpreter_and_lock(self) -> None:
        """The fingerprint carries the interpreter and the resolved lock hash."""
        fingerprint = environment_fingerprint(REPO / "uv.lock")
        self.assertIn("python", fingerprint)
        self.assertEqual(fingerprint["uv_lock"]["hash"], hash_file(REPO / "uv.lock"))

    def test_fingerprint_tolerates_a_missing_lock(self) -> None:
        """With no lock file to be found, the entry is None rather than an error."""
        with TemporaryDirectory() as directory:
            self.assertIsNone(environment_fingerprint(start=directory)["uv_lock"])
            self.assertIsNone(environment_fingerprint(Path(directory) / "absent.lock")["uv_lock"])

    def test_provenance_records_the_commit(self) -> None:
        """Inside a repository, the commit is recorded."""
        provenance = git_provenance(REPO)
        assert provenance is not None, "the project itself is a git repository"
        self.assertEqual(len(provenance["commit"]), 40)

    def test_provenance_is_none_outside_a_repository(self) -> None:
        """Outside a repository, provenance is absent rather than an error."""
        with TemporaryDirectory() as directory:
            self.assertIsNone(git_provenance(directory))


class JobFolderTest(unittest.TestCase):
    """Test what a job folder reports about itself."""

    def test_status_is_read_from_the_markers(self) -> None:
        """Status comes from marker files, not from a mutable field."""
        with TemporaryDirectory() as directory:
            folder = JobFolder(Path(directory) / "job")
            self.assertEqual(folder.status, "absent")
            self.assertFalse(folder.done)
            folder.path.mkdir()
            self.assertEqual(folder.status, "absent")
            (folder.path / DONE).write_text("")
            self.assertEqual(folder.status, "done")
            self.assertTrue(folder.done)

    def test_failed_marker_carries_the_traceback(self) -> None:
        """A failed job's marker holds its traceback."""
        with TemporaryDirectory() as directory:
            folder = JobFolder(Path(directory) / "job")
            folder.path.mkdir()
            (folder.path / FAILED).write_text("Traceback ...")
            self.assertEqual(folder.status, "failed")
            self.assertEqual(folder.traceback(), "Traceback ...")

    def test_metric_names_are_absent_without_a_metrics_directory(self) -> None:
        """A job that recorded nothing reports no metrics."""
        with TemporaryDirectory() as directory:
            self.assertEqual(JobFolder(Path(directory) / "job").metric_names(), [])


class JobStoreTest(unittest.TestCase):
    """Test the store, and the staging then promotion protocol."""

    def setUp(self) -> None:
        """Open a store in a temporary directory."""
        self._directory = TemporaryDirectory()
        self.root = Path(self._directory.name) / "study"
        self.store = JobStore(self.root)

    def tearDown(self) -> None:
        """Remove the temporary directory."""
        self._directory.cleanup()

    def writer(self, key: str = "abc"):
        """Start a job with a minimal manifest."""
        return self.store.writer(key, build_manifest(key, helper, {}, start=self._directory.name))

    def test_root_is_created(self) -> None:
        """Opening a store creates its root."""
        self.assertTrue(self.root.is_dir())

    def test_key_is_hex_for_bytes(self) -> None:
        """A binary key names its folder in hex."""
        self.assertEqual(self.store.name_for(b"\x01\xff"), "01ff")
        self.assertEqual(self.store.name_for("already"), "already")

    def test_nothing_is_visible_before_promotion(self) -> None:
        """A job under construction is not at its final path."""
        writer = self.writer()
        writer.metric_path("loss", "float").write_text("step,value\n0,1.0\n")
        self.assertEqual(self.store.folder_for("abc").status, "absent")
        self.assertTrue(writer.path.is_dir())

    def test_finishing_promotes_and_marks_done(self) -> None:
        """Finishing a job renames it into place with its marker."""
        writer = self.writer()
        writer.metric_path("loss", "float").write_text("step,value\n0,1.0\n")
        folder = writer.finish("done")
        self.assertTrue(folder.done)
        self.assertFalse(writer.path.exists())
        self.assertEqual(folder.metric_names(), ["loss"])
        self.assertEqual(folder.manifest()["status"], "done")

    def test_manifest_records_the_metrics_and_timings(self) -> None:
        """The manifest lists the metrics written and how long the job took."""
        writer = self.writer()
        writer.metric_path("test accuracy", "float").write_text("step,value\n")
        manifest = writer.finish("done").manifest()
        self.assertIn("test accuracy", manifest["metrics"])
        self.assertIn("seconds", manifest["timings"])
        self.assertIn("finished", manifest["timings"])

    def test_deps_holds_the_environment_fingerprint(self) -> None:
        """The fingerprint is written beside the manifest."""
        deps = self.writer().finish("done").deps()
        self.assertIn("python", deps)

    def test_failing_promotes_with_the_traceback(self) -> None:
        """A failed job is promoted too, so that it can be inspected."""
        folder = self.writer().finish("failed", "boom")
        self.assertEqual(folder.status, "failed")
        self.assertEqual(folder.traceback(), "boom")

    def test_promotion_replaces_an_earlier_attempt(self) -> None:
        """Re-running a job replaces its folder rather than merging into it."""
        first = self.writer()
        first.metric_path("loss", "float").write_text("step,value\n0,1.0\n")
        first.finish("failed", "boom")
        second = self.writer()
        second.metric_path("accuracy", "float").write_text("step,value\n0,1.0\n")
        folder = second.finish("done")
        self.assertTrue(folder.done)
        self.assertIsNone(folder.traceback())
        self.assertEqual(folder.metric_names(), ["accuracy"])

    def test_discard_leaves_no_trace(self) -> None:
        """Discarding a staged job promotes nothing, as after an interrupt."""
        writer = self.writer()
        writer.discard()
        self.assertEqual(self.store.folder_for("abc").status, "absent")
        self.assertFalse(writer.path.exists())

    def test_iteration_skips_the_staging_area(self) -> None:
        """The staging directory is not a job."""
        self.writer()  # creates the staging area
        self.store.writer("def", build_manifest("def", helper, {})).finish("done")
        self.assertEqual([folder.key for folder in self.store], ["def"])
        self.assertTrue((self.root / STAGING).is_dir())

    def test_done_filters_on_the_marker(self) -> None:
        """Only successfully completed jobs are iterated by `done`."""
        self.store.writer("aaa", build_manifest("aaa", helper, {})).finish("done")
        self.store.writer("bbb", build_manifest("bbb", helper, {})).finish("failed", "boom")
        self.assertEqual([folder.key for folder in self.store.done()], ["aaa"])

    def test_metric_names_span_completed_jobs(self) -> None:
        """Metric names are gathered across completed jobs only."""
        first = self.store.writer("aaa", build_manifest("aaa", helper, {}))
        first.metric_path("loss", "float").write_text("")
        first.finish("done")
        second = self.store.writer("bbb", build_manifest("bbb", helper, {}))
        second.metric_path("hidden", "float").write_text("")
        second.finish("failed", "boom")
        self.assertEqual(self.store.metric_names(), ["loss"])

    def test_prune_staging_removes_leftovers(self) -> None:
        """Leftover staging directories from crashed jobs can be pruned."""
        self.writer()
        self.assertEqual(self.store.prune_staging(), 1)
        self.assertEqual(self.store.prune_staging(), 0)


class CurrentJobTest(unittest.TestCase):
    """Test the context binding a running job."""

    def test_no_job_by_default(self) -> None:
        """Outside a run, there is no current job."""
        self.assertIsNone(current_job())

    def test_binding_is_scoped_and_restored(self) -> None:
        """A bound job is current only inside its block, nesting included."""
        with TemporaryDirectory() as directory:
            store = JobStore(Path(directory) / "study")
            outer = store.writer("aaa", build_manifest("aaa", helper, {}))
            inner = store.writer("bbb", build_manifest("bbb", helper, {}))
            with bind_job(outer):
                self.assertIs(current_job(), outer)
                with bind_job(inner):
                    self.assertIs(current_job(), inner)
                self.assertIs(current_job(), outer)
            self.assertIsNone(current_job())


if __name__ == "__main__":
    unittest.main()
