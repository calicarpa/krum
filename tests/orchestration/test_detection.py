"""What a change must invalidate, checked the way a researcher would hit it.

Each case writes a small project to a temporary directory, runs its sweep
script in a fresh interpreter, changes one thing, and runs it again. A fresh
interpreter per run is what a researcher does, and it is the only way to
change a compiled extension, which a process cannot unload.

Every case first checks that an untouched rerun is skipped, so that a re-run
after the change is attributable to the change, not to an unstable key.

Impure experiments (network, `/dev/urandom`) are out of scope by design, and
have no case here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import unittest
import zipfile
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

# Appended to every sweep script: one job, reported as JSON on the last line
SWEEP = """
if __name__ == "__main__":
    import json
    from krum.orchestration import Orchestrator
    orch = Orchestrator("out", source=".")
    orch.run(my_exp, x=1)
    summary = orch.drain()
    value = orch.get("m")["value"][0]
    print(json.dumps({"status": summary.outcomes[0].status, "value": value}))
"""

HEADER = "from krum.orchestration import Metric\n"


def compiler() -> str | None:
    """A C compiler, if one is on the path."""
    return shutil.which("cc")


class Project:
    """A researcher's project: a directory holding a sweep script and its helpers."""

    def __init__(self, root: Path) -> None:
        """Lay the project out at `root`."""
        self.root = root
        self.env: dict[str, str] = {}

    def write(self, name: str, text: str) -> Path:
        """Write a file of the project, creating its directory."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def edit(self, name: str, old: str, new: str) -> None:
        """Replace text in a file of the project, which must contain it."""
        path = self.root / name
        text = path.read_text()
        if old not in text:
            raise AssertionError(f"{old!r} is not in {name}")
        path.write_text(text.replace(old, new))

    def run(self) -> dict[str, Any]:
        """Run the sweep script in a fresh interpreter, returning its report.

        Bytecode caching is off: a cached file is trusted when its source keeps
        its size and mtime to the second, which two quick edits easily do.
        """
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **self.env}
        done = subprocess.run(
            [sys.executable, "main.py"], cwd=self.root, env=env, capture_output=True, text=True, check=False
        )
        if done.returncode != 0:
            raise AssertionError(f"the sweep failed:\n{done.stderr}")
        return json.loads(done.stdout.strip().splitlines()[-1])

    def deps(self) -> list[Path]:
        """Every job's recorded fingerprint."""
        return sorted((self.root / "out").glob("*/deps.json"))


class DetectionTestCase(unittest.TestCase):
    """Shared fixture: a project in a temporary directory, outside any repository."""

    def setUp(self) -> None:
        """Create an empty project."""
        self._directory = TemporaryDirectory()
        self.project = Project(Path(self._directory.name).resolve())

    def tearDown(self) -> None:
        """Remove the project."""
        self._directory.cleanup()

    def assertRerunsAfter(self, change: Callable[[], None], before: float, after: float | None = None) -> None:
        """Check that a change, and only a change, re-runs the job.

        Args:
            change: Applies the change to the project.
            before: The value the job records before the change.
            after: The value it records after, if the change affects it.
        """
        first = self.project.run()
        self.assertEqual(first, {"status": "done", "value": before})
        untouched = self.project.run()
        self.assertEqual(untouched["status"], "skipped", "an untouched rerun must be skipped")
        change()
        changed = self.project.run()
        self.assertEqual(changed["status"], "done", "the change did not re-run the job")
        if after is not None:
            self.assertEqual(changed["value"], after)


class ExperimentTest(DetectionTestCase):
    """The function the orchestrator runs."""

    def test_changing_the_body(self) -> None:
        """An edit to the experiment's own body."""
        self.project.write("main.py", HEADER + "def my_exp(x):\n    Metric('m', dtype=float).push(0, x * 2)\n" + SWEEP)
        self.assertRerunsAfter(lambda: self.project.edit("main.py", "x * 2", "x * 30"), 2.0, 30.0)


class HelperTest(DetectionTestCase):
    """Researcher functions the experiment calls."""

    def helper(self, name: str = "utils.py") -> None:
        """Write the helper module every case calls into."""
        self.project.write(name, "def common_stuff(x):\n    return x * 2\n")

    def test_star_import_of_a_sibling_module(self) -> None:
        """`from utils import *`, `utils.py` sitting next to the sweep script."""
        self.helper()
        self.project.write(
            "main.py",
            HEADER + "from utils import *\n"
            "def my_exp(x):\n    Metric('m', dtype=float).push(0, common_stuff(x))\n" + SWEEP,
        )
        self.assertRerunsAfter(lambda: self.project.edit("utils.py", "x * 2", "x * 30"), 2.0, 30.0)

    def test_module_attribute(self) -> None:
        """`import utils; utils.common_stuff(x)`, reached by attribute."""
        self.helper()
        self.project.write(
            "main.py",
            HEADER + "import utils\n"
            "def my_exp(x):\n    Metric('m', dtype=float).push(0, utils.common_stuff(x))\n" + SWEEP,
        )
        self.assertRerunsAfter(lambda: self.project.edit("utils.py", "x * 2", "x * 30"), 2.0, 30.0)

    def test_helper_of_a_helper(self) -> None:
        """A change two local modules away from the experiment."""
        self.project.write("deep.py", "def deeper(x):\n    return x * 2\n")
        self.project.write("utils.py", "from deep import deeper\ndef common_stuff(x):\n    return deeper(x)\n")
        self.project.write(
            "main.py",
            HEADER + "from utils import common_stuff\n"
            "def my_exp(x):\n    Metric('m', dtype=float).push(0, common_stuff(x))\n" + SWEEP,
        )
        self.assertRerunsAfter(lambda: self.project.edit("deep.py", "x * 2", "x * 30"), 2.0, 30.0)

    def test_helper_in_a_package(self) -> None:
        """A helper inside a local package of the project."""
        self.project.write("lab/__init__.py", "")
        self.helper("lab/utils.py")
        self.project.write(
            "main.py",
            HEADER + "from lab.utils import common_stuff\n"
            "def my_exp(x):\n    Metric('m', dtype=float).push(0, common_stuff(x))\n" + SWEEP,
        )
        self.assertRerunsAfter(lambda: self.project.edit("lab/utils.py", "x * 2", "x * 30"), 2.0, 30.0)

    def test_helper_shipped_in_a_zip(self) -> None:
        """Helpers delivered as a `.zip` put on `sys.path`."""
        archive = self.project.root / "deps.zip"

        def pack(factor: int) -> None:
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("utils.py", f"def common_stuff(x):\n    return x * {factor}\n")

        pack(2)
        self.project.write(
            "main.py",
            "import sys\nsys.path.insert(0, 'deps.zip')\n" + HEADER + "from utils import common_stuff\n"
            "def my_exp(x):\n    Metric('m', dtype=float).push(0, common_stuff(x))\n" + SWEEP,
        )
        self.assertRerunsAfter(lambda: pack(30), 2.0, 30.0)

    def test_import_inside_the_body(self) -> None:
        """`import utils` inside the experiment, so not loaded when the job is enqueued."""
        self.helper()
        self.project.write(
            "main.py",
            HEADER + "def my_exp(x):\n    import utils\n"
            "    Metric('m', dtype=float).push(0, utils.common_stuff(x))\n" + SWEEP,
        )
        self.assertRerunsAfter(lambda: self.project.edit("utils.py", "x * 2", "x * 30"), 2.0, 30.0)


@unittest.skipUnless(compiler(), "needs a C compiler")
class NativeTest(DetectionTestCase):
    """Researcher functions compiled into a shared object."""

    def compile(self, source: str, output: str, python: bool = False) -> None:
        """Compile a C file of the project into a shared object."""
        cc = compiler()
        assert cc is not None, "the class is skipped without a compiler"
        command = [cc, "-shared", "-fPIC", "-o", output, source]
        if python:
            command += ["-I", sysconfig.get_paths()["include"]]
            if sys.platform == "darwin":
                command += ["-undefined", "dynamic_lookup"]
        subprocess.run(command, cwd=self.project.root, check=True, capture_output=True)

    def test_extension_module(self) -> None:
        """A CPython extension module next to the sweep script, rebuilt with different code."""
        self.project.write(
            "fmod.c",
            "#include <Python.h>\n"
            "static PyObject* f(PyObject* s, PyObject* a) { return PyFloat_FromDouble(PyFloat_AsDouble(a) * 2); }\n"
            'static PyMethodDef M[] = {{"f", f, METH_O, ""}, {0}};\n'
            'static struct PyModuleDef D = {PyModuleDef_HEAD_INIT, "fmod", 0, -1, M};\n'
            "PyMODINIT_FUNC PyInit_fmod(void) { return PyModule_Create(&D); }\n",
        )
        output = "fmod" + sysconfig.get_config_var("EXT_SUFFIX")
        self.compile("fmod.c", output, python=True)
        self.project.write(
            "main.py",
            HEADER + "from fmod import f\ndef my_exp(x):\n    Metric('m', dtype=float).push(0, f(x))\n" + SWEEP,
        )

        def rebuild() -> None:
            self.project.edit("fmod.c", "* 2", "* 30")
            self.compile("fmod.c", output, python=True)

        self.assertRerunsAfter(rebuild, 2.0, 30.0)

    def test_ctypes_library(self) -> None:
        """A plain shared library loaded through `ctypes`, rebuilt with different code."""
        self.project.write("f.c", "double f(double x) { return x * 2; }\n")
        self.compile("f.c", "libf.so")
        self.project.write(
            "main.py",
            "import ctypes\n" + HEADER + "LIB = ctypes.CDLL('./libf.so')\n"
            "LIB.f.restype = ctypes.c_double\nLIB.f.argtypes = [ctypes.c_double]\n"
            "def my_exp(x):\n    Metric('m', dtype=float).push(0, LIB.f(x))\n" + SWEEP,
        )

        def rebuild() -> None:
            self.project.edit("f.c", "x * 2", "x * 30")
            self.compile("f.c", "libf.so")

        self.assertRerunsAfter(rebuild, 2.0, 30.0)


class GlobalTest(DetectionTestCase):
    """Global variables the experiment, or its helpers, read."""

    def test_constant_in_the_script(self) -> None:
        """A module-level config dict in the sweep script."""
        self.project.write(
            "main.py",
            HEADER + "CONFIG = {'scale': 2}\n"
            "def my_exp(x):\n    Metric('m', dtype=float).push(0, x * CONFIG['scale'])\n" + SWEEP,
        )
        self.assertRerunsAfter(lambda: self.project.edit("main.py", "'scale': 2", "'scale': 30"), 2.0, 30.0)

    def test_constant_in_a_helper_module(self) -> None:
        """A constant a helper reads, the helper's code unchanged."""
        self.project.write("utils.py", "SCALE = 2\ndef common_stuff(x):\n    return x * SCALE\n")
        self.project.write(
            "main.py",
            HEADER + "from utils import common_stuff\n"
            "def my_exp(x):\n    Metric('m', dtype=float).push(0, common_stuff(x))\n" + SWEEP,
        )
        self.assertRerunsAfter(lambda: self.project.edit("utils.py", "SCALE = 2", "SCALE = 30"), 2.0, 30.0)

    def test_constant_read_from_the_environment(self) -> None:
        """A constant computed at import, from an environment variable."""
        self.project.write(
            "main.py",
            "import os\n" + HEADER + "SCALE = int(os.environ.get('SCALE', '2'))\n"
            "def my_exp(x):\n    Metric('m', dtype=float).push(0, x * SCALE)\n" + SWEEP,
        )
        self.assertRerunsAfter(lambda: self.project.env.update(SCALE="30"), 2.0, 30.0)

    def test_constant_of_a_helper_imported_inside_the_body(self) -> None:
        """A constant read by a helper that is only imported while the job runs."""
        self.project.write("utils.py", "SCALE = 2\ndef common_stuff(x):\n    return x * SCALE\n")
        self.project.write(
            "main.py",
            HEADER + "def my_exp(x):\n    from utils import common_stuff\n"
            "    Metric('m', dtype=float).push(0, common_stuff(x))\n" + SWEEP,
        )
        self.assertRerunsAfter(lambda: self.project.edit("utils.py", "SCALE = 2", "SCALE = 30"), 2.0, 30.0)

    def test_constant_changed_after_enqueueing(self) -> None:
        """A global set between `orch.run` and the drain."""
        self.project.write(
            "main.py",
            HEADER + "import os\nCONFIG = {'scale': 0}\n"
            "def my_exp(x):\n    Metric('m', dtype=float).push(0, x * CONFIG['scale'])\n"
            + SWEEP.replace(
                "orch.run(my_exp, x=1)\n",
                "orch.run(my_exp, x=1)\n    CONFIG['scale'] = int(os.environ.get('SCALE', '2'))\n",
            ),
        )
        self.assertRerunsAfter(lambda: self.project.env.update(SCALE="30"), 2.0, 30.0)


class EnvironmentTest(DetectionTestCase):
    """The interpreter and the external libraries."""

    def write_experiment(self) -> None:
        """An experiment that depends on nothing but the environment."""
        self.project.write("main.py", HEADER + "def my_exp(x):\n    Metric('m', dtype=float).push(0, x * 2)\n" + SWEEP)

    def record_python(self, version: str) -> None:
        """Pretend every recorded job ran under another interpreter version."""
        for path in self.project.deps():
            deps = json.loads(path.read_text())
            deps["witnesses"]["python"] = version
            path.write_text(json.dumps(deps))

    def current_python(self, patch: bool = False) -> str:
        """The running interpreter's version, to the minor or to the patch."""
        parts = sys.version_info[:3] if patch else sys.version_info[:2]
        return ".".join(map(str, parts))

    def test_python_minor_version(self) -> None:
        """A job recorded under another minor version of Python."""
        self.write_experiment()
        major, minor = sys.version_info[:2]
        self.assertRerunsAfter(lambda: self.record_python(f"{major}.{minor - 1}"), 2.0, 2.0)

    def test_python_patch_version_is_not_a_change(self) -> None:
        """A patch release keeps bytecode and results; it is not a reason to re-run."""
        self.write_experiment()
        self.assertEqual(self.project.run()["status"], "done")
        self.record_python(self.current_python())
        self.assertEqual(self.project.run()["status"], "skipped")

    def test_lock_file_of_the_project(self) -> None:
        """A dependency bump recorded in the project's `uv.lock`."""
        self.write_experiment()
        self.project.write("uv.lock", 'version = 1\n[[package]]\nname = "torch"\nversion = "2.5.0"\n')
        self.assertRerunsAfter(lambda: self.project.edit("uv.lock", '"2.5.0"', '"2.6.0"'), 2.0, 2.0)

    def test_installed_library_without_a_lock_file(self) -> None:
        """A library upgraded in place, in a project that has no `uv.lock`."""
        site = self.project.root / "site"
        self.project.env["PYTHONPATH"] = str(site)

        def install(version: str) -> None:
            shutil.rmtree(site, ignore_errors=True)
            info = site / f"fakedep-{version}.dist-info"
            info.mkdir(parents=True)
            (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: fakedep\nVersion: {version}\n")

        install("2.0")
        self.project.write(
            "main.py",
            HEADER + "from importlib.metadata import version\n"
            "def my_exp(x):\n    Metric('m', dtype=float).push(0, x * int(version('fakedep').split('.')[0]))\n" + SWEEP,
        )
        self.assertRerunsAfter(lambda: install("30.0"), 2.0, 30.0)
