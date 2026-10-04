"""Tests for the structure a metric reads back as.

The point of `MetricTable` is that recording and reading metrics costs no
dataframe dependency, so the first thing worth pinning is that importing the
package does not import pandas.
"""

import csv
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from krum.orchestration.metrics import MetricTable, concat, infer_value, parse_value


class ImportTest(unittest.TestCase):
    """The library must not reach for a dataframe of its own accord."""

    def test_importing_the_package_does_not_import_pandas(self) -> None:
        """`import krum.orchestration` leaves pandas alone.

        Checked in a child interpreter, since this one has pandas imported by
        other tests. This is the invariant the whole structure exists for, and
        it holds whether or not pandas is installed.
        """
        done = subprocess.run(
            [sys.executable, "-c", "import krum.orchestration, sys; print('pandas' in sys.modules)"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(done.stdout.strip(), "False")

    def test_the_converter_is_written_against_the_public_accessor(self) -> None:
        """`to_pandas` uses only `to_dict`, so a reader's own converter is equal to it."""
        table = MetricTable({"step": [0, 1], "value": [1.5, 0.5]})
        frame = table.to_pandas()
        self.assertEqual(list(frame.columns), list(table.to_dict()))
        self.assertEqual(frame["value"].tolist(), table["value"])


class MetricTableTest(unittest.TestCase):
    """Test the structure itself."""

    def table(self) -> MetricTable:
        """A small table with a parameter column."""
        return MetricTable({"step": [0, 10], "value": [1.5, 0.5], "n": [10, 10]})

    def test_columns_keep_their_order(self) -> None:
        """Columns are reported in the order they were given."""
        self.assertEqual(self.table().columns, ("step", "value", "n"))

    def test_length_is_the_row_count(self) -> None:
        """A table's length is its rows, not its columns."""
        self.assertEqual(len(self.table()), 2)

    def test_empty_table_has_no_rows(self) -> None:
        """A table with no columns has no rows either."""
        self.assertEqual(len(MetricTable({})), 0)
        self.assertEqual(MetricTable({}).columns, ())

    def test_a_column_reads_back_as_a_list(self) -> None:
        """Indexing by name gives a plain list."""
        self.assertEqual(self.table()["value"], [1.5, 0.5])

    def test_a_column_is_a_copy(self) -> None:
        """Mutating what a column returns does not reach into the table."""
        table = self.table()
        table["value"].append(99.0)
        self.assertEqual(table["value"], [1.5, 0.5])

    def test_an_unknown_column_raises(self) -> None:
        """Asking for a column that is not there raises."""
        with self.assertRaises(KeyError):
            self.table()["nope"]

    def test_membership_is_by_column_name(self) -> None:
        """`in` asks about columns."""
        self.assertIn("value", self.table())
        self.assertNotIn("nope", self.table())

    def test_rows_are_dicts_in_column_order(self) -> None:
        """Each row carries every column, in the table's order."""
        self.assertEqual(
            list(self.table().rows()),
            [{"step": 0, "value": 1.5, "n": 10}, {"step": 10, "value": 0.5, "n": 10}],
        )

    def test_iterating_gives_the_rows(self) -> None:
        """Iteration is row-wise, agreeing with the length."""
        table = self.table()
        self.assertEqual(len(list(table)), len(table))
        self.assertEqual(list(table), list(table.rows()))

    def test_to_dict_is_a_copy(self) -> None:
        """The exported mapping does not alias the table's own lists."""
        table = self.table()
        exported = table.to_dict()
        exported["value"].append(99.0)
        self.assertEqual(table["value"], [1.5, 0.5])

    def test_ragged_columns_are_refused(self) -> None:
        """Columns of differing lengths are not a table."""
        with self.assertRaises(ValueError):
            MetricTable({"step": [0, 1], "value": [1.5]})

    def test_repr_reports_the_shape(self) -> None:
        """The repr says how many rows and which columns."""
        self.assertIn("2 rows", repr(self.table()))

    def test_to_csv_round_trips(self) -> None:
        """Writing a CSV and reading it back gives the same cells."""
        with TemporaryDirectory() as directory:
            path = self.table().to_csv(Path(directory) / "out.csv")
            with path.open(newline="") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["step", "value", "n"])
        self.assertEqual(rows[1], ["0", "1.5", "10"])


class ConcatTest(unittest.TestCase):
    """Test stacking tables whose columns need not agree."""

    def test_stacking_keeps_every_row(self) -> None:
        """Rows from both tables survive, in order."""
        table = concat([MetricTable({"step": [0], "value": [1.0]}), MetricTable({"step": [1], "value": [2.0]})])
        self.assertEqual(table["step"], [0, 1])
        self.assertEqual(table["value"], [1.0, 2.0])

    def test_a_column_missing_from_one_table_is_filled(self) -> None:
        """A parameter only one job carries reads as None for the others.

        Nothing is coerced on the way: the integer stays an integer where it
        was recorded, which a dataframe does not promise once a column has a
        gap in it.
        """
        table = concat([
            MetricTable({"step": [0], "value": [1.0], "n": [10]}),
            MetricTable({"step": [0], "value": [2.0], "n": [10], "extra": [7]}),
        ])
        self.assertEqual(table["extra"], [None, 7])
        self.assertIsInstance(table["extra"][1], int)

    def test_the_documented_column_order_survives(self) -> None:
        """Metric columns lead, parameters follow, the job key is last."""
        table = concat([
            MetricTable({"step": [0], "value": [1.0], "n": [10], "job_key": ["aaa"]}),
            MetricTable({"step": [0], "value": [2.0], "n": [10], "extra": [7], "job_key": ["bbb"]}),
        ])
        self.assertEqual(table.columns, ("step", "value", "n", "extra", "job_key"))

    def test_stacking_nothing_is_an_empty_table(self) -> None:
        """No tables at all is an empty table, not an error."""
        self.assertEqual(len(concat([])), 0)


class ParsingTest(unittest.TestCase):
    """Values are parsed back as what they were recorded as."""

    def test_declared_dtypes_are_honoured(self) -> None:
        """A declared dtype decides the parse, rather than being guessed at."""
        self.assertEqual(parse_value("float")("3"), 3.0)
        self.assertIsInstance(parse_value("float")("3"), float)
        self.assertEqual(parse_value("int")("3"), 3)
        self.assertEqual(parse_value("str")("3"), "3")

    def test_bool_is_not_parsed_by_truthiness(self) -> None:
        """`bool("False")` is True, so a recorded bool is compared, not called."""
        self.assertIs(parse_value("bool")("False"), False)
        self.assertIs(parse_value("bool")("True"), True)

    def test_torch_dtypes_map_to_python_types(self) -> None:
        """A torch dtype reads back as the Python type it stands for."""
        self.assertIsInstance(parse_value("torch.float32")("1"), float)
        self.assertIsInstance(parse_value("torch.int64")("1"), int)

    def test_an_unrecorded_dtype_is_inferred(self) -> None:
        """With nothing recorded, int is tried before float, then the text stands."""
        self.assertEqual(infer_value("3"), 3)
        self.assertEqual(infer_value("3.5"), 3.5)
        self.assertEqual(infer_value("krum"), "krum")
        self.assertIsInstance(parse_value("")("3"), int)


if __name__ == "__main__":
    unittest.main()
