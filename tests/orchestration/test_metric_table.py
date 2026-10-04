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

from krum.orchestration.metrics import InMemoryTable, MetricTable, concat, infer_value, parse_value


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
        table = InMemoryTable({"step": [0, 1], "value": [1.5, 0.5]})
        frame = table.to_pandas()
        self.assertEqual(list(frame.columns), list(table.to_dict()))
        self.assertEqual(frame["value"].tolist(), table["value"])


class StreamingRows(MetricTable):
    """A table that holds no rows, to check what the interface really demands.

    Stands in for the out-of-core implementation a later release may want: it
    generates its rows on demand and keeps none of them.
    """

    def __init__(self, height: int) -> None:
        """Promise `height` rows, without holding any."""
        self._height = height

    @property
    def columns(self) -> tuple[str, ...]:
        """The two columns it generates."""
        return ("step", "value")

    def __len__(self) -> int:
        """How many rows it will generate."""
        return self._height

    def rows(self):
        """Generate the rows, one at a time."""
        for step in range(self._height):
            yield {"step": step, "value": float(step) / 2}


class InterfaceTest(unittest.TestCase):
    """What an implementation must provide, and what it gets for free."""

    def test_the_interface_cannot_be_instantiated(self) -> None:
        """`MetricTable` is the contract, not a table."""
        with self.assertRaises(TypeError):
            MetricTable()

    def test_only_three_members_are_required(self) -> None:
        """Columns, length and rows are the whole obligation."""
        self.assertEqual(MetricTable.__abstractmethods__, frozenset({"columns", "__len__", "rows"}))

    def test_an_implementation_gets_the_rest_derived(self) -> None:
        """Providing the three gives the rest, including the converters.

        This is what makes a streaming implementation a later possibility
        rather than a rewrite: nothing below is implemented by this class.
        """
        table = StreamingRows(4)
        self.assertIsInstance(table, MetricTable)
        self.assertEqual(len(table), 4)
        self.assertIn("value", table)
        self.assertNotIn("nope", table)
        self.assertEqual(table["value"], [0.0, 0.5, 1.0, 1.5])
        self.assertEqual(table.to_dict(), {"step": [0, 1, 2, 3], "value": [0.0, 0.5, 1.0, 1.5]})
        self.assertEqual(len(list(table)), 4)
        self.assertIn("4 rows", repr(table))
        self.assertEqual(table.to_pandas()["value"].tolist(), [0.0, 0.5, 1.0, 1.5])

    def test_a_derived_unknown_column_still_raises(self) -> None:
        """The derived column accessor reports a missing column the same way."""
        with self.assertRaises(KeyError):
            StreamingRows(2)["nope"]

    def test_the_derived_csv_writer_streams(self) -> None:
        """`to_csv` goes through `rows`, so it works for a table holding nothing."""
        with TemporaryDirectory() as directory:
            path = StreamingRows(3).to_csv(Path(directory) / "out.csv")
            with path.open(newline="") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["step", "value"])
        self.assertEqual(len(rows), 4)

    def test_the_eager_table_is_one_implementation_of_it(self) -> None:
        """`InMemoryTable` is the in-memory implementation, not the interface."""
        self.assertTrue(issubclass(InMemoryTable, MetricTable))
        self.assertIsInstance(InMemoryTable({"step": [0]}), MetricTable)


class MetricTableTest(unittest.TestCase):
    """Test the structure itself."""

    def table(self) -> InMemoryTable:
        """A small table with a parameter column."""
        return InMemoryTable({"step": [0, 10], "value": [1.5, 0.5], "n": [10, 10]})

    def test_columns_keep_their_order(self) -> None:
        """Columns are reported in the order they were given."""
        self.assertEqual(self.table().columns, ("step", "value", "n"))

    def test_length_is_the_row_count(self) -> None:
        """A table's length is its rows, not its columns."""
        self.assertEqual(len(self.table()), 2)

    def test_empty_table_has_no_rows(self) -> None:
        """A table with no columns has no rows either."""
        self.assertEqual(len(InMemoryTable({})), 0)
        self.assertEqual(InMemoryTable({}).columns, ())

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
            InMemoryTable({"step": [0, 1], "value": [1.5]})

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
        table = concat([InMemoryTable({"step": [0], "value": [1.0]}), InMemoryTable({"step": [1], "value": [2.0]})])
        self.assertEqual(table["step"], [0, 1])
        self.assertEqual(table["value"], [1.0, 2.0])

    def test_a_column_missing_from_one_table_is_filled(self) -> None:
        """A parameter only one job carries reads as None for the others.

        Nothing is coerced on the way: the integer stays an integer where it
        was recorded, which a dataframe does not promise once a column has a
        gap in it.
        """
        table = concat([
            InMemoryTable({"step": [0], "value": [1.0], "n": [10]}),
            InMemoryTable({"step": [0], "value": [2.0], "n": [10], "extra": [7]}),
        ])
        self.assertEqual(table["extra"], [None, 7])
        self.assertIsInstance(table["extra"][1], int)

    def test_the_documented_column_order_survives(self) -> None:
        """Metric columns lead, parameters follow, the job key is last."""
        table = concat([
            InMemoryTable({"step": [0], "value": [1.0], "n": [10], "job_key": ["aaa"]}),
            InMemoryTable({"step": [0], "value": [2.0], "n": [10], "extra": [7], "job_key": ["bbb"]}),
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
