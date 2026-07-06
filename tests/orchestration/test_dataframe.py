"""Tests for the MetricDataFrame storage class."""

import unittest

import pandas as pd

from krum.orchestration._frozendict import FrozenDict
from krum.orchestration.dataframe import MetricDataFrame


class MetricDataFrameTest(unittest.TestCase):
    """Test MetricDataFrame storage, filtering, and rendering."""

    def test_empty_frame_has_step_and_value_columns(self) -> None:
        """A frame with no samples has step/value columns and no rows."""
        frame = MetricDataFrame(float).to_pandas()
        self.assertEqual(list(frame.columns), ["step", "value"])
        self.assertEqual(len(frame), 0)

    def test_records_appear_as_rows_with_flat_columns(self) -> None:
        """Recorded samples become rows with parameters as ordinary columns."""
        store = MetricDataFrame(float)
        store.record({"n": 10}, 0, 1.5)
        store.record({"n": 10}, 1, 2.5)
        frame = store.to_pandas()
        self.assertEqual(list(frame.columns), ["n", "step", "value"])
        self.assertEqual(list(frame["n"]), [10, 10])
        self.assertEqual(list(frame["value"]), [1.5, 2.5])

    def test_parameter_columns_follow_first_seen_order(self) -> None:
        """The parameter union preserves first-seen name order."""
        store = MetricDataFrame(float)
        store.record({"n": 10, "f": 2}, 0, 1.0)
        store.record({"aggregator": "krum", "n": 20}, 0, 2.0)
        self.assertEqual(
            list(store.to_pandas().columns),
            ["n", "f", "aggregator", "step", "value"],
        )

    def test_params_are_frozen_and_stored_once_per_run(self) -> None:
        """All samples of one run share a single immutable parameter key."""
        store = MetricDataFrame(float)
        for step in range(5):
            store.record({"n": 10, "f": 2}, step, float(step))
        self.assertEqual(len(store._samples), 1)
        self.assertIsInstance(next(iter(store._samples)), FrozenDict)
        self.assertEqual(len(store), 5)

    def test_distinct_param_sets_are_separate_runs(self) -> None:
        """Different parameter sets are stored as separate runs."""
        store = MetricDataFrame(float)
        store.record({"n": 10}, 0, 1.0)
        store.record({"n": 20}, 0, 2.0)
        self.assertEqual(len(store._samples), 2)
        self.assertEqual(len(store), 2)

    def test_parameter_order_does_not_change_run_identity(self) -> None:
        """Mappings with equal items identify the same run."""
        store = MetricDataFrame(float)
        store.record({"n": 10, "f": 2}, 0, 1.0)
        store.record({"f": 2, "n": 10}, 1, 2.0)
        self.assertEqual(len(store._samples), 1)
        self.assertEqual(len(store), 2)

    def test_skip_if_exists_keeps_first_value(self) -> None:
        """skip_if_exists ignores a repeated (run, step), keeping the first value."""
        store = MetricDataFrame(float)
        store.record({"n": 10}, 0, 1.0)
        store.record({"n": 10}, 0, 9.0, skip_if_exists=True)
        self.assertEqual(len(store), 1)
        self.assertEqual(store.to_pandas().iloc[0]["value"], 1.0)

    def test_repeated_step_overwrites_without_skip(self) -> None:
        """Without skip_if_exists a repeated (run, step) overwrites the value."""
        store = MetricDataFrame(float)
        store.record({"n": 10}, 0, 1.0)
        store.record({"n": 10}, 0, 9.0)
        self.assertEqual(len(store), 1)
        self.assertEqual(store.to_pandas().iloc[0]["value"], 9.0)

    def test_filter_returns_narrowed_metric_dataframe(self) -> None:
        """filter() narrows runs and preserves the storage API."""
        store = MetricDataFrame(float)
        store.record({"n": 10, "f": 2}, 0, 1.0)
        store.record({"n": 10, "f": 3}, 0, 2.0)
        store.record({"n": 20, "f": 2}, 0, 3.0)
        narrowed = store.filter(n=10)
        self.assertIsInstance(narrowed, MetricDataFrame)
        self.assertEqual(len(narrowed), 2)
        self.assertEqual(set(narrowed.to_pandas()["value"]), {1.0, 2.0})

    def test_filter_without_arguments_returns_independent_full_store(self) -> None:
        """filter() with no arguments copies every run into a new store."""
        store = MetricDataFrame(float)
        store.record({"n": 10}, 0, 1.0)
        narrowed = store.filter()
        narrowed.record({"n": 10}, 1, 2.0)
        self.assertEqual(len(store), 1)
        self.assertEqual(len(narrowed), 2)

    def test_unknown_filter_returns_empty_store(self) -> None:
        """Filtering on an absent parameter returns no runs."""
        store = MetricDataFrame(float)
        store.record({"n": 10}, 0, 1.0)
        self.assertEqual(len(store.filter(unknown=1)), 0)

    def test_missing_parameter_is_filled_with_pd_na(self) -> None:
        """The parameter union fills absent values without creating an index."""
        store = MetricDataFrame(float)
        store.record({"name": "first"}, 0, 1.0)
        store.record({"name": "second", "extra": "set"}, 0, 2.0)
        frame = store.to_pandas()
        self.assertEqual(list(frame.columns), ["name", "extra", "step", "value"])
        self.assertIs(frame.iloc[0]["extra"], pd.NA)
        self.assertEqual(frame.iloc[1]["extra"], "set")

    def test_to_pandas_returns_a_fresh_frame(self) -> None:
        """Mutating a materialised frame leaves the store intact."""
        store = MetricDataFrame(float)
        store.record({"n": 10}, 0, 1.0)
        frame = store.to_pandas()
        frame["value"] = 999.0
        self.assertEqual(store.to_pandas().iloc[0]["value"], 1.0)

    def test_len_counts_all_samples(self) -> None:
        """len() counts the total number of recorded samples across runs."""
        store = MetricDataFrame(float)
        store.record({"n": 10}, 0, 1.0)
        store.record({"n": 10}, 1, 2.0)
        store.record({"n": 20}, 0, 3.0)
        self.assertEqual(len(store), 3)


if __name__ == "__main__":
    unittest.main()
