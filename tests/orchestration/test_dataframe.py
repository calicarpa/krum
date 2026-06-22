"""Tests for the MetricDataFrame storage class."""

import unittest

from krum.orchestration.dataframe import MetricDataFrame


class MetricDataFrameTest(unittest.TestCase):
    """Test MetricDataFrame storage and rendering."""

    def test_empty_frame_has_step_and_value_columns(self) -> None:
        """A frame with no samples has step/value columns and no rows."""
        frame = MetricDataFrame(float).dataframe()
        self.assertEqual(list(frame.columns), ["step", "value"])
        self.assertEqual(len(frame), 0)

    def test_records_appear_as_rows_indexed_by_params(self) -> None:
        """Recorded samples become rows indexed by their parameters."""
        store = MetricDataFrame(float)
        store.record({"n": 10}, 0, 1.5)
        store.record({"n": 10}, 1, 2.5)
        frame = store.dataframe()
        self.assertEqual(list(frame.columns), ["step", "value"])
        self.assertEqual(list(frame.index.names), ["n"])
        self.assertEqual(list(frame["value"]), [1.5, 2.5])

    def test_index_level_order_follows_first_seen(self) -> None:
        """Index levels follow the order parameters were first seen."""
        store = MetricDataFrame(float)
        store.record({"n": 10, "f": 2, "aggregator": "krum"}, 0, 1.0)
        self.assertEqual(
            list(store.dataframe().index.names), ["n", "f", "aggregator"]
        )

    def test_params_stored_once_per_run(self) -> None:
        """All samples of one run share a single stored parameter set."""
        store = MetricDataFrame(float)
        for step in range(5):
            store.record({"n": 10, "f": 2}, step, float(step))
        self.assertEqual(len(store._samples), 1)  # one distinct run
        self.assertEqual(len(store), 5)  # five samples

    def test_distinct_param_sets_are_separate_runs(self) -> None:
        """Different parameter sets are stored as separate runs."""
        store = MetricDataFrame(float)
        store.record({"n": 10}, 0, 1.0)
        store.record({"n": 20}, 0, 2.0)
        self.assertEqual(len(store._samples), 2)
        self.assertEqual(len(store), 2)

    def test_skip_if_exists_keeps_first_value(self) -> None:
        """skip_if_exists ignores a repeated (run, step), keeping the first value."""
        store = MetricDataFrame(float)
        store.record({"n": 10}, 0, 1.0)
        store.record({"n": 10}, 0, 9.0, skip_if_exists=True)
        self.assertEqual(len(store), 1)
        self.assertEqual(store.dataframe().iloc[0]["value"], 1.0)

    def test_repeated_step_overwrites_without_skip(self) -> None:
        """Without skip_if_exists a repeated (run, step) overwrites the value."""
        store = MetricDataFrame(float)
        store.record({"n": 10}, 0, 1.0)
        store.record({"n": 10}, 0, 9.0)
        self.assertEqual(len(store), 1)
        self.assertEqual(store.dataframe().iloc[0]["value"], 9.0)

    def test_call_filters_by_parameter(self) -> None:
        """Calling the frame filters rows by parameter value."""
        store = MetricDataFrame(float)
        store.record({"n": 10, "f": 2}, 0, 1.0)
        store.record({"n": 10, "f": 3}, 0, 2.0)
        store.record({"n": 20, "f": 2}, 0, 3.0)
        sliced = store(n=10)
        self.assertEqual(len(sliced), 2)
        self.assertEqual(set(sliced["value"]), {1.0, 2.0})

    def test_call_without_filters_returns_full_frame(self) -> None:
        """Calling with no filters returns every row."""
        store = MetricDataFrame(float)
        store.record({"n": 10}, 0, 1.0)
        store.record({"n": 20}, 0, 2.0)
        self.assertEqual(len(store()), 2)

    def test_missing_parameter_is_filled_with_nan(self) -> None:
        """A run missing a parameter another run has shows NaN, not an error."""
        store = MetricDataFrame(float)
        store.record({"n": 10}, 0, 1.0)
        store.record({"n": 20, "extra": 7}, 0, 2.0)
        frame = store.dataframe()
        self.assertEqual(list(frame.index.names), ["n", "extra"])
        self.assertTrue(frame.index.get_level_values("extra").isna().any())

    def test_dataframe_is_a_fresh_copy(self) -> None:
        """Each call materialises a new frame; mutating it leaves the store intact."""
        store = MetricDataFrame(float)
        store.record({"n": 10}, 0, 1.0)
        frame = store.dataframe()
        frame["value"] = 999.0
        self.assertEqual(store.dataframe().iloc[0]["value"], 1.0)

    def test_len_counts_all_samples(self) -> None:
        """len() counts the total number of recorded samples across runs."""
        store = MetricDataFrame(float)
        store.record({"n": 10}, 0, 1.0)
        store.record({"n": 10}, 1, 2.0)
        store.record({"n": 20}, 0, 3.0)
        self.assertEqual(len(store), 3)


if __name__ == "__main__":
    unittest.main()
