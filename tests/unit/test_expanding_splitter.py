"""Tests for expanding and rolling window splitters."""

import numpy as np
import pytest

from wavecast.core.exceptions import ConfigError
from wavecast.core.types import TimeSeries
from wavecast.experiments.splitter import expanding_window_split, rolling_window_split


def _make_ts(ticker: str, n: int = 1000) -> TimeSeries:
    """Create a TimeSeries with daily timestamps."""
    timestamps = np.arange("2020-01-01", n, dtype="datetime64[D]")[:n]
    rng = np.random.default_rng(42)
    values = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    return TimeSeries(
        values=values,
        timestamps=timestamps,
        ticker=ticker,
        interval="1d",
    )


class TestExpandingWindowSplit:
    def test_correct_number_of_splits(self):
        series = {"AAPL": _make_ts("AAPL", n=100)}
        # initial_train_end=50, test_window=10, step=10
        # splits at: train[0:50]/test[50:60], train[0:60]/test[60:70],
        #            train[0:70]/test[70:80], train[0:80]/test[80:90],
        #            train[0:90]/test[90:100]
        splits = expanding_window_split(series, 50, 10, 10)
        assert len(splits) == 5

    def test_train_grows_each_split(self):
        series = {"AAPL": _make_ts("AAPL", n=100)}
        splits = expanding_window_split(series, 50, 10, 10)
        train_sizes = [len(train["AAPL"].values) for train, _ in splits]
        # Train should grow: 50, 60, 70, 80, 90
        assert train_sizes == [50, 60, 70, 80, 90]

    def test_test_window_fixed_size(self):
        series = {"AAPL": _make_ts("AAPL", n=100)}
        splits = expanding_window_split(series, 50, 10, 10)
        for _, test in splits:
            assert len(test["AAPL"].values) == 10

    def test_no_overlap_between_train_and_test(self):
        series = {"AAPL": _make_ts("AAPL", n=100)}
        splits = expanding_window_split(series, 50, 10, 10)
        for train, test in splits:
            train_ts = train["AAPL"].timestamps
            test_ts = test["AAPL"].timestamps
            assert train_ts[-1] < test_ts[0]

    def test_full_coverage(self):
        series = {"AAPL": _make_ts("AAPL", n=100)}
        splits = expanding_window_split(series, 50, 10, 10)
        # Last split test should reach the end
        _, last_test = splits[-1]
        last_test_end = last_test["AAPL"].timestamps[-1]
        original_end = series["AAPL"].timestamps[-1]
        assert last_test_end == original_end

    def test_multiple_tickers(self):
        series = {
            "AAPL": _make_ts("AAPL", n=100),
            "MSFT": _make_ts("MSFT", n=100),
        }
        splits = expanding_window_split(series, 50, 10, 10)
        for train, test in splits:
            assert set(train.keys()) == {"AAPL", "MSFT"}
            assert set(test.keys()) == {"AAPL", "MSFT"}

    def test_data_too_short_raises(self):
        series = {"AAPL": _make_ts("AAPL", n=20)}
        with pytest.raises(ConfigError, match="Data too short"):
            expanding_window_split(series, 15, 10, 5)

    def test_empty_series_raises(self):
        with pytest.raises(ConfigError, match="must not be empty"):
            expanding_window_split({}, 50, 10, 10)

    def test_invalid_params_raise(self):
        series = {"AAPL": _make_ts("AAPL", n=100)}
        with pytest.raises(ConfigError, match="initial_train_end must be > 0"):
            expanding_window_split(series, 0, 10, 10)
        with pytest.raises(ConfigError, match="test_window_size must be > 0"):
            expanding_window_split(series, 50, 0, 10)
        with pytest.raises(ConfigError, match="step_size must be > 0"):
            expanding_window_split(series, 50, 10, 0)

    def test_step_larger_than_test_window(self):
        series = {"AAPL": _make_ts("AAPL", n=100)}
        splits = expanding_window_split(series, 50, 10, 20)
        # 50+10<=100 -> split 1, 70+10<=100 -> split 2, 90+10<=100 -> split 3
        assert len(splits) == 3

    def test_preserves_metadata(self):
        series = {"AAPL": _make_ts("AAPL", n=100)}
        splits = expanding_window_split(series, 50, 10, 10)
        for train, test in splits:
            assert train["AAPL"].ticker == "AAPL"
            assert train["AAPL"].interval == "1d"
            assert test["AAPL"].ticker == "AAPL"
            assert test["AAPL"].interval == "1d"

    def test_values_timestamps_aligned(self):
        series = {"AAPL": _make_ts("AAPL", n=100)}
        splits = expanding_window_split(series, 50, 10, 10)
        for train, test in splits:
            assert len(train["AAPL"].values) == len(train["AAPL"].timestamps)
            assert len(test["AAPL"].values) == len(test["AAPL"].timestamps)


class TestRollingWindowSplit:
    def test_correct_number_of_splits(self):
        series = {"AAPL": _make_ts("AAPL", n=100)}
        # train=40, test=10, step=10 -> offsets: 0,10,20,30,40,50
        # 0+40+10=50<=100, ..., 50+40+10=100<=100 -> 6 splits
        splits = rolling_window_split(series, 40, 10, 10)
        assert len(splits) == 6

    def test_fixed_train_window_size(self):
        series = {"AAPL": _make_ts("AAPL", n=100)}
        splits = rolling_window_split(series, 40, 10, 10)
        for train, _ in splits:
            assert len(train["AAPL"].values) == 40

    def test_fixed_test_window_size(self):
        series = {"AAPL": _make_ts("AAPL", n=100)}
        splits = rolling_window_split(series, 40, 10, 10)
        for _, test in splits:
            assert len(test["AAPL"].values) == 10

    def test_train_slides_forward(self):
        series = {"AAPL": _make_ts("AAPL", n=100)}
        splits = rolling_window_split(series, 40, 10, 10)
        starts = [train["AAPL"].timestamps[0] for train, _ in splits]
        # Each start should be 10 days later
        for i in range(1, len(starts)):
            diff = (starts[i] - starts[i - 1]).astype(int)
            assert diff == 10

    def test_no_overlap_between_train_and_test(self):
        series = {"AAPL": _make_ts("AAPL", n=100)}
        splits = rolling_window_split(series, 40, 10, 10)
        for train, test in splits:
            train_ts = train["AAPL"].timestamps
            test_ts = test["AAPL"].timestamps
            assert train_ts[-1] < test_ts[0]

    def test_data_too_short_raises(self):
        series = {"AAPL": _make_ts("AAPL", n=20)}
        with pytest.raises(ConfigError, match="Data too short"):
            rolling_window_split(series, 15, 10, 5)

    def test_empty_series_raises(self):
        with pytest.raises(ConfigError, match="must not be empty"):
            rolling_window_split({}, 40, 10, 10)

    def test_invalid_params_raise(self):
        series = {"AAPL": _make_ts("AAPL", n=100)}
        with pytest.raises(ConfigError, match="train_window_size must be > 0"):
            rolling_window_split(series, 0, 10, 10)
        with pytest.raises(ConfigError, match="test_window_size must be > 0"):
            rolling_window_split(series, 40, 0, 10)
        with pytest.raises(ConfigError, match="step_size must be > 0"):
            rolling_window_split(series, 40, 10, 0)

    def test_multiple_tickers(self):
        series = {
            "AAPL": _make_ts("AAPL", n=100),
            "MSFT": _make_ts("MSFT", n=100),
        }
        splits = rolling_window_split(series, 40, 10, 10)
        for train, test in splits:
            assert set(train.keys()) == {"AAPL", "MSFT"}
            assert set(test.keys()) == {"AAPL", "MSFT"}

    def test_preserves_metadata(self):
        series = {"AAPL": _make_ts("AAPL", n=100)}
        splits = rolling_window_split(series, 40, 10, 10)
        for train, test in splits:
            assert train["AAPL"].ticker == "AAPL"
            assert test["AAPL"].interval == "1d"
