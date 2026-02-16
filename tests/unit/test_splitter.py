"""Tests for walk-forward splitter."""

import numpy as np
import pytest

from wavecast.core.exceptions import ConfigError
from wavecast.core.types import TimeSeries
from wavecast.experiments.splitter import walk_forward_split


def _make_ts(ticker: str, start: str = "2020-01-01", n: int = 1000) -> TimeSeries:
    """Create a TimeSeries with daily timestamps."""
    timestamps = np.arange(start, n, dtype="datetime64[D]")[:n]
    rng = np.random.default_rng(42)
    values = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    return TimeSeries(
        values=values,
        timestamps=timestamps,
        ticker=ticker,
        interval="1d",
    )


def test_basic_split():
    series = {"AAPL": _make_ts("AAPL")}
    train, test = walk_forward_split(series, "2021-06-30", "2021-07-01")
    assert "AAPL" in train
    assert "AAPL" in test
    assert len(train["AAPL"].values) > 0
    assert len(test["AAPL"].values) > 0


def test_no_date_overlap():
    series = {"AAPL": _make_ts("AAPL")}
    train, test = walk_forward_split(series, "2021-06-30", "2021-07-01")

    train_max = train["AAPL"].timestamps[-1]
    test_min = test["AAPL"].timestamps[0]
    assert train_max < test_min


def test_proper_boundaries():
    series = {"AAPL": _make_ts("AAPL")}
    train, test = walk_forward_split(series, "2021-12-31", "2022-01-01")

    # All train timestamps <= train_end
    assert np.all(train["AAPL"].timestamps <= np.datetime64("2021-12-31"))
    # All test timestamps >= test_start
    assert np.all(test["AAPL"].timestamps >= np.datetime64("2022-01-01"))


def test_multiple_tickers():
    series = {
        "AAPL": _make_ts("AAPL"),
        "MSFT": _make_ts("MSFT"),
        "SPY": _make_ts("SPY"),
    }
    train, test = walk_forward_split(series, "2021-06-30", "2021-07-01")
    assert set(train.keys()) == {"AAPL", "MSFT", "SPY"}
    assert set(test.keys()) == {"AAPL", "MSFT", "SPY"}


def test_train_end_gte_test_start_raises():
    series = {"AAPL": _make_ts("AAPL")}
    with pytest.raises(ConfigError, match="must be before"):
        walk_forward_split(series, "2022-01-01", "2021-06-30")


def test_same_date_raises():
    series = {"AAPL": _make_ts("AAPL")}
    with pytest.raises(ConfigError, match="must be before"):
        walk_forward_split(series, "2021-06-30", "2021-06-30")


def test_no_train_data_raises():
    # All data after train_end
    series = {"AAPL": _make_ts("AAPL", start="2022-01-01")}
    with pytest.raises(ConfigError, match="No training data"):
        walk_forward_split(series, "2020-12-31", "2021-01-01")


def test_no_test_data_raises():
    # All data before test_start
    series = {"AAPL": _make_ts("AAPL", start="2020-01-01", n=100)}
    with pytest.raises(ConfigError, match="No test data"):
        walk_forward_split(series, "2025-01-01", "2025-01-02")


def test_preserves_metadata():
    ts = _make_ts("AAPL")
    series = {"AAPL": ts}
    train, test = walk_forward_split(series, "2021-06-30", "2021-07-01")

    assert train["AAPL"].ticker == "AAPL"
    assert train["AAPL"].interval == "1d"
    assert test["AAPL"].ticker == "AAPL"
    assert test["AAPL"].interval == "1d"


def test_values_and_timestamps_aligned():
    series = {"AAPL": _make_ts("AAPL")}
    train, test = walk_forward_split(series, "2021-06-30", "2021-07-01")

    assert len(train["AAPL"].values) == len(train["AAPL"].timestamps)
    assert len(test["AAPL"].values) == len(test["AAPL"].timestamps)


def test_total_samples_preserved():
    ts = _make_ts("AAPL")
    series = {"AAPL": ts}
    train, test = walk_forward_split(series, "2021-06-30", "2021-07-01")

    total = len(train["AAPL"].values) + len(test["AAPL"].values)
    # With gap between train_end and test_start, total should equal original
    # since consecutive dates means no gap
    assert total == len(ts.values)
