"""Tests for data preprocessing."""

import numpy as np
import pytest

from wavecast.core.types import MarketLabel, TimeSeries
from wavecast.data.preprocessing import handle_nans, label_returns, log_returns, normalize_zscore


@pytest.fixture
def simple_ts():
    return TimeSeries(
        values=np.array([100.0, 110.0, 105.0, 115.0, 120.0]),
        timestamps=np.arange("2020-01-01", 5, dtype="datetime64[D]"),
        ticker="TEST",
    )


def test_normalize_zscore(simple_ts):
    result = normalize_zscore(simple_ts)
    assert result.ticker == "TEST"
    np.testing.assert_almost_equal(np.mean(result.values), 0.0, decimal=10)
    np.testing.assert_almost_equal(np.std(result.values), 1.0, decimal=10)


def test_log_returns(simple_ts):
    result = log_returns(simple_ts)
    assert result.length == simple_ts.length - 1
    expected_first = np.log(110.0 / 100.0)
    np.testing.assert_almost_equal(result.values[0], expected_first, decimal=10)


def test_handle_nans_ffill():
    ts = TimeSeries(
        values=np.array([1.0, np.nan, 3.0, np.nan, 5.0]),
        timestamps=np.arange("2020-01-01", 5, dtype="datetime64[D]"),
        ticker="TEST",
    )
    result = handle_nans(ts, method="ffill")
    assert not np.any(np.isnan(result.values))
    assert result.values[1] == 1.0  # forward filled


def test_handle_nans_drop():
    ts = TimeSeries(
        values=np.array([1.0, np.nan, 3.0, np.nan, 5.0]),
        timestamps=np.arange("2020-01-01", 5, dtype="datetime64[D]"),
        ticker="TEST",
    )
    result = handle_nans(ts, method="drop")
    assert not np.any(np.isnan(result.values))
    assert result.length == 3


def test_label_returns(simple_ts):
    # label_returns takes prices and computes returns internally
    labels = label_returns(simple_ts)
    assert len(labels) == simple_ts.length - 1
    for label in labels:
        assert label in [MarketLabel.UP, MarketLabel.DOWN, MarketLabel.FLAT]
