"""Tests for Parquet cache."""

import numpy as np

from wavecast.core.types import TimeSeries
from wavecast.data.cache import ParquetCache


def test_cache_put_get(tmp_path):
    cache = ParquetCache(tmp_path)
    ts = TimeSeries(
        values=np.array([1.0, 2.0, 3.0]),
        timestamps=np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]"),
        ticker="AAPL",
        interval="1d",
    )
    cache.put(ts)
    result = cache.get("AAPL", "1d")
    assert result is not None
    assert result.ticker == "AAPL"
    assert result.length == 3


def test_cache_get_missing(tmp_path):
    cache = ParquetCache(tmp_path)
    result = cache.get("NONEXISTENT", "1d")
    assert result is None


def test_cache_list(tmp_path):
    cache = ParquetCache(tmp_path)
    ts = TimeSeries(
        values=np.array([1.0, 2.0]),
        timestamps=np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"),
        ticker="SPY",
        interval="1d",
    )
    cache.put(ts)
    cached = cache.list_cached()
    assert len(cached) >= 1


def test_cache_clear(tmp_path):
    cache = ParquetCache(tmp_path)
    ts = TimeSeries(
        values=np.array([1.0]),
        timestamps=np.array(["2020-01-01"], dtype="datetime64[D]"),
        ticker="TEST",
        interval="1d",
    )
    cache.put(ts)
    cache.clear()
    result = cache.get("TEST", "1d")
    assert result is None
