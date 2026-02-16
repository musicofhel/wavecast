"""Tests for core types."""

import numpy as np
import pytest

from wavecast.core.types import (
    MarketLabel,
    RegimeType,
    TimeSeries,
    WaveletDecomposition,
)


def test_timeseries_creation():
    ts = TimeSeries(
        values=np.array([1.0, 2.0, 3.0]),
        timestamps=np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]"),
        ticker="TEST",
    )
    assert ts.length == 3
    assert ts.ticker == "TEST"


def test_timeseries_slice():
    ts = TimeSeries(
        values=np.arange(10, dtype=np.float64),
        timestamps=np.arange("2020-01-01", 10, dtype="datetime64[D]"),
        ticker="TEST",
    )
    sliced = ts.slice(2, 5)
    assert sliced.length == 3
    assert sliced.values[0] == 2.0
    assert sliced.ticker == "TEST"


def test_wavelet_decomposition_detail_at_level():
    coeffs = [np.array([1.0]), np.array([2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    decomp = WaveletDecomposition(
        coefficients=coeffs, wavelet="db4", level=2, original_length=6
    )
    # level 2 = coefficients[1], level 1 = coefficients[2]
    np.testing.assert_array_equal(decomp.detail_at_level(2), np.array([2.0, 3.0]))
    np.testing.assert_array_equal(decomp.detail_at_level(1), np.array([4.0, 5.0, 6.0]))


def test_wavelet_decomposition_invalid_level():
    coeffs = [np.array([1.0]), np.array([2.0])]
    decomp = WaveletDecomposition(
        coefficients=coeffs, wavelet="db4", level=1, original_length=2
    )
    with pytest.raises(ValueError):
        decomp.detail_at_level(0)
    with pytest.raises(ValueError):
        decomp.detail_at_level(2)


def test_market_label_values():
    assert MarketLabel.UP.value == "up"
    assert MarketLabel.DOWN.value == "down"
    assert MarketLabel.FLAT.value == "flat"


def test_regime_type_values():
    assert RegimeType.TRENDING.value == "trending"
    assert RegimeType.MEAN_REVERTING.value == "mean_reverting"
    assert RegimeType.RANDOM_WALK.value == "random_walk"
