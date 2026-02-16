"""Tests for wavecast.sax.paa."""

import numpy as np
import pytest

from wavecast.core.exceptions import SAXError
from wavecast.sax.paa import inverse_paa, paa


def test_paa_reduces_length():
    series = np.arange(100, dtype=np.float64)
    result = paa(series, 10)
    assert len(result) == 10


def test_paa_preserves_mean():
    series = np.arange(100, dtype=np.float64)
    result = paa(series, 10)
    np.testing.assert_almost_equal(result.mean(), series.mean(), decimal=1)


def test_paa_single_segment():
    series = np.array([1.0, 2.0, 3.0, 4.0])
    result = paa(series, 1)
    assert len(result) == 1
    np.testing.assert_almost_equal(result[0], 2.5)


def test_paa_same_length():
    series = np.array([1.0, 2.0, 3.0, 4.0])
    result = paa(series, 4)
    np.testing.assert_array_almost_equal(result, series)


def test_paa_clamps_to_series_length():
    series = np.array([1.0, 2.0, 3.0])
    result = paa(series, 10)
    assert len(result) == 3


def test_paa_empty_series():
    result = paa(np.array([], dtype=np.float64), 5)
    assert len(result) == 0


def test_paa_invalid_segments():
    with pytest.raises(SAXError):
        paa(np.array([1.0, 2.0]), 0)


def test_inverse_paa_roundtrip():
    series = np.arange(100, dtype=np.float64)
    compressed = paa(series, 10)
    expanded = inverse_paa(compressed, 100)
    assert len(expanded) == 100
    np.testing.assert_almost_equal(expanded.mean(), series.mean(), decimal=1)


def test_inverse_paa_preserves_values():
    values = np.array([1.0, 5.0, 3.0])
    expanded = inverse_paa(values, 9)
    assert len(expanded) == 9
    # Each value should be repeated 3 times
    np.testing.assert_array_almost_equal(expanded[:3], [1.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(expanded[3:6], [5.0, 5.0, 5.0])
    np.testing.assert_array_almost_equal(expanded[6:9], [3.0, 3.0, 3.0])
