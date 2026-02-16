"""Tests for ShapeDTW."""

import numpy as np

from wavecast.dtw.shape_dtw import shape_descriptor, shape_dtw_distance


def test_slope_descriptor():
    series = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
    desc = shape_descriptor(series, descriptor="slope")
    # Slope descriptor may be len-1 (np.diff) or same length (padded)
    assert len(desc) >= len(series) - 1


def test_derivative_descriptor():
    series = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
    desc = shape_descriptor(series, descriptor="derivative")
    assert len(desc) >= len(series) - 1


def test_shape_dtw_distance_identical():
    s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    d = shape_dtw_distance(s, s, descriptor="slope", window=3)
    assert d < 1e-10


def test_shape_dtw_distance_different():
    s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    d = shape_dtw_distance(s1, s2, descriptor="slope", window=3)
    assert d > 0
