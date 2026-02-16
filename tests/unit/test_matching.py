"""Tests for DTW matching."""

import numpy as np

from wavecast.core.types import MarketLabel, Shapelet
from wavecast.dtw.matching import match_against_library, match_single
from wavecast.shapelets.library import ShapeletLibrary


def test_match_single_identical():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = match_single(a, a, window=5)
    assert result.distance < 1e-10
    assert len(result.warping_path) > 0


def test_match_single_different():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    result = match_single(a, b, window=5)
    assert result.distance > 0


def test_match_against_library():
    shapelets = [
        Shapelet(
            id=f"s{i}",
            coefficients=np.random.default_rng(i).standard_normal(10),
            wavelet_level=1, ticker="T", label=MarketLabel.UP,
            information_gain=0.1, start_index=0, end_index=10, threshold=1.0,
        )
        for i in range(5)
    ]
    lib = ShapeletLibrary(shapelets)
    query = np.random.default_rng(99).standard_normal(10)
    result = match_against_library(query, lib, level=1)
    assert len(result.matches) > 0
    # Should be sorted by distance
    for i in range(len(result.matches) - 1):
        assert result.matches[i].distance <= result.matches[i + 1].distance
