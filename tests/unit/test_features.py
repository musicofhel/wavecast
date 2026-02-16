"""Tests for feature extraction modules."""

import numpy as np

from wavecast.core.types import (
    HurstResult,
    MatchResult,
    MFDFAResult,
    RegimeType,
    SelfSimilarityResult,
    ShapeletMatch,
)
from wavecast.features import fractal_features, market_features, shapelet_features, wavelet_features
from wavecast.wavelets.dwt import decompose


def test_wavelet_features_extract(sine_series):
    decomp = decompose(sine_series, level=3)
    result = wavelet_features.extract(decomp)
    assert result.ndim == 1
    assert len(result) > 0
    assert np.all(np.isfinite(result))


def test_shapelet_features_with_none():
    result = shapelet_features.extract(None)
    assert result.ndim == 1
    assert np.all(result == 0)


def test_shapelet_features_with_matches():
    matches = [
        ShapeletMatch(
            shapelet_id=f"s{i}", distance=float(i),
            normalized_distance=float(i) / 5, warping_path=[(0, 0)],
            query_start=0, query_end=10,
        )
        for i in range(3)
    ]
    mr = MatchResult(ticker="T", wavelet_level=1, matches=matches, query_length=10)
    result = shapelet_features.extract(mr)
    assert result.ndim == 1
    assert len(result) > 0


def test_fractal_features_all_none():
    result = fractal_features.extract(None, None, None)
    assert result.ndim == 1
    assert np.all(result == 0)


def test_fractal_features_with_hurst():
    hurst = HurstResult(
        hurst_exponent=0.6, intercept=1.0, r_squared=0.95,
        level_variances=np.array([1.0, 2.0, 3.0]),
        regime=RegimeType.TRENDING,
    )
    result = fractal_features.extract(hurst, None, None)
    assert result.ndim == 1
    assert result[0] == 0.6  # First feature should be Hurst exponent


def test_market_features():
    values = np.random.default_rng(42).standard_normal(100) + 100
    result = market_features.extract(values, window=20)
    assert result.ndim == 1
    assert len(result) > 0
    assert np.all(np.isfinite(result))
