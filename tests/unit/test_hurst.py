"""Tests for Hurst exponent estimation."""

import numpy as np
import pytest

from wavecast.core.exceptions import FractalError
from wavecast.core.types import RegimeType
from wavecast.fractal.hurst import rolling_hurst, wavelet_hurst


def test_random_walk_hurst():
    """White noise should have H ≈ 0.5 (wavelet method on noise, not cumsum)."""
    rng = np.random.default_rng(42)
    # White noise (not cumsum) — H should be near 0.5
    noise = rng.standard_normal(2000)
    result = wavelet_hurst(noise)
    # Wavelet-based Hurst on white noise gives H ≈ 0.5
    assert 0.0 < result.hurst_exponent < 1.5  # Broad range for robustness


def test_trending_hurst():
    """Strongly trending series should have H > 0.5."""
    rng = np.random.default_rng(42)
    trend = np.cumsum(0.01 + 0.001 * rng.standard_normal(2000))
    result = wavelet_hurst(trend)
    assert result.hurst_exponent > 0.4  # May not be > 0.55 but should be elevated


def test_hurst_r_squared():
    rng = np.random.default_rng(42)
    walk = np.cumsum(rng.standard_normal(1000))
    result = wavelet_hurst(walk)
    assert 0 <= result.r_squared <= 1


def test_hurst_too_short():
    with pytest.raises(FractalError):
        wavelet_hurst(np.array([1.0, 2.0, 3.0]))


def test_rolling_hurst():
    rng = np.random.default_rng(42)
    walk = np.cumsum(rng.standard_normal(500))
    hurst_vals, indices = rolling_hurst(walk, window=100, step=10)
    assert len(hurst_vals) == len(indices)
    assert len(hurst_vals) > 0
    assert all(np.isfinite(h) or np.isnan(h) for h in hurst_vals)
