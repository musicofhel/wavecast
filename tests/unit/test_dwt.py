"""Tests for DWT decomposition."""

import numpy as np
import pytest

from wavecast.core.exceptions import DecompositionError
from wavecast.wavelets.dwt import compute_level_stats, decompose


def test_decompose_sine(sine_series):
    decomp = decompose(sine_series, wavelet="db4", level=3)
    assert decomp.level == 3
    assert decomp.wavelet == "db4"
    assert len(decomp.coefficients) == 4  # cA3 + cD3 + cD2 + cD1
    assert decomp.original_length == sine_series.length


def test_decompose_array():
    data = np.sin(np.linspace(0, 4 * np.pi, 256))
    decomp = decompose(data, wavelet="haar", level=3)
    assert decomp.level == 3
    assert decomp.ticker == ""


def test_decompose_too_short():
    data = np.array([1.0])
    with pytest.raises(DecompositionError):
        decompose(data, level=3)


def test_decompose_level_too_high():
    data = np.random.default_rng(42).standard_normal(16)
    with pytest.raises(DecompositionError):
        decompose(data, wavelet="db4", level=10)


def test_decompose_invalid_wavelet():
    data = np.random.default_rng(42).standard_normal(100)
    with pytest.raises(DecompositionError):
        decompose(data, wavelet="nonexistent")


def test_compute_level_stats(sine_series):
    decomp = decompose(sine_series, level=4)
    stats = compute_level_stats(decomp)
    assert len(stats) == 5  # 1 approx + 4 detail levels
    for s in stats:
        assert s.energy >= 0
        assert s.variance >= 0
        assert s.entropy >= 0
        assert s.num_coefficients > 0


def test_detail_at_level(sine_series):
    decomp = decompose(sine_series, level=3)
    d1 = decomp.detail_at_level(1)  # finest
    d3 = decomp.detail_at_level(3)  # coarsest
    assert len(d1) > len(d3)  # finer level has more coefficients
