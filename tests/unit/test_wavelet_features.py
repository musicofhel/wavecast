"""Tests for wavelet feature extraction."""

import numpy as np

from wavecast.wavelets.dwt import decompose
from wavecast.wavelets.features import extract_cross_level_features, extract_level_features


def test_extract_level_features(sine_series):
    decomp = decompose(sine_series, level=4)
    features = extract_level_features(decomp)
    assert features.ndim in (1, 2)  # May be 2D (levels x features) or 1D (flattened)
    assert features.size > 0
    assert np.all(np.isfinite(features))


def test_extract_cross_level_features(sine_series):
    decomp = decompose(sine_series, level=4)
    features = extract_cross_level_features(decomp)
    assert features.ndim == 1
    assert len(features) > 0
    assert np.all(np.isfinite(features))
