"""Tests for SAX feature extraction."""

import numpy as np

from wavecast.core.config import SAXConfig
from wavecast.features.sax_features import SAX_FEATURE_SIZE, extract
from wavecast.wavelets.dwt import decompose


def test_extract_produces_correct_shape(sine_series):
    decomp = decompose(sine_series, level=3)
    result = extract(decomp)
    assert result.ndim == 1
    assert len(result) == SAX_FEATURE_SIZE


def test_extract_all_finite(sine_series):
    decomp = decompose(sine_series, level=3)
    result = extract(decomp)
    assert np.all(np.isfinite(result))


def test_extract_with_custom_config(sine_series):
    decomp = decompose(sine_series, level=3)
    cfg = SAXConfig(n_segments=10, alphabet_size=4, word_length=3, word_stride=1)
    result = extract(decomp, sax_config=cfg)
    assert len(result) == SAX_FEATURE_SIZE
    assert np.all(np.isfinite(result))


def test_extract_different_levels(sine_series):
    decomp3 = decompose(sine_series, level=3)
    decomp5 = decompose(sine_series, level=5)
    r3 = extract(decomp3)
    r5 = extract(decomp5)
    # Same output size due to padding
    assert len(r3) == len(r5) == SAX_FEATURE_SIZE
    # But different values since more levels are populated
    assert not np.array_equal(r3, r5)


def test_extract_default_config(sine_series):
    decomp = decompose(sine_series, level=3)
    # None config should use defaults
    result = extract(decomp, sax_config=None)
    assert len(result) == SAX_FEATURE_SIZE
