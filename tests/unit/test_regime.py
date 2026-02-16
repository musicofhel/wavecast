"""Tests for regime detection."""

import numpy as np

from wavecast.core.types import RegimeType
from wavecast.fractal.regime import detect_regime


def test_detect_regime_basic():
    rng = np.random.default_rng(42)
    walk = np.cumsum(rng.standard_normal(500))
    result = detect_regime(walk)
    assert result.regime in [RegimeType.TRENDING, RegimeType.MEAN_REVERTING, RegimeType.RANDOM_WALK]
    assert 0 <= result.confidence <= 1
    assert result.hurst is not None
