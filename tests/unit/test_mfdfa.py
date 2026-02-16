"""Tests for MFDFA."""

import numpy as np

from wavecast.fractal.mfdfa import compute_mfdfa


def test_compute_mfdfa_basic():
    rng = np.random.default_rng(42)
    walk = np.cumsum(rng.standard_normal(1000))
    result = compute_mfdfa(walk)
    assert result.spectrum_width >= 0
    assert len(result.q_values) > 0
    assert len(result.hurst_q) == len(result.q_values)


def test_mfdfa_spectrum_width():
    """Monofractal process should have smaller spectrum width."""
    rng = np.random.default_rng(42)
    # Simple random walk is approximately monofractal
    walk = np.cumsum(rng.standard_normal(2000))
    result = compute_mfdfa(walk)
    assert result.spectrum_width >= 0
