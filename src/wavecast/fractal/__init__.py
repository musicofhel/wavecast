"""Fractal analysis: Hurst exponent, MFDFA, self-similarity."""

from wavecast.fractal.hurst import rolling_hurst, wavelet_hurst
from wavecast.fractal.mfdfa import compute_mfdfa
from wavecast.fractal.regime import detect_regime, rolling_regime
from wavecast.fractal.self_similarity import cross_scale_similarity

__all__ = [
    "wavelet_hurst",
    "rolling_hurst",
    "compute_mfdfa",
    "cross_scale_similarity",
    "detect_regime",
    "rolling_regime",
]
