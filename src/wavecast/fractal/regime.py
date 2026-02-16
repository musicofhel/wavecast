"""Fractal-based market regime detection."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wavecast.core.config import FractalConfig
from wavecast.core.exceptions import FractalError
from wavecast.core.types import RegimeDetection

from .hurst import wavelet_hurst
from .mfdfa import compute_mfdfa


def detect_regime(
    values: NDArray,
    config: FractalConfig | None = None,
) -> RegimeDetection:
    """Detect the current market regime using fractal analysis.

    Combines Hurst exponent with optional MFDFA for a confidence-weighted
    regime classification.
    """
    values = np.array(values, dtype=np.float64, copy=True)
    if config is None:
        config = FractalConfig()

    hurst_result = wavelet_hurst(values)

    # Attempt MFDFA (requires more data, so allow failure)
    mfdfa_result = None
    if len(values) >= 100:
        try:
            mfdfa_result = compute_mfdfa(
                values,
                q_range=config.mfdfa_q_range,
                q_steps=config.mfdfa_q_steps,
            )
        except FractalError:
            pass

    # Confidence is based on R-squared of the Hurst fit and
    # distance from the thresholds
    h = hurst_result.hurst_exponent
    base_confidence = hurst_result.r_squared

    # How far from the "uncertain" band [0.45, 0.55]
    if h > config.trending_threshold:
        distance = (h - config.trending_threshold) / (1.0 - config.trending_threshold)
    elif h < config.mean_revert_threshold:
        distance = (config.mean_revert_threshold - h) / config.mean_revert_threshold
    else:
        distance = 0.0

    confidence = float(np.clip(base_confidence * (0.5 + 0.5 * distance), 0.0, 1.0))

    # Boost confidence if MFDFA confirms narrow spectrum (single fractal = cleaner regime)
    if mfdfa_result is not None and mfdfa_result.spectrum_width < 0.3:
        confidence = min(confidence + 0.1, 1.0)

    return RegimeDetection(
        regime=hurst_result.regime,
        hurst=hurst_result,
        mfdfa=mfdfa_result,
        confidence=confidence,
        window_size=len(values),
    )


def rolling_regime(
    values: NDArray,
    window: int = 252,
    step: int = 21,
    config: FractalConfig | None = None,
) -> list[RegimeDetection]:
    """Compute rolling regime detection over a sliding window."""
    values = np.array(values, dtype=np.float64, copy=True)
    n = len(values)
    if n < window:
        raise FractalError(f"Series length {n} is shorter than window {window}")

    if config is None:
        config = FractalConfig()

    results: list[RegimeDetection] = []
    for start in range(0, n - window + 1, step):
        segment = values[start : start + window]
        results.append(detect_regime(segment, config=config))

    return results
