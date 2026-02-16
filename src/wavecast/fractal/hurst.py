"""Hurst exponent estimation via wavelet variance method."""

from __future__ import annotations

import numpy as np
import pywt
from numpy.typing import NDArray

from wavecast.core.exceptions import FractalError
from wavecast.core.types import HurstResult, RegimeType


def wavelet_hurst(
    values: NDArray,
    wavelet: str = "db4",
    max_level: int | None = None,
) -> HurstResult:
    """Estimate the Hurst exponent using wavelet variance scaling.

    Performs DWT decomposition, computes variance at each detail level,
    then fits log2(var_j) vs log2(scale_j) to extract H.
    """
    values = np.array(values, dtype=np.float64, copy=True)
    if len(values) < 16:
        raise FractalError("Need at least 16 data points for wavelet Hurst estimation")

    if max_level is None:
        max_level = pywt.dwt_max_level(len(values), pywt.Wavelet(wavelet).dec_len)
    max_level = max(min(max_level, pywt.dwt_max_level(len(values), pywt.Wavelet(wavelet).dec_len)), 1)

    coeffs = pywt.wavedec(values, wavelet, level=max_level)
    # coeffs = [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    details = coeffs[1:]  # detail coefficients from coarsest to finest

    if len(details) < 2:
        raise FractalError("Need at least 2 detail levels for regression")

    level_variances = np.array([np.var(d) for d in details], dtype=np.float64)

    # Filter out zero-variance levels to avoid log(0)
    valid = level_variances > 0
    if valid.sum() < 2:
        raise FractalError("Insufficient non-zero variance levels for regression")

    # Levels go from n (coarsest) down to 1 (finest) in the details list
    num_levels = len(details)
    j_values = np.arange(num_levels, 0, -1, dtype=np.float64)  # [n, n-1, ..., 1]

    log_scale = np.log2(2.0 ** j_values[valid])
    log_var = np.log2(level_variances[valid])

    slope, intercept = np.polyfit(log_scale, log_var, 1)
    hurst_exponent = (slope + 1.0) / 2.0

    # Compute R-squared
    predicted = slope * log_scale + intercept
    ss_res = np.sum((log_var - predicted) ** 2)
    ss_tot = np.sum((log_var - np.mean(log_var)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Classify regime
    if hurst_exponent > 0.55:
        regime = RegimeType.TRENDING
    elif hurst_exponent < 0.45:
        regime = RegimeType.MEAN_REVERTING
    else:
        regime = RegimeType.RANDOM_WALK

    return HurstResult(
        hurst_exponent=float(hurst_exponent),
        intercept=float(intercept),
        r_squared=float(r_squared),
        level_variances=level_variances,
        regime=regime,
    )


def rolling_hurst(
    values: NDArray,
    window: int = 252,
    step: int = 1,
    wavelet: str = "db4",
) -> tuple[NDArray, NDArray]:
    """Compute rolling Hurst exponent over a sliding window.

    Returns:
        Tuple of (hurst_values, center_indices) arrays.
    """
    values = np.array(values, dtype=np.float64, copy=True)
    n = len(values)
    if n < window:
        raise FractalError(f"Series length {n} is shorter than window {window}")

    hurst_vals = []
    indices = []

    for start in range(0, n - window + 1, step):
        segment = values[start : start + window]
        try:
            result = wavelet_hurst(segment, wavelet=wavelet)
            hurst_vals.append(result.hurst_exponent)
        except FractalError:
            hurst_vals.append(np.nan)
        indices.append(start + window // 2)

    return np.array(hurst_vals, dtype=np.float64), np.array(indices, dtype=np.int64)
