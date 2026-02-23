"""Fractional differencing for wavelet coefficients.

Replaces np.diff (d=1.0) with fractional differencing (d=d_optimal)
to preserve long-range memory while achieving stationarity.

References:
    López de Prado, "Advances in Financial Machine Learning" (2018), Ch. 5
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _fracdiff_weights(d: float, window: int) -> NDArray:
    """Compute truncated fractional differencing weights.

    w_k = prod_{i=1}^{k} (d - i + 1) / i  for k = 0, 1, ..., window-1
    w_0 = 1.0 always.
    """
    w = np.zeros(window, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, window):
        w[k] = w[k - 1] * (d - k + 1) / k
    return w


def fracdiff(series: NDArray, d: float, window: int = 100) -> NDArray:
    """Apply fractional differencing of order d to a 1-D series.

    Args:
        series: 1-D float array.
        d: Differencing order (0 < d <= 1). d=1.0 is standard first-difference.
        window: Truncation length for weight series.

    Returns:
        Fractionally differenced series (same length as input, but first
        `window-1` values use partial windows).
    """
    n = len(series)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    w = _fracdiff_weights(d, min(window, n))
    result = np.zeros(n, dtype=np.float64)

    for t in range(n):
        # Use available history up to `window` points
        k_max = min(t + 1, len(w))
        # series[t], series[t-1], ..., series[t-k_max+1] dotted with w[0..k_max-1]
        chunk = series[t - k_max + 1 : t + 1][::-1]
        result[t] = np.dot(w[:k_max], chunk)

    return result


def find_min_d(
    series: NDArray,
    max_d: float = 1.0,
    step: float = 0.05,
    pvalue: float = 0.05,
    window: int = 100,
) -> float:
    """Find minimum fractional differencing order for stationarity.

    Uses ADF test to check stationarity at each d value.
    Returns the smallest d where ADF p-value < threshold.

    Args:
        series: 1-D float array (raw, undifferenced).
        max_d: Maximum d to try (1.0 = standard differencing).
        step: Step size for d search.
        pvalue: ADF p-value threshold for stationarity.
        window: Truncation window for fracdiff weights.

    Returns:
        Optimal d value. Returns max_d if no smaller d achieves stationarity.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        # Fallback: return 0.5 if statsmodels not available
        return 0.5

    d_values = np.arange(0.05, max_d + step, step)
    for d in d_values:
        fd = fracdiff(series, d, window)
        # Skip if constant or too short
        if len(fd) < 20 or np.std(fd) < 1e-10:
            continue
        try:
            adf_result = adfuller(fd, maxlag=min(10, len(fd) // 4), autolag=None)
            if adf_result[1] < pvalue:
                return float(d)
        except (ValueError, np.linalg.LinAlgError):
            continue

    return max_d


def fracdiff_coefficients(
    coefficients: NDArray,
    d: float | None = None,
    window: int = 100,
) -> tuple[NDArray, float]:
    """Fractionally difference wavelet coefficients.

    If d is None, find optimal d via find_min_d().

    Args:
        coefficients: Raw wavelet detail coefficients.
        d: Differencing order. If None, auto-detected.
        window: Truncation window.

    Returns:
        (fracdiff_series, d_used). The fracdiff_series has the SAME length
        as input (unlike np.diff which shortens by 1).
    """
    if d is None:
        d = find_min_d(coefficients, window=window)

    fd = fracdiff(coefficients, d, window)
    return fd, d
