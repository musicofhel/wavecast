"""Piecewise Aggregate Approximation (PAA)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wavecast.core.exceptions import SAXError


def paa(series: NDArray, n_segments: int) -> NDArray:
    """Reduce a time series to n_segments via mean-pooling.

    Each segment value is the mean of the corresponding consecutive frames.

    Args:
        series: Input 1-D array.
        n_segments: Number of output segments.

    Returns:
        Array of length n_segments with mean-pooled values.
    """
    n = len(series)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n_segments <= 0:
        raise SAXError(f"n_segments must be > 0, got {n_segments}")

    # Clamp to series length
    n_segments = min(n_segments, n)

    result = np.zeros(n_segments, dtype=np.float64)
    for i in range(n_segments):
        start = i * n // n_segments
        end = (i + 1) * n // n_segments
        result[i] = np.mean(series[start:end])
    return result


def inverse_paa(paa_values: NDArray, original_length: int) -> NDArray:
    """Expand PAA values back to original length by repeating.

    Args:
        paa_values: Array of PAA segment values.
        original_length: Target output length.

    Returns:
        Array of length original_length with repeated PAA values.
    """
    n_segments = len(paa_values)
    if n_segments == 0:
        return np.array([], dtype=np.float64)

    result = np.zeros(original_length, dtype=np.float64)
    for i in range(n_segments):
        start = i * original_length // n_segments
        end = (i + 1) * original_length // n_segments
        result[start:end] = paa_values[i]
    return result
