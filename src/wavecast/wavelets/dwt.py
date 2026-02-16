"""Discrete Wavelet Transform decomposition and analysis."""

from __future__ import annotations

import numpy as np
import pywt
from numpy.typing import NDArray

from wavecast.core.exceptions import DecompositionError
from wavecast.core.types import LevelStats, TimeSeries, WaveletDecomposition


def decompose(
    data: NDArray | TimeSeries,
    wavelet: str = "db4",
    level: int = 5,
) -> WaveletDecomposition:
    """Perform multi-level DWT decomposition.

    Args:
        data: Input array or TimeSeries.
        wavelet: PyWavelets wavelet name (e.g. 'db4', 'haar', 'sym5').
        level: Decomposition depth.

    Returns:
        WaveletDecomposition with coefficients [cA_n, cD_n, ..., cD_1].

    Raises:
        DecompositionError: If the data is too short or wavelet is invalid.
    """
    if isinstance(data, TimeSeries):
        values = np.array(data.values, dtype=np.float64, copy=True)
        ticker = data.ticker
    else:
        values = np.array(data, dtype=np.float64, copy=True)
        ticker = ""

    if values.ndim != 1:
        raise DecompositionError(
            f"Expected 1D data, got shape {values.shape}"
        )

    if len(values) < 2:
        raise DecompositionError(
            "Data must have at least 2 samples for DWT"
        )

    try:
        w = pywt.Wavelet(wavelet)
    except ValueError as e:
        raise DecompositionError(f"Invalid wavelet '{wavelet}': {e}") from e

    max_level = pywt.dwt_max_level(len(values), w.dec_len)
    if level > max_level:
        raise DecompositionError(
            f"Requested level {level} exceeds max level {max_level} "
            f"for data length {len(values)} with wavelet '{wavelet}'"
        )

    try:
        coeffs = pywt.wavedec(values, wavelet, level=level)
    except Exception as e:
        raise DecompositionError(f"DWT failed: {e}") from e

    return WaveletDecomposition(
        coefficients=coeffs,
        wavelet=wavelet,
        level=level,
        original_length=len(values),
        ticker=ticker,
    )


def compute_level_stats(decomp: WaveletDecomposition) -> list[LevelStats]:
    """Compute statistics for each decomposition level.

    Level 0 = approximation coefficients, levels 1..N = detail coefficients
    (1 = finest/highest frequency).

    Args:
        decomp: A WaveletDecomposition result.

    Returns:
        List of LevelStats, one per level (approximation + all detail levels).
    """
    stats: list[LevelStats] = []

    for i, coeffs in enumerate(decomp.coefficients):
        coeffs = coeffs.astype(np.float64)

        energy = float(np.sum(coeffs ** 2))
        variance = float(np.var(coeffs))
        mean = float(np.mean(coeffs))
        std = float(np.std(coeffs))

        # Shannon entropy of normalized squared coefficients
        sq = coeffs ** 2
        total = np.sum(sq)
        if total > 0:
            p = sq / total
            p = p[p > 0]
            entropy = float(-np.sum(p * np.log2(p)))
        else:
            entropy = 0.0

        # Level numbering: index 0 = approximation (level N),
        # index 1 = detail level N, ..., index N = detail level 1
        if i == 0:
            level_num = 0  # approximation
            approx_period = 2 ** decomp.level  # coarsest scale
        else:
            level_num = decomp.level - i + 1
            approx_period = 2 ** level_num

        stats.append(
            LevelStats(
                level=level_num,
                energy=energy,
                variance=variance,
                entropy=entropy,
                mean=mean,
                std=std,
                num_coefficients=len(coeffs),
                approximate_period_days=float(approx_period),
            )
        )

    return stats
