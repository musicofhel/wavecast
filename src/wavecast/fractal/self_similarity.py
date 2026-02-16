"""Cross-scale self-similarity analysis via DTW."""

from __future__ import annotations

import numpy as np
from dtaidistance import dtw
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from wavecast.core.exceptions import FractalError
from wavecast.core.types import SelfSimilarityResult, WaveletDecomposition


def _resample(arr: NDArray, target_length: int) -> NDArray:
    """Resample a 1-D array to target_length using linear interpolation."""
    if len(arr) == target_length:
        return arr
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target_length)
    f = interp1d(x_old, arr, kind="linear")
    return f(x_new)


def cross_scale_similarity(decomp: WaveletDecomposition) -> SelfSimilarityResult:
    """Compute DTW-based similarity between adjacent wavelet detail levels.

    For each pair of adjacent detail levels, resamples to the same length
    and computes DTW distance. Similarity = 1 / (1 + distance).
    """
    details = decomp.details  # [cD_n, cD_{n-1}, ..., cD_1] (coarsest to finest)
    if len(details) < 2:
        raise FractalError("Need at least 2 detail levels for cross-scale similarity")

    level_pairs: list[tuple[int, int]] = []
    dtw_distances: list[float] = []
    similarity_scores: list[float] = []

    for i in range(len(details) - 1):
        coarser = details[i]
        finer = details[i + 1]

        # Resample both to the length of the longer one
        target_len = max(len(coarser), len(finer))
        coarser_resampled = _resample(coarser.astype(np.float64), target_len)
        finer_resampled = _resample(finer.astype(np.float64), target_len)

        # Normalize to zero mean, unit variance for fair comparison
        c_std = np.std(coarser_resampled)
        f_std = np.std(finer_resampled)
        if c_std > 0:
            coarser_resampled = (coarser_resampled - np.mean(coarser_resampled)) / c_std
        if f_std > 0:
            finer_resampled = (finer_resampled - np.mean(finer_resampled)) / f_std

        d = dtw.distance(coarser_resampled, finer_resampled)
        sim = 1.0 / (1.0 + d)

        # Levels: coarser is level (n - i), finer is level (n - i - 1)
        coarser_level = decomp.level - i
        finer_level = decomp.level - i - 1
        level_pairs.append((coarser_level, finer_level))
        dtw_distances.append(float(d))
        similarity_scores.append(float(sim))

    mean_sim = float(np.mean(similarity_scores)) if similarity_scores else 0.0

    return SelfSimilarityResult(
        level_pairs=level_pairs,
        dtw_distances=dtw_distances,
        similarity_scores=similarity_scores,
        mean_similarity=mean_sim,
    )
