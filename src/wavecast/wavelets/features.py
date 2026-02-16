"""Feature extraction from wavelet decompositions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from wavecast.core.types import WaveletDecomposition


def extract_level_features(decomp: WaveletDecomposition) -> NDArray:
    """Extract per-level statistical features from a wavelet decomposition.

    For each level (approximation + all detail levels), computes 7 features:
    [energy, variance, entropy, mean, std, skewness, kurtosis].

    Args:
        decomp: A WaveletDecomposition result.

    Returns:
        2D array of shape (n_levels, 7).
    """
    n_levels = len(decomp.coefficients)
    features = np.zeros((n_levels, 7), dtype=np.float64)

    for i, coeffs in enumerate(decomp.coefficients):
        c = coeffs.astype(np.float64)

        # Energy
        energy = np.sum(c ** 2)

        # Variance
        variance = np.var(c)

        # Shannon entropy of normalized squared coefficients
        sq = c ** 2
        total = np.sum(sq)
        if total > 0:
            p = sq / total
            p = p[p > 0]
            entropy = -np.sum(p * np.log2(p))
        else:
            entropy = 0.0

        # Basic statistics
        mean = np.mean(c)
        std = np.std(c)

        # Higher-order moments (handle constant arrays)
        if std > 0 and len(c) >= 3:
            skewness = float(scipy_stats.skew(c))
        else:
            skewness = 0.0

        if std > 0 and len(c) >= 4:
            kurtosis = float(scipy_stats.kurtosis(c))
        else:
            kurtosis = 0.0

        features[i] = [energy, variance, entropy, mean, std, skewness, kurtosis]

    return features


def extract_cross_level_features(decomp: WaveletDecomposition) -> NDArray:
    """Extract cross-level features from a wavelet decomposition.

    Computes:
    - Energy ratio for each level (level_energy / total_energy)
    - Total energy across all levels

    Args:
        decomp: A WaveletDecomposition result.

    Returns:
        1D array of shape (n_levels + 1,) where the first n_levels entries
        are energy ratios and the last entry is the total energy.
    """
    n_levels = len(decomp.coefficients)
    energies = np.array(
        [np.sum(c.astype(np.float64) ** 2) for c in decomp.coefficients],
        dtype=np.float64,
    )
    total_energy = np.sum(energies)

    features = np.zeros(n_levels + 1, dtype=np.float64)

    if total_energy > 0:
        features[:n_levels] = energies / total_energy
    else:
        features[:n_levels] = 0.0

    features[n_levels] = total_energy

    return features
