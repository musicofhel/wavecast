"""Wavelet-domain feature extraction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis, skew

from wavecast.core.types import WaveletDecomposition

# Per-level features: energy, variance, entropy, mean, std, skew, kurtosis = 7
_FEATURES_PER_LEVEL = 7
# Cross-level: energy ratio pairs + total energy + energy concentration
_MAX_LEVELS = 10


def _safe_entropy(coeffs: NDArray) -> float:
    """Shannon entropy of squared coefficient distribution."""
    sq = coeffs**2
    total = sq.sum()
    if total == 0:
        return 0.0
    p = sq / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def extract(decomp: WaveletDecomposition) -> NDArray:
    """Extract feature vector from wavelet decomposition.

    Returns a flat array with per-level statistics and cross-level features.
    Feature layout: [level_1_feats, level_2_feats, ..., level_n_feats, approx_feats, cross_level_feats]
    """
    details = decomp.details  # [cD_n, ..., cD_1]
    approx = decomp.approximation

    level_features = []

    # Detail level features (from coarsest cD_n to finest cD_1)
    for d in details:
        d = d.astype(np.float64)
        energy = float(np.sum(d**2))
        var = float(np.var(d))
        ent = _safe_entropy(d)
        mn = float(np.mean(d))
        sd = float(np.std(d))
        sk = float(skew(d)) if len(d) > 2 else 0.0
        kt = float(kurtosis(d)) if len(d) > 2 else 0.0
        level_features.extend([energy, var, ent, mn, sd, sk, kt])

    # Pad to fixed size if fewer levels than MAX
    n_detail_levels = len(details)
    for _ in range(n_detail_levels, _MAX_LEVELS):
        level_features.extend([0.0] * _FEATURES_PER_LEVEL)

    # Approximation features (same 7 stats)
    a = approx.astype(np.float64)
    level_features.extend([
        float(np.sum(a**2)),
        float(np.var(a)),
        _safe_entropy(a),
        float(np.mean(a)),
        float(np.std(a)),
        float(skew(a)) if len(a) > 2 else 0.0,
        float(kurtosis(a)) if len(a) > 2 else 0.0,
    ])

    # Cross-level features
    energies = np.array([np.sum(d**2) for d in details], dtype=np.float64)
    total_energy = energies.sum()
    energy_ratios = energies / total_energy if total_energy > 0 else np.zeros_like(energies)

    # Energy concentration (Gini-like): how concentrated is energy across levels
    sorted_ratios = np.sort(energy_ratios)[::-1]
    cumulative = np.cumsum(sorted_ratios)
    concentration = float(cumulative[0]) if len(cumulative) > 0 else 0.0

    cross_feats = [float(total_energy), concentration]
    # Pad energy ratios to MAX_LEVELS
    padded_ratios = np.zeros(_MAX_LEVELS)
    padded_ratios[: len(energy_ratios)] = energy_ratios
    cross_feats.extend(padded_ratios.tolist())

    level_features.extend(cross_feats)

    return np.array(level_features, dtype=np.float64)
