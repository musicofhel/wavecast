"""Fractal-based feature extraction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wavecast.core.types import HurstResult, MFDFAResult, SelfSimilarityResult

# Hurst: H, R2, regime_trending, regime_mean_reverting = 4
# MFDFA: spectrum_width, H(q=-5), H(q=0-ish), H(q=5), mean_alpha = 5
# SelfSim: mean_similarity, min_similarity, max_similarity = 3
_HURST_FEATURES = 4
_MFDFA_FEATURES = 5
_SELFSIM_FEATURES = 3
FRACTAL_FEATURE_SIZE = _HURST_FEATURES + _MFDFA_FEATURES + _SELFSIM_FEATURES


def extract(
    hurst: HurstResult | None,
    mfdfa: MFDFAResult | None,
    self_sim: SelfSimilarityResult | None,
) -> NDArray:
    """Extract fractal features from analysis results.

    Returns zeros for any None input.
    """
    features = np.zeros(FRACTAL_FEATURE_SIZE, dtype=np.float64)
    idx = 0

    # Hurst features
    if hurst is not None:
        features[idx] = hurst.hurst_exponent
        features[idx + 1] = hurst.r_squared
        features[idx + 2] = 1.0 if hurst.is_trending else 0.0
        features[idx + 3] = 1.0 if hurst.is_mean_reverting else 0.0
    idx += _HURST_FEATURES

    # MFDFA features
    if mfdfa is not None:
        features[idx] = mfdfa.spectrum_width
        # H at extreme q values and midpoint
        if len(mfdfa.hurst_q) >= 3:
            features[idx + 1] = mfdfa.hurst_q[0]  # H(q_min)
            features[idx + 2] = mfdfa.hurst_q[len(mfdfa.hurst_q) // 2]  # H(q_mid)
            features[idx + 3] = mfdfa.hurst_q[-1]  # H(q_max)
        if len(mfdfa.alpha) > 0:
            features[idx + 4] = float(np.mean(mfdfa.alpha))
    idx += _MFDFA_FEATURES

    # Self-similarity features
    if self_sim is not None:
        features[idx] = self_sim.mean_similarity
        if self_sim.similarity_scores:
            features[idx + 1] = float(np.min(self_sim.similarity_scores))
            features[idx + 2] = float(np.max(self_sim.similarity_scores))

    return features
