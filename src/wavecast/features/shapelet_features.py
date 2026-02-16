"""Shapelet-based feature extraction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wavecast.core.types import MatchResult

# Feature vector size: top_k * 3 (distance, normalized_distance, length_ratio) + 4 stats
_FEATURES_PER_MATCH = 3
_STAT_FEATURES = 4  # mean_dist, std_dist, min_dist, max_dist


def extract(match_result: MatchResult | None, top_k: int = 5) -> NDArray:
    """Extract features from shapelet matching results.

    For each of the top_k matches: DTW distance, normalized distance, length ratio.
    Plus aggregate stats: mean/std/min/max of distances.
    Returns zeros if match_result is None or empty.
    """
    n_features = top_k * _FEATURES_PER_MATCH + _STAT_FEATURES
    features = np.zeros(n_features, dtype=np.float64)

    if match_result is None or len(match_result.matches) == 0:
        return features

    matches = match_result.matches[:top_k]
    query_len = match_result.query_length

    distances = []
    for i, m in enumerate(matches):
        features[i * _FEATURES_PER_MATCH] = m.distance
        features[i * _FEATURES_PER_MATCH + 1] = m.normalized_distance
        # Length ratio: shapelet length / query length
        shapelet_len = m.query_end - m.query_start if m.query_end > m.query_start else 1
        features[i * _FEATURES_PER_MATCH + 2] = shapelet_len / max(query_len, 1)
        distances.append(m.distance)

    # Aggregate stats
    offset = top_k * _FEATURES_PER_MATCH
    if distances:
        d = np.array(distances)
        features[offset] = float(np.mean(d))
        features[offset + 1] = float(np.std(d))
        features[offset + 2] = float(np.min(d))
        features[offset + 3] = float(np.max(d))

    return features
