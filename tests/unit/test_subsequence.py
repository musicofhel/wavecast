"""Tests for DTW subsequence search."""

import numpy as np

from wavecast.dtw.subsequence import subsequence_search


def test_find_planted_pattern():
    """Should find a planted pattern within a longer series."""
    rng = np.random.default_rng(42)
    long_series = rng.standard_normal(200)
    # Plant a distinctive pattern at index 50
    pattern = np.array([3.0, 5.0, 3.0, 5.0, 3.0])
    long_series[50:55] = pattern

    results = subsequence_search(pattern, long_series, window=5, top_k=3)
    assert len(results) > 0
    # Best match should be near index 50
    best_idx, best_dist = results[0]
    assert abs(best_idx - 50) <= 2


def test_empty_or_error():
    """Query longer than series should return empty or raise."""
    from wavecast.core.exceptions import DTWError

    query = np.array([1.0, 2.0, 3.0])
    long_series = np.array([1.0])
    try:
        results = subsequence_search(query, long_series, top_k=5)
        assert len(results) == 0
    except DTWError:
        pass  # Also acceptable
