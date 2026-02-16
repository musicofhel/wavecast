"""Tests for cross-scale self-similarity."""

from wavecast.fractal.self_similarity import cross_scale_similarity
from wavecast.wavelets.dwt import decompose


def test_cross_scale_similarity(sine_series):
    decomp = decompose(sine_series, level=4)
    result = cross_scale_similarity(decomp)
    assert len(result.level_pairs) > 0
    assert len(result.dtw_distances) == len(result.level_pairs)
    assert len(result.similarity_scores) == len(result.level_pairs)
    assert 0 <= result.mean_similarity <= 1
    for score in result.similarity_scores:
        assert 0 <= score <= 1
