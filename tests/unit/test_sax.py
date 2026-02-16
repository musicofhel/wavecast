"""Tests for wavecast.sax.sax."""

import numpy as np

from wavecast.sax.sax import breakpoints, sax_distance, sax_transform


def test_breakpoints_symmetry():
    bps = breakpoints(4)
    assert len(bps) == 3
    # Standard normal breakpoints should be symmetric around 0
    np.testing.assert_almost_equal(bps[0], -bps[-1], decimal=10)
    np.testing.assert_almost_equal(bps[1], 0.0, decimal=10)


def test_breakpoints_monotonic():
    bps = breakpoints(8)
    assert len(bps) == 7
    for i in range(len(bps) - 1):
        assert bps[i] < bps[i + 1]


def test_sax_transform_valid_symbols():
    rng = np.random.default_rng(42)
    series = rng.standard_normal(200)
    result = sax_transform(series, n_segments=20, alphabet_size=4)
    assert len(result.symbols) == 20
    # All symbols should be in 'a' to 'd'
    for ch in result.symbols:
        assert ch in "abcd"


def test_sax_transform_alphabet_size():
    rng = np.random.default_rng(42)
    series = rng.standard_normal(200)
    result = sax_transform(series, n_segments=10, alphabet_size=8)
    for ch in result.symbols:
        assert ch in "abcdefgh"


def test_sax_transform_metadata():
    rng = np.random.default_rng(42)
    series = rng.standard_normal(100)
    result = sax_transform(series, n_segments=10, alphabet_size=4)
    assert result.alphabet_size == 4
    assert result.n_segments == 10
    assert result.original_length == 100


def test_sax_transform_empty():
    result = sax_transform(np.array([]), n_segments=10, alphabet_size=4)
    assert result.symbols == ""
    assert result.n_segments == 0


def test_sax_transform_constant_series():
    series = np.ones(100)
    result = sax_transform(series, n_segments=10, alphabet_size=4)
    # All same symbols since std=0
    assert len(set(result.symbols)) == 1


def test_sax_distance_same_string():
    dist = sax_distance("abcd", "abcd", n=100, alphabet_size=4)
    assert dist == 0.0


def test_sax_distance_adjacent_symbols():
    # Adjacent symbols should have distance 0
    dist = sax_distance("aaaa", "bbbb", n=100, alphabet_size=4)
    assert dist == 0.0


def test_sax_distance_far_symbols():
    # Far apart symbols should have positive distance
    dist = sax_distance("aaaa", "dddd", n=100, alphabet_size=4)
    assert dist > 0.0


def test_sax_distance_symmetric():
    dist1 = sax_distance("abcd", "dcba", n=100, alphabet_size=4)
    dist2 = sax_distance("dcba", "abcd", n=100, alphabet_size=4)
    np.testing.assert_almost_equal(dist1, dist2)
