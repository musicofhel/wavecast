"""Tests for DTW similarity."""

import numpy as np

from wavecast.dtw.similarity import pairwise_dtw_matrix, similarity_matrix


def test_pairwise_symmetric():
    rng = np.random.default_rng(42)
    series = [rng.standard_normal(20) for _ in range(4)]
    matrix = pairwise_dtw_matrix(series, window=5)
    assert matrix.shape == (4, 4)
    # Symmetric
    np.testing.assert_array_almost_equal(matrix, matrix.T, decimal=10)
    # Diagonal is zero
    for i in range(4):
        assert matrix[i, i] < 1e-10


def test_similarity_matrix_range():
    dist = np.array([[0.0, 1.0], [1.0, 0.0]])
    sim = similarity_matrix(dist)
    assert sim.shape == (2, 2)
    # Diagonal should be 1.0 (similarity to self)
    np.testing.assert_almost_equal(sim[0, 0], 1.0)
    # Off-diagonal should be between 0 and 1
    assert 0 < sim[0, 1] < 1
