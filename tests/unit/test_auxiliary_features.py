"""Tests for auxiliary feature computation."""

from __future__ import annotations

import numpy as np

from wavecast.data.auxiliary_features import (
    N_AUX_FEATURES,
    compute_detail_auxiliary_features,
)


def test_basic_output_shape():
    """Output shape is (n-1, 4) for n input coefficients."""
    coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compute_detail_auxiliary_features(coeffs)
    assert result.shape == (4, N_AUX_FEATURES)


def test_empty_input():
    """Returns empty array for < 2 coefficients."""
    result = compute_detail_auxiliary_features(np.array([1.0]))
    assert result.shape == (0, N_AUX_FEATURES)

    result = compute_detail_auxiliary_features(np.array([]))
    assert result.shape == (0, N_AUX_FEATURES)


def test_coeff_sign_feature():
    """Feature 0 is sign of original coefficient at position i+1."""
    coeffs = np.array([-2.0, 3.0, -1.0, 0.0, 4.0])
    result = compute_detail_auxiliary_features(coeffs)
    # Signs at positions 1,2,3,4 = +1, -1, 0, +1
    expected_signs = np.array([1.0, -1.0, 0.0, 1.0])
    np.testing.assert_array_equal(result[:, 0], expected_signs)


def test_magnitude_zscore_feature():
    """Feature 1 is z-scored |coefficient| at position i+1."""
    coeffs = np.array([0.0, 1.0, 3.0, 5.0, 7.0])
    result = compute_detail_auxiliary_features(coeffs)
    # |coeffs at 1..4| = [1, 3, 5, 7], z-score them
    abs_vals = np.array([1.0, 3.0, 5.0, 7.0])
    expected = (abs_vals - np.mean(abs_vals)) / np.std(abs_vals)
    np.testing.assert_allclose(result[:, 1], expected, atol=1e-10)


def test_volatility_ratio_feature():
    """Feature 2 is |delta| / EMA(|delta|), capped at 5.0."""
    # Constant deltas should give ratio near 1.0
    coeffs = np.arange(10, dtype=np.float64)  # deltas all = 1.0
    result = compute_detail_auxiliary_features(coeffs, rolling_window=4)
    # After warmup, ratio should converge to 1.0
    assert result[-1, 2] > 0.9
    assert result[-1, 2] < 1.1


def test_volatility_ratio_capped():
    """Volatility ratio is capped at 5.0."""
    # Spike: mostly small, then one huge delta
    coeffs = np.zeros(20)
    coeffs[-1] = 100.0  # huge spike at end
    result = compute_detail_auxiliary_features(coeffs, rolling_window=4)
    assert np.all(result[:, 2] <= 5.0)


def test_approx_direction_feature():
    """Feature 3 is sign of approximation delta at mapped position."""
    detail = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Approx with clear direction: rising then falling
    approx = np.array([1.0, 3.0, 5.0, 4.0, 2.0])
    result = compute_detail_auxiliary_features(detail, approx)
    # Approx deltas: [2, 2, -1, -2] → signs: [1, 1, -1, -1]
    # With 4 detail deltas and 4 approx deltas, mapping is 1:1
    expected = np.array([1.0, 1.0, -1.0, -1.0])
    np.testing.assert_array_equal(result[:, 3], expected)


def test_no_approx_gives_zeros():
    """Without approximation coefficients, feature 3 is all zeros."""
    coeffs = np.array([1.0, 2.0, 3.0, 4.0])
    result = compute_detail_auxiliary_features(coeffs)
    np.testing.assert_array_equal(result[:, 3], 0.0)


def test_constant_coefficients():
    """Handles constant coefficients gracefully (no div by zero)."""
    coeffs = np.ones(10)
    result = compute_detail_auxiliary_features(coeffs)
    assert result.shape == (9, N_AUX_FEATURES)
    # Signs all +1
    np.testing.assert_array_equal(result[:, 0], 1.0)
    # Z-score of constants is 0
    np.testing.assert_array_equal(result[:, 1], 0.0)
    # All deltas are 0, so volatility ratio is 0/ema → 0
    np.testing.assert_array_equal(result[:, 2], 0.0)


def test_approx_mapping_different_lengths():
    """Approx can have different length than detail — mapping by ratio."""
    detail = np.arange(20, dtype=np.float64)
    # Approx much shorter (different DWT level)
    approx = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
    result = compute_detail_auxiliary_features(detail, approx)
    # 19 deltas, 4 approx deltas — ratio = 4/19 ≈ 0.21
    # All approx direction values should be valid signs
    assert result.shape == (19, N_AUX_FEATURES)
    assert np.all(np.isin(result[:, 3], [-1.0, 0.0, 1.0]))
