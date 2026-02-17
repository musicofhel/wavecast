"""Tests for return target computation."""

import numpy as np
import pytest

from wavecast.targets.returns import (
    ReturnTargetResult,
    assign_quantile_labels,
    compute_quantile_boundaries,
    compute_return_targets,
    compute_sample_returns,
)


def test_compute_sample_returns_basic():
    """Returns are correctly computed from price data."""
    # Price: [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120]
    prices = np.array([100.0 + 2 * i for i in range(11)])
    token_positions = np.array([0, 1], dtype=np.int64)
    levels = np.array([1, 1], dtype=np.int64)  # 2^1 = 2 bar span
    n_coeffs = np.array([5, 5], dtype=np.int64)
    n_symbols = np.array([5, 5], dtype=np.int64)

    returns, valid = compute_sample_returns(
        token_positions, levels, prices, n_coeffs, n_symbols
    )
    assert valid.all()
    assert len(returns) == 2
    # token_pos=0, level=1: coeff_idx=0, bar_start=0, bar_end=2
    # return = (104 - 100) / 100 = 0.04
    assert returns[0] == pytest.approx(0.04, abs=1e-6)


def test_compute_sample_returns_out_of_bounds():
    """Returns NaN for out-of-bounds bar indices."""
    prices = np.array([100.0, 102.0, 104.0])
    token_positions = np.array([10], dtype=np.int64)
    levels = np.array([3], dtype=np.int64)  # 2^3 = 8 bar span, way too large
    n_coeffs = np.array([5], dtype=np.int64)
    n_symbols = np.array([5], dtype=np.int64)

    returns, valid = compute_sample_returns(
        token_positions, levels, prices, n_coeffs, n_symbols
    )
    assert not valid[0]
    assert np.isnan(returns[0])


def test_compute_sample_returns_zero_symbols():
    """Handles zero n_symbols gracefully."""
    prices = np.array([100.0, 102.0])
    token_positions = np.array([0], dtype=np.int64)
    levels = np.array([1], dtype=np.int64)
    n_coeffs = np.array([5], dtype=np.int64)
    n_symbols = np.array([0], dtype=np.int64)

    returns, valid = compute_sample_returns(
        token_positions, levels, prices, n_coeffs, n_symbols
    )
    assert not valid[0]
    assert np.isnan(returns[0])


def test_compute_quantile_boundaries_per_level():
    """Per-level boundaries differ between levels."""
    rng = np.random.default_rng(42)
    n = 1000
    levels = np.array([1] * 500 + [2] * 500, dtype=np.int64)
    # Level 1: small returns, Level 2: large returns
    returns = np.empty(n)
    returns[:500] = rng.normal(0, 0.001, 500)
    returns[500:] = rng.normal(0, 0.01, 500)
    valid = np.ones(n, dtype=np.bool_)

    boundaries = compute_quantile_boundaries(
        returns, levels, valid, per_level=True
    )
    assert 1 in boundaries
    assert 2 in boundaries
    # Level 2 boundaries should be ~10x larger than level 1
    assert np.abs(boundaries[2]).mean() > np.abs(boundaries[1]).mean() * 3


def test_compute_quantile_boundaries_global():
    """Global boundaries ignore level distinctions."""
    rng = np.random.default_rng(42)
    n = 100
    returns = rng.normal(0, 0.01, n)
    levels = np.ones(n, dtype=np.int64)
    valid = np.ones(n, dtype=np.bool_)

    boundaries = compute_quantile_boundaries(
        returns, levels, valid, per_level=False
    )
    assert -1 in boundaries
    assert len(boundaries) == 1
    assert len(boundaries[-1]) == 4  # default 4 boundaries for 5 classes


def test_assign_quantile_labels_basic():
    """Labels are correctly assigned based on boundaries."""
    boundaries = {1: np.array([-0.02, -0.005, 0.005, 0.02])}
    returns = np.array([-0.05, -0.01, 0.0, 0.01, 0.05])
    levels = np.ones(5, dtype=np.int64)

    labels = assign_quantile_labels(returns, levels, boundaries)
    assert labels[0] == 0  # strong_down
    assert labels[1] == 1  # down
    assert labels[2] == 2  # flat
    assert labels[3] == 3  # up
    assert labels[4] == 4  # strong_up


def test_assign_quantile_labels_nan_returns():
    """NaN returns default to FLAT (class 2)."""
    boundaries = {1: np.array([-0.02, -0.005, 0.005, 0.02])}
    returns = np.array([np.nan, 0.01])
    levels = np.ones(2, dtype=np.int64)

    labels = assign_quantile_labels(returns, levels, boundaries)
    assert labels[0] == 2  # NaN → FLAT


def test_compute_return_targets_end_to_end():
    """Full pipeline produces valid ReturnTargetResult."""
    prices = np.array([100.0 + i * 0.5 for i in range(100)])
    n = 20
    token_positions = np.arange(n, dtype=np.int64)
    levels = np.array([1] * 10 + [2] * 10, dtype=np.int64)
    n_coeffs = np.full(n, 50, dtype=np.int64)
    n_symbols = np.full(n, 50, dtype=np.int64)

    result = compute_return_targets(
        token_positions, levels, prices, n_coeffs, n_symbols
    )
    assert isinstance(result, ReturnTargetResult)
    assert len(result.returns) == n
    assert len(result.quantile_labels) == n
    assert result.valid_mask.any()
    assert all(0 <= lab <= 4 for lab in result.quantile_labels)
