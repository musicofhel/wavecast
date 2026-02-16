"""Tests for evaluation metrics."""

import numpy as np

from wavecast.evaluation.metrics import (
    compute_all,
    directional_accuracy,
    mae,
    max_drawdown,
    profit_factor,
    rmse,
    sharpe_ratio,
)


def test_rmse_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_rmse_known():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    expected = np.sqrt(1 / 3)
    np.testing.assert_almost_equal(rmse(y_true, y_pred), expected)


def test_mae_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0


def test_directional_accuracy_perfect():
    y_true = np.array([1.0, -1.0, 1.0])
    y_pred = np.array([0.5, -0.2, 0.3])
    assert directional_accuracy(y_true, y_pred) == 1.0


def test_directional_accuracy_worst():
    y_true = np.array([1.0, -1.0, 1.0])
    y_pred = np.array([-0.5, 0.2, -0.3])
    assert directional_accuracy(y_true, y_pred) == 0.0


def test_sharpe_ratio_positive():
    rng = np.random.default_rng(42)
    returns = 0.001 + 0.005 * rng.standard_normal(252)  # Positive mean, some variance
    s = sharpe_ratio(returns)
    assert s > 0


def test_max_drawdown():
    equity = np.array([100, 110, 105, 95, 100, 90])
    dd = max_drawdown(equity.astype(float))
    # Max drawdown from 110 to 90 = 20/110 ≈ 0.1818
    assert 0.15 < dd < 0.20


def test_profit_factor():
    returns = np.array([0.1, -0.05, 0.2, -0.03])
    pf = profit_factor(returns)
    expected = (0.1 + 0.2) / (0.05 + 0.03)
    np.testing.assert_almost_equal(pf, expected)


def test_compute_all():
    y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    y_pred = np.array([0.005, -0.01, 0.025, 0.005, 0.015])
    metrics = compute_all(y_true, y_pred)
    assert metrics.rmse >= 0
    assert metrics.mae >= 0
    assert 0 <= metrics.directional_accuracy <= 1
