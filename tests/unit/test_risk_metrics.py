"""Tests for risk metric functions."""

from __future__ import annotations

import numpy as np
import pytest

from wavecast.evaluation.metrics import (
    avg_win_loss_ratio,
    calmar_ratio,
    conditional_var,
    expectancy,
    max_drawdown,
    sortino_ratio,
    tail_ratio,
    value_at_risk,
    win_rate,
)


class TestSortinoRatio:
    def test_positive_returns(self):
        """All positive returns → no downside deviation → 0."""
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.005])
        result = sortino_ratio(returns)
        assert result == 0.0  # no downside returns

    def test_mixed_returns(self):
        """Mixed returns should produce nonzero Sortino."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 252)
        result = sortino_ratio(returns)
        assert isinstance(result, float)
        assert result != 0.0


class TestCalmarRatio:
    def test_no_drawdown(self):
        """Monotonically increasing equity → zero drawdown → 0."""
        returns = np.array([0.01] * 100)
        result = calmar_ratio(returns)
        assert result == 0.0  # max_drawdown returns 0 for monotonic increase

    def test_with_drawdown(self):
        """Returns with drawdown should produce finite Calmar."""
        returns = np.array([0.05, -0.10, 0.03, -0.02, 0.04])
        result = calmar_ratio(returns)
        assert isinstance(result, float)


class TestValueAtRisk:
    def test_known_distribution(self):
        """VaR at 95% on uniform returns."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 10000)
        var95 = value_at_risk(returns, confidence=0.95)
        assert var95 > 0  # VaR should be positive
        assert abs(var95 - 0.0165) < 0.003  # ~1.65 std devs for 95%

    def test_empty_returns(self):
        assert value_at_risk(np.array([]), confidence=0.95) == 0.0


class TestConditionalVaR:
    def test_cvar_exceeds_var(self):
        """CVaR should be >= VaR (it's the expected loss beyond VaR)."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 5000)
        var = value_at_risk(returns, 0.95)
        cvar = conditional_var(returns, 0.95)
        assert cvar >= var

    def test_empty(self):
        assert conditional_var(np.array([]), 0.95) == 0.0


class TestWinRate:
    def test_all_positive(self):
        returns = np.array([0.01, 0.02, 0.03])
        assert win_rate(returns) == 1.0

    def test_all_negative(self):
        returns = np.array([-0.01, -0.02, -0.03])
        assert win_rate(returns) == 0.0

    def test_mixed(self):
        returns = np.array([0.01, -0.01, 0.02, -0.02])
        assert win_rate(returns) == 0.5


class TestAvgWinLossRatio:
    def test_symmetric(self):
        """Equal wins and losses → ratio 1.0."""
        returns = np.array([0.01, -0.01, 0.01, -0.01])
        assert avg_win_loss_ratio(returns) == pytest.approx(1.0)

    def test_no_losses(self):
        returns = np.array([0.01, 0.02])
        assert avg_win_loss_ratio(returns) == 0.0  # no losses → 0


class TestExpectancy:
    def test_positive_edge(self):
        """60% wins at 1:1 ratio → positive expectancy."""
        returns = np.array([0.01] * 60 + [-0.01] * 40)
        result = expectancy(returns)
        assert result > 0

    def test_zero_returns(self):
        assert expectancy(np.array([])) == 0.0


class TestTailRatio:
    def test_symmetric(self):
        """Symmetric distribution → tail ratio close to 1."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 1000)
        result = tail_ratio(returns)
        assert abs(result - 1.0) < 0.3  # approximately symmetric

    def test_too_few_samples(self):
        returns = np.array([0.01, -0.01])
        assert tail_ratio(returns) == 0.0
