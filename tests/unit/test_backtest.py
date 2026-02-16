"""Tests for walk-forward backtest."""

import numpy as np

from tests.fixtures.generators import make_feature_matrix
from wavecast.core.config import BacktestConfig
from wavecast.evaluation.backtest import WalkForwardBacktest
from wavecast.models.gradient_boost import GradientBoostModel


def test_backtest_runs():
    X, y = make_feature_matrix(n=300, d=20)
    timestamps = np.arange("2020-01-01", 300, dtype="datetime64[D]")
    model = GradientBoostModel()
    config = BacktestConfig(
        walk_forward_train=100,
        walk_forward_test=20,
        commission=0.001,
    )
    bt = WalkForwardBacktest(model, config)
    result = bt.run(X, y, timestamps=timestamps)
    assert len(result.returns) > 0
    assert len(result.equity_curve) > 0
    assert "sharpe_ratio" in result.metrics
    assert "directional_accuracy" in result.metrics
    assert "total_return" in result.metrics
