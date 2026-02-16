"""Tests for SignalBacktest engine."""

from __future__ import annotations

import numpy as np
import pytest

from wavecast.signals.backtest import SignalBacktest
from wavecast.signals.costs import TransactionCostModel
from wavecast.signals.position import PositionSizer
from wavecast.signals.types import SignalSeries, TradingSignal


def _make_signals(
    directions: list[int], confidences: list[float] | None = None
) -> SignalSeries:
    """Helper to create signal series from direction list."""
    n = len(directions)
    if confidences is None:
        confidences = [0.8] * n
    start = np.datetime64("2024-01-01")
    timestamps = np.arange(start, start + np.timedelta64(n, "D"), np.timedelta64(1, "D"))
    signals = [
        TradingSignal(
            timestamp=timestamps[i],
            direction=directions[i],
            confidence=confidences[i],
            raw_probability=0.5,
            token_id=50,
        )
        for i in range(n)
    ]
    return SignalSeries(signals=signals, ticker="TEST")


class TestAllLong:
    def test_all_long_positive_market(self):
        """All long signals in rising market -> positive returns."""
        signals = _make_signals([1] * 100)
        returns = np.full(100, 0.001)  # 0.1% per period
        bt = SignalBacktest(
            cost_model=TransactionCostModel.zero(),
            position_sizer=PositionSizer(method="fixed", max_position=1.0),
        )
        result = bt.run(signals, returns)
        assert result.returns.sum() > 0
        assert result.equity_curve[-1] > result.equity_curve[0]


class TestAlternating:
    def test_alternating_incurs_costs(self):
        """Alternating long/short should incur high transaction costs."""
        dirs = [1, -1] * 50
        signals = _make_signals(dirs)
        returns = np.full(100, 0.001)
        # With costs
        bt_cost = SignalBacktest(
            cost_model=TransactionCostModel(commission_rate=0.01),
            position_sizer=PositionSizer(method="fixed"),
        )
        result_cost = bt_cost.run(signals, returns)
        # Without costs
        bt_free = SignalBacktest(
            cost_model=TransactionCostModel.zero(),
            position_sizer=PositionSizer(method="fixed"),
        )
        result_free = bt_free.run(signals, returns)
        assert result_cost.equity_curve[-1] < result_free.equity_curve[-1]


class TestHold:
    def test_flat_signals_no_trading(self):
        """All flat signals -> no position changes, returns = 0."""
        signals = _make_signals([0] * 50)
        returns = np.full(50, 0.01)
        bt = SignalBacktest(cost_model=TransactionCostModel.zero())
        result = bt.run(signals, returns)
        assert np.allclose(result.returns, 0.0)


class TestConfidenceSizing:
    def test_higher_confidence_larger_position(self):
        """Higher confidence -> larger position with linear sizing."""
        high_conf = _make_signals([1] * 10, [0.9] * 10)
        low_conf = _make_signals([1] * 10, [0.3] * 10)
        returns = np.full(10, 0.01)
        bt = SignalBacktest(
            cost_model=TransactionCostModel.zero(),
            position_sizer=PositionSizer(method="linear", max_position=1.0),
        )
        result_high = bt.run(high_conf, returns)
        bt.position_sizer.reset()
        result_low = bt.run(low_conf, returns)
        assert result_high.equity_curve[-1] > result_low.equity_curve[-1]


class TestMetricsComplete:
    def test_all_16_metrics_present(self):
        """Result should contain all 16 metrics."""
        signals = _make_signals([1, -1, 1, -1, 1] * 20)
        returns = np.random.default_rng(42).normal(0.001, 0.01, 100)
        bt = SignalBacktest()
        result = bt.run(signals, returns)
        expected_metrics = {
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown",
            "profit_factor",
            "value_at_risk_95",
            "conditional_var_95",
            "win_rate",
            "avg_win_loss_ratio",
            "expectancy",
            "tail_ratio",
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "num_trades",
            "avg_trade_return",
        }
        assert expected_metrics.issubset(set(result.metrics.keys()))


class TestEquityStart:
    def test_equity_starts_at_initial_capital(self):
        """Equity curve first value should reflect initial capital."""
        signals = _make_signals([1])
        returns = np.array([0.0])
        bt = SignalBacktest(
            cost_model=TransactionCostModel.zero(),
            initial_capital=50000.0,
        )
        result = bt.run(signals, returns)
        assert result.equity_curve[0] == pytest.approx(50000.0)


class TestPerTrade:
    def test_trade_records_created(self):
        """Direction changes should create trade records."""
        signals = _make_signals([1, 1, 1, -1, -1, -1, 1, 1])
        returns = np.full(8, 0.001)
        bt = SignalBacktest(cost_model=TransactionCostModel.zero())
        result = bt.run(signals, returns)
        # Should have at least 2 trades (long->short, short->long, plus final close)
        assert len(result.trades) >= 2


class TestLengthMismatch:
    def test_raises_on_length_mismatch(self):
        """Should raise ValueError when signals and returns have different lengths."""
        signals = _make_signals([1] * 10)
        returns = np.full(5, 0.001)
        bt = SignalBacktest()
        with pytest.raises(ValueError, match="Signal length"):
            bt.run(signals, returns)
