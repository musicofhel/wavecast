"""Tests for transaction cost model."""

from __future__ import annotations

import pytest

from wavecast.signals.costs import TransactionCostModel


class TestTransactionCostModel:
    def test_commission_calculation(self):
        """Commission = trade_value * rate."""
        model = TransactionCostModel(commission_rate=0.001, spread_bps=0.0, slippage_bps=0.0)
        commission, spread, slippage = model.compute(10000.0, direction_change=False)
        assert commission == pytest.approx(10.0)
        assert spread == 0.0
        assert slippage == 0.0

    def test_spread_calculation(self):
        """Spread = trade_value * bps / 10000."""
        model = TransactionCostModel(commission_rate=0.0, spread_bps=2.0, slippage_bps=0.0)
        commission, spread, slippage = model.compute(10000.0, direction_change=False)
        assert spread == pytest.approx(2.0)

    def test_slippage_calculation(self):
        """Slippage = trade_value * bps / 10000."""
        model = TransactionCostModel(commission_rate=0.0, spread_bps=0.0, slippage_bps=1.0)
        _, _, slippage = model.compute(10000.0, direction_change=False)
        assert slippage == pytest.approx(1.0)

    def test_direction_change_doubles_costs(self):
        """Direction change = close old + open new = 2× costs."""
        model = TransactionCostModel(commission_rate=0.001, spread_bps=2.0, slippage_bps=1.0)
        total_no_change = model.total(10000.0, direction_change=False)
        total_change = model.total(10000.0, direction_change=True)
        assert total_change == pytest.approx(total_no_change * 2)

    def test_zero_cost_model(self):
        """Zero-cost model for testing."""
        model = TransactionCostModel.zero()
        assert model.total(10000.0, direction_change=True) == 0.0
