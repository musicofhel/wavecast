"""Tests for B4 trade-management variants (holding periods, vol-scaled sizing)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wavecast.signals.grid import GridCell
from wavecast.signals.trade_mgmt import (
    apply_holding,
    run_mgmt_cell,
    vol_scaled_confidence,
)


def _ts(n: int) -> np.ndarray:
    return pd.date_range("2024-01-01", periods=n, freq="D").to_numpy(dtype="datetime64[ns]")


class TestApplyHolding:
    def test_hold_one_is_identity(self):
        d = np.array([1, -1, 1, 1, 0], dtype=np.float64)
        np.testing.assert_array_equal(apply_holding(d, 1), d)

    def test_hold_three_forward_fills(self):
        d = np.array([1, -1, -1, -1, 1, 1], dtype=np.float64)
        out = apply_holding(d, 3)
        np.testing.assert_array_equal(out, [1, 1, 1, -1, -1, -1])

    def test_invalid_hold_raises(self):
        with pytest.raises(ValueError):
            apply_holding(np.array([1.0]), 0)

    def test_empty_input(self):
        assert len(apply_holding(np.array([]), 3)) == 0


class TestVolScaledConfidence:
    def test_insufficient_history_is_flat(self):
        r = np.array([0.01] * 25)
        conf = vol_scaled_confidence(r, lookback=20, target_vol=0.01)
        np.testing.assert_allclose(conf[:20], 0.0)

    def test_low_vol_gets_full_position(self):
        r = np.full(40, 0.001) + np.linspace(0, 1e-6, 40)
        conf = vol_scaled_confidence(r, lookback=20, target_vol=0.05)
        assert conf[-1] == pytest.approx(1.0)

    def test_high_vol_gets_scaled_down(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0, 0.05, 100)
        conf = vol_scaled_confidence(r, lookback=20, target_vol=0.01)
        assert 0 < conf[-1] < 1

    def test_zero_std_window_stays_flat(self):
        r = np.zeros(30)
        conf = vol_scaled_confidence(r, lookback=10, target_vol=0.01)
        np.testing.assert_allclose(conf, 0.0)

    def test_bad_lookback_raises(self):
        with pytest.raises(ValueError):
            vol_scaled_confidence(np.ones(5), lookback=1)


class TestRunMgmtCell:
    def _series(self, n: int = 120) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        return _ts(n), rng.normal(0, 0.01, n)

    def test_hold_reduces_turnover_and_trades(self):
        ts, rets = self._series()
        cell = GridCell(ticker="X", interval="1d", rule="persistence")
        h1 = run_mgmt_cell(cell, rets, ts, hold=1)
        h5 = run_mgmt_cell(cell, rets, ts, hold=5)
        assert h5["turnover"] < h1["turnover"]
        assert h5["num_trades"] <= h1["num_trades"]

    def test_unknown_rule_raises(self):
        ts, rets = self._series()
        with pytest.raises(ValueError):
            run_mgmt_cell(GridCell(ticker="X", interval="1d", rule="nope"), rets, ts)

    def test_model_rule_rejected(self):
        ts, rets = self._series()
        with pytest.raises(ValueError):
            run_mgmt_cell(GridCell(ticker="X", interval="1d", rule="model"), rets, ts)

    def test_unknown_sizing_raises(self):
        ts, rets = self._series()
        cell = GridCell(ticker="X", interval="1d", rule="persistence")
        with pytest.raises(ValueError):
            run_mgmt_cell(cell, rets, ts, sizing="kelly_crazy")

    def test_result_schema(self):
        ts, rets = self._series()
        cell = GridCell(ticker="X", interval="1d", rule="mean_reversion")
        row = run_mgmt_cell(cell, rets, ts, hold=2, sizing="vol_scaled")
        for key in ("sharpe", "total_return", "num_trades", "hold", "sizing",
                    "turnover", "flat_rate", "cost_bps_round_trip"):
            assert key in row
        assert row["rule"] == "mean_reversion"
        assert 0 <= row["flat_rate"] <= 1
        assert 0 <= row["turnover"] <= 1

    def test_hold_one_matches_plain_persistence_directions(self):
        # With hold=1 and flat sizing the result must equal the B1 harness cell.
        from wavecast.signals.grid import run_cell

        ts, rets = self._series()
        cell = GridCell(ticker="X", interval="1d", rule="persistence")
        mgmt = run_mgmt_cell(cell, rets, ts, hold=1, sizing="flat")
        plain = run_cell(cell, rets, ts)
        assert mgmt["sharpe"] == pytest.approx(plain["sharpe"])
