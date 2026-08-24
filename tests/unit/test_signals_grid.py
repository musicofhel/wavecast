"""Tests for the B1 backtest grid harness."""

from __future__ import annotations

import json

import numpy as np
import pytest

from wavecast.signals.grid import (
    COST_BPS_ROUND_TRIP,
    TRADING_RULES,
    GridCell,
    append_results,
    cost_model_for,
    load_results,
    persistence_baseline_sharpe,
    rule_mean_reversion,
    rule_model,
    rule_momentum,
    rule_persistence,
    run_cell,
    run_grid,
)


def _make_series(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.01, size=n)
    timestamps = np.arange(np.datetime64("2026-01-01T00:00"), n * 60, dtype="datetime64[m]")
    return timestamps, returns


class TestRules:
    def test_all_rules_registered(self):
        assert {"persistence", "mean_reversion", "momentum", "model"} <= set(TRADING_RULES)

    def test_persistence_copies_lagged_sign(self):
        timestamps, returns = _make_series(50)
        series = rule_persistence(returns, timestamps)
        dirs = series.directions
        assert dirs[0] == 0  # no prior bar -> flat
        assert np.array_equal(dirs[1:], np.sign(returns[:-1]))

    def test_mean_reversion_is_negated_persistence(self):
        timestamps, returns = _make_series(50)
        mr = rule_mean_reversion(returns, timestamps).directions
        per = rule_persistence(returns, timestamps).directions
        assert np.array_equal(mr, -per)

    def test_momentum_flat_until_lookback(self):
        timestamps, returns = _make_series(20)
        series = rule_momentum(returns, timestamps, lookback=3)
        dirs = series.directions
        assert np.all(dirs[:3] == 0)
        # spot check one bar
        assert dirs[5] == np.sign(np.sum(returns[2:5]))

    def test_model_rule_requires_predictions(self):
        timestamps, returns = _make_series(10)
        with pytest.raises(ValueError, match="predicted_directions"):
            rule_model(returns, timestamps)

    def test_model_rule_length_mismatch(self):
        timestamps, returns = _make_series(10)
        with pytest.raises(ValueError, match="!="):
            rule_model(returns, timestamps, predicted_directions=np.ones(5))


class TestCostModel:
    def test_round_trip_split(self):
        cm = cost_model_for(COST_BPS_ROUND_TRIP)
        assert cm.spread_bps == pytest.approx(3.5)
        # direction change doubles the leg -> full round trip = 7bps of trade value
        total = cm.total(10000.0, direction_change=True)
        assert total == pytest.approx(7.0)


class TestRunCell:
    def test_result_row_shape(self):
        timestamps, returns = _make_series()
        cell = GridCell(ticker="SPY", interval="1h", rule="persistence")
        row = run_cell(cell, returns, timestamps)
        assert row["ticker"] == "SPY"
        assert row["rule"] == "persistence"
        assert row["n_bars"] == 200
        assert row["cost_bps_round_trip"] == 7.0
        for key in ("sharpe", "total_return", "expectancy", "win_rate", "num_trades",
                    "max_drawdown", "flat_rate", "cell_key", "start", "end"):
            assert key in row

    def test_unknown_rule_raises(self):
        timestamps, returns = _make_series()
        with pytest.raises(ValueError, match="unknown rule"):
            run_cell(GridCell(ticker="X", interval="1h", rule="nope"), returns, timestamps)

    def test_persistence_baseline_matches_run_cell(self):
        timestamps, returns = _make_series()
        direct = persistence_baseline_sharpe(returns, timestamps)
        via_cell = run_cell(
            GridCell(ticker="b", interval="1h", rule="persistence"), returns, timestamps
        )["sharpe"]
        assert direct == via_cell


class TestGrid:
    def test_grid_runs_and_attaches_baseline(self):
        ts1, r1 = _make_series(seed=1)
        ts2, r2 = _make_series(n=150, seed=2)
        data = {("SPY", "1h"): (ts1, r1),("AAPL", "1d"): (ts2, r2)}
        cells = [
            GridCell(ticker="SPY", interval="1h", rule="momentum", params={"lookback": 4}),
            GridCell(ticker="AAPL", interval="1d", rule="mean_reversion"),
            GridCell(ticker="AAPL", interval="1d", rule="persistence"),
        ]
        rows = run_grid(cells, data)
        assert len(rows) == 3
        by_key = {(r["ticker"], r["rule"]): r for r in rows}
        # persistence cell gets no baseline (it IS the baseline)
        assert "baseline_persistence_sharpe" not in by_key[("AAPL", "persistence")]
        assert "baseline_persistence_sharpe" in by_key[("AAPL", "mean_reversion")]
        assert "baseline_persistence_sharpe" in by_key[("SPY", "momentum")]

    def test_missing_data_key(self):
        ts, r = _make_series(10)
        with pytest.raises(KeyError, match="MSFT"):
            run_grid([GridCell(ticker="MSFT", interval="1h", rule="persistence")],
                     {("SPY", "1h"): (ts, r)})

    def test_model_rule_via_grid(self):
        ts, r = _make_series()
        preds = np.concatenate([[0.0], np.sign(r[:-1])])
        confs = np.full(len(r), 0.9)
        rows = run_grid(
            [GridCell(ticker="SPY", interval="1h", rule="model")],
            {("SPY", "1h"): (ts, r)},
            predictions={("SPY", "1h"): (preds, confs)},
            cost_bps=0.0,
        )
        assert rows[0]["flat_rate"] == pytest.approx(1 / len(r))  # only the seed bar is flat


class TestLedger:
    def test_append_and_load_roundtrip(self, tmp_path):
        p = tmp_path / "results.jsonl"
        rows = [{"cell_key": "abc", "sharpe": 1.5}, {"cell_key": "def", "sharpe": -0.2}]
        append_results(rows, p)
        append_results([{"cell_key": "ghi"}], p)
        loaded = load_results(p)
        assert len(loaded) == 3
        assert loaded[0]["sharpe"] == 1.5

    def test_load_missing_file_returns_empty(self, tmp_path):
        assert load_results(tmp_path / "nope.jsonl") == []

    def test_json_serializable(self, tmp_path):
        ts, r = _make_series()
        rows = run_grid([GridCell(ticker="SPY", interval="1h", rule="persistence")],
                        {("SPY", "1h"): (ts, r)})
        p = tmp_path / "out.jsonl"
        append_results(rows, p)
        parsed = [json.loads(line) for line in p.read_text().splitlines()]
        assert parsed[0]["rule"] == "persistence"

    def test_cell_key_stable_and_param_sensitive(self):
        a = GridCell(ticker="SPY", interval="1h", rule="momentum", params={"lookback": 3})
        b = GridCell(ticker="SPY", interval="1h", rule="momentum", params={"lookback": 4})
        c = GridCell(ticker="SPY", interval="1h", rule="momentum", params={"lookback": 3})
        assert a.key() == c.key()
        assert a.key() != b.key()
