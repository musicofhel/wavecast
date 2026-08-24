"""Tests for B2 per-ticker stability (signals/stability.py)."""

from __future__ import annotations

import numpy as np
import pytest

from wavecast.signals.stability import evaluate_stability, spearman, split_window


def _series(n: int = 400):
    ts = np.arange(np.datetime64("2022-01-03T00:00:00"), n, dtype="datetime64[h]")
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0, 0.01, size=n)
    return ts, rets


class TestSplitWindow:
    def test_splits_at_cutoff(self):
        ts, rets = _series()
        (tr_ts, tr_rets), (te_ts, te_rets) = split_window(ts, rets, "2022-01-10")
        assert tr_ts[-1] <= np.datetime64("2022-01-10")
        assert te_ts[0] > np.datetime64("2022-01-10")
        assert len(tr_ts) == len(tr_rets)
        assert len(te_ts) == len(te_rets)
        assert len(tr_ts) + len(te_ts) == len(ts)

    def test_empty_window_raises(self):
        ts, rets = _series()
        with pytest.raises(ValueError, match="empty window"):
            split_window(ts, rets, "2021-01-01")
        with pytest.raises(ValueError, match="empty window"):
            split_window(ts, rets, "2023-01-01")


class TestSpearman:
    def test_perfect_positive(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert spearman(x, x * 2 + 1) == pytest.approx(1.0)

    def test_perfect_negative(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert spearman(x, -x) == pytest.approx(-1.0)

    def test_constant_returns_zero(self):
        assert spearman(np.ones(4), np.arange(4.0)) == 0.0

    def test_ties_averaged(self):
        # [1,1,2] ranks -> [1.5,1.5,3]; against monotone y correlation < 1
        val = spearman([1.0, 1.0, 2.0], [1.0, 2.0, 3.0])
        assert 0.8 < val <= 1.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="equal-length"):
            spearman([1.0, 2.0], [1.0])


class TestEvaluateStability:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(42)
        data = {}
        base = np.datetime64("2022-01-01T00:00:00")
        for i, ticker in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]):
            n = 1200
            ts = base + np.arange(n).astype("timedelta64[D]")
            drift = (i % 3 - 1) * 0.0005
            rets = rng.normal(drift, 0.01, size=n)
            data[(ticker, "1d")] = (ts, rets)
        return data

    def test_basic_shape(self, data):
        res = evaluate_stability(
            ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"], "1d", "persistence", data
        )
        assert res.n_tickers == 6
        assert -1.0 <= res.spearman_sharpe <= 1.0
        assert not np.isnan(res.spearman_sharpe)
        assert res.top_k_test_sharpe is not None

    def test_missing_ticker_skipped(self, data):
        res = evaluate_stability(["AAA", "ZZZ"], "1d", "persistence", data)
        assert res.n_tickers == 1

    def test_deterministic(self, data):
        a = evaluate_stability(sorted(t for t, _ in data), "1d", "momentum", data, params={"lookback": 3})
        b = evaluate_stability(sorted(t for t, _ in data), "1d", "momentum", data, params={"lookback": 3})
        assert a.spearman_sharpe == b.spearman_sharpe

    def test_top_k_selection_reported(self, data):
        res = evaluate_stability(sorted(t for t, _ in data), "1d", "mean_reversion", data, top_k=2)
        # full_test and top_k_test are means over the same test sharpes
        assert isinstance(res.full_test_sharpe, float)
        assert res.full_test_sharpe == pytest.approx(res.test_mean_sharpe)
