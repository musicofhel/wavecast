"""Tests for honest forward-ledger evaluation (economic metrics only)."""

import json

import pytest

from wavecast.evaluation.forward_ledger import (
    evaluate,
    format_report,
    load_ledger,
    run_positions,
    trade_stats,
    wilson_ci,
)


def _row(ticker="AAPL", ts="2026-03-01T14:00:00", direction=1, ret=0.001, a2i=False):
    actual = 1 if ret > 0 else -1 if ret < 0 else 0
    return {
        "id": "x",
        "timestamp": ts,
        "target_timestamp": ts,
        "ticker": ticker,
        "interval": "1h",
        "horizon": 1,
        "predicted_direction": direction,
        "actual_return": ret,
        "actual_direction": actual,
        "correct": direction == actual,
        "a2i_trade": a2i,
    }


class TestWilsonCI:
    def test_full_sample_bounds(self):
        lo, hi = wilson_ci(50, 100)
        assert 0.40 < lo < 0.50 < hi < 0.60

    def test_zero_total(self):
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_perfect_score_still_has_lower_bound(self):
        lo, hi = wilson_ci(10, 10)
        assert hi <= 1.0
        assert lo > 0.5


class TestLoadLedger:
    def test_skips_unresolved(self, tmp_path):
        p = tmp_path / "ledger.jsonl"
        resolved = _row()
        pending = dict(_row(), actual_return=None, actual_direction=None, correct=None)
        p.write_text("\n".join(json.dumps(r) for r in [resolved, pending]))
        rows = load_ledger(str(p))
        assert len(rows) == 1
        assert rows[0]["ticker"] == "AAPL"

    def test_sorts_by_target_time(self, tmp_path):
        p = tmp_path / "ledger.jsonl"
        p.write_text(
            json.dumps(_row(ts="2026-03-02T14:00:00"))
            + "\n"
            + json.dumps(_row(ts="2026-03-01T14:00:00"))
        )
        rows = load_ledger(str(p))
        assert rows[0]["target_timestamp"] < rows[1]["target_timestamp"]


class TestEvaluate:
    def test_accuracy_matches_correct_flags(self):
        rows = [_row(direction=1, ret=0.01), _row(direction=-1, ret=0.01), _row()]
        rep = evaluate(rows)
        # row2 wrong, others right
        assert rep.accuracy == pytest.approx(2 / 3)
        assert rep.n_resolved == 3

    def test_flat_rate_counts_zero_directions(self):
        rows = [_row(direction=0), _row()]
        rep = evaluate(rows)
        assert rep.flat_rate == pytest.approx(0.5)

    def test_filtered_subset(self):
        rows = [_row(a2i=True), _row(a2i=True, direction=-1, ret=-0.01), _row()]
        rep = evaluate(rows)
        assert rep.filtered_n == 2
        assert rep.filtered_accuracy == pytest.approx(1.0)

    def test_no_filtered_rows_gives_none(self):
        rep = evaluate([_row()])
        assert rep.filtered_accuracy is None

    def test_per_ticker_and_month_buckets(self):
        rows = [
            _row(ticker="MSFT", ts="2026-03-05T14:00:00"),
            _row(ts="2026-04-05T14:00:00"),
            _row(ts="2026-04-06T14:00:00"),
        ]
        rep = evaluate(rows)
        assert set(rep.per_ticker) == {"AAPL", "MSFT"}
        assert set(rep.per_month) == {"2026-03", "2026-04"}
        assert rep.per_month["2026-04"]["n"] == 2


class TestTradeStats:
    def test_model_beats_random_baseline_on_known_data(self):
        # Model always predicts up; returns alternate so always-up loses.
        rows = [_row(ret=r) for r in [0.01, -0.02, 0.01, -0.02]]
        stats = trade_stats(rows, cost_bps=7.0)
        up = stats["always_up"]
        down = stats["always_down"]
        assert down["total_return"] > up["total_return"]

    def test_costs_reduce_return_on_churn(self):
        rows = [_row(ret=0.001), _row(ret=0.001)]
        flat = run_positions(rows, [1, 1], cost=7e-4)
        churn = run_positions(rows, [1, -1], cost=7e-4)
        assert churn["total_return"] < flat["total_return"]

    def test_persistence_uses_prior_actual(self):
        # persistence predicts previous actual sign; first row is flat (no history)
        rows = [_row(ret=0.01), _row(ret=0.01)]
        stats = trade_stats(rows, cost_bps=0.0)
        assert stats["persistence"]["n"] == 2
        # first opportunity held flat -> zero return on it
        assert stats["persistence"]["total_return"] == pytest.approx(0.01)

    def test_a2i_filter_flattens_non_trades(self):
        rows = [_row(ret=0.01, a2i=False), _row(ret=0.01, a2i=True)]
        stats = trade_stats(rows, cost_bps=0.0)
        assert stats["model_a2i_filtered"]["total_return"] == pytest.approx(0.01)

    def test_sharpe_sign_follows_expectancy(self):
        rows = [_row(ret=r) for r in [-0.01] * 19 + [0.05]]
        stats = trade_stats(rows, cost_bps=0.0)
        assert stats["always_down"]["sharpe_annualized"] > 0
        assert stats["always_up"]["sharpe_annualized"] < 0


class TestFormatReport:
    def test_contains_key_sections(self):
        rep = evaluate([_row(a2i=True)])
        text = format_report(rep)
        assert "economic directional accuracy" in text
        assert "flat-prediction rate" in text
        assert "A2i-filtered" in text
        assert "per-ticker" in text
