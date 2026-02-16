"""Tests for forward test reporting."""

from __future__ import annotations

import json
from pathlib import Path

from wavecast.forward.report import export_forward_json, generate_forward_report
from wavecast.forward.types import ForwardTestSummary


def _make_summary(resolved: int = 5, pending: int = 2) -> ForwardTestSummary:
    """Create a test summary."""
    return ForwardTestSummary(
        test_name="test_v1",
        start_time="2025-06-01T09:00:00",
        tickers=["AAPL", "MSFT"],
        intervals=["1h"],
        total_predictions=resolved + pending,
        resolved_predictions=resolved,
        pending_predictions=pending,
        accuracy=0.6,
        directional_accuracy=0.8,
        cumulative_pnl=0.0234,
        max_drawdown=0.015,
        win_rate=0.65,
        per_ticker={
            "AAPL": {"accuracy": 0.7, "directional_accuracy": 0.85, "cumulative_pnl": 0.02},
            "MSFT": {"accuracy": 0.5, "directional_accuracy": 0.75, "cumulative_pnl": 0.0034},
        },
        per_interval={
            "1h": {"accuracy": 0.6, "directional_accuracy": 0.8, "cumulative_pnl": 0.0234},
        },
    )


class TestForwardReport:
    def test_text_format(self) -> None:
        """Report has expected sections."""
        summary = _make_summary()
        report = generate_forward_report(summary)

        assert "Forward Test Report" in report
        assert "test_v1" in report
        assert "AAPL" in report
        assert "MSFT" in report
        assert "Accuracy" in report
        assert "PnL" in report
        assert "Per Ticker" in report
        assert "Per Interval" in report

    def test_json_export(self, tmp_path: Path) -> None:
        """JSON export produces valid JSON."""
        summary = _make_summary()
        out_path = tmp_path / "report.json"

        export_forward_json(summary, out_path)

        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["test_name"] == "test_v1"
        assert data["resolved_predictions"] == 5
        assert data["accuracy"] == 0.6

    def test_empty_data(self) -> None:
        """Report handles zero resolved predictions."""
        summary = _make_summary(resolved=0, pending=3)
        summary.accuracy = 0.0
        summary.directional_accuracy = 0.0
        summary.cumulative_pnl = 0.0
        summary.max_drawdown = 0.0
        summary.win_rate = 0.0
        summary.per_ticker = {}
        summary.per_interval = {}
        report = generate_forward_report(summary)

        assert "No resolved predictions yet" in report
        assert "Per Ticker" not in report

    def test_per_ticker_breakdown(self) -> None:
        """Per-ticker section renders correctly."""
        summary = _make_summary()
        report = generate_forward_report(summary)

        # Should have both tickers in the per-ticker section
        assert "AAPL" in report
        assert "MSFT" in report
        # Should have accuracy values formatted as percentages
        assert "70.0%" in report  # AAPL accuracy
        assert "50.0%" in report  # MSFT accuracy
