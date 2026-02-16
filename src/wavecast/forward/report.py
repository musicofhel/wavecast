"""Forward test reporting: text and JSON output."""

from __future__ import annotations

import json
from pathlib import Path

from wavecast.forward.types import ForwardTestSummary


def generate_forward_report(summary: ForwardTestSummary) -> str:
    """Generate a formatted text report from a forward test summary."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  Forward Test Report: {summary.test_name}")
    lines.append("=" * 60)
    lines.append(f"  Started: {summary.start_time}")
    lines.append(f"  Tickers: {', '.join(summary.tickers)}")
    lines.append(f"  Intervals: {', '.join(summary.intervals)}")
    lines.append("")

    lines.append("--- Predictions ---")
    lines.append(f"  Total:    {summary.total_predictions}")
    lines.append(f"  Resolved: {summary.resolved_predictions}")
    lines.append(f"  Pending:  {summary.pending_predictions}")
    lines.append("")

    if summary.resolved_predictions > 0:
        lines.append("--- Accuracy ---")
        lines.append(f"  Accuracy:             {summary.accuracy:.1%}")
        lines.append(f"  Directional Accuracy: {summary.directional_accuracy:.1%}")
        lines.append(f"  Win Rate:             {summary.win_rate:.1%}")
        lines.append("")

        lines.append("--- PnL ---")
        lines.append(f"  Cumulative PnL: {summary.cumulative_pnl:+.4f}")
        lines.append(f"  Max Drawdown:   {summary.max_drawdown:.4f}")
        lines.append("")

        if summary.per_ticker:
            lines.append("--- Per Ticker ---")
            lines.append(f"  {'Ticker':<10} {'Accuracy':>10} {'Dir.Acc':>10} {'PnL':>10}")
            lines.append(f"  {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
            for t, metrics in sorted(summary.per_ticker.items()):
                acc = metrics.get("accuracy", 0)
                dir_acc = metrics.get("directional_accuracy", 0)
                pnl = metrics.get("cumulative_pnl", 0)
                lines.append(f"  {t:<10} {acc:>10.1%} {dir_acc:>10.1%} {pnl:>+10.4f}")
            lines.append("")

        if summary.per_interval:
            lines.append("--- Per Interval ---")
            lines.append(f"  {'Interval':<10} {'Accuracy':>10} {'Dir.Acc':>10} {'PnL':>10}")
            lines.append(f"  {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
            for iv, metrics in sorted(summary.per_interval.items()):
                acc = metrics.get("accuracy", 0)
                dir_acc = metrics.get("directional_accuracy", 0)
                pnl = metrics.get("cumulative_pnl", 0)
                lines.append(f"  {iv:<10} {acc:>10.1%} {dir_acc:>10.1%} {pnl:>+10.4f}")
            lines.append("")
    else:
        lines.append("  No resolved predictions yet.")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def export_forward_json(summary: ForwardTestSummary, path: Path) -> None:
    """Export forward test summary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)
