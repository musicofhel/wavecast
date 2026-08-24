"""Timescale comparison utilities (Phase B / B3).

B1 showed raw hourly per-bar flipping is cost-dominated; B4 showed holding
overlays cut turnover and rescue daily rules. Comparing intervals on raw
per-bar results therefore conflates timescale with turnover. This module
normalizes to ANNUAL turnover so an hourly hold=5 cell can be compared against
a daily hold=1 cell on roughly equal trading-activity grounds.
"""

from __future__ import annotations

from typing import Any

# Approximate bars per year by interval (252 trading days; ~7 hourly bars/day).
BARS_PER_YEAR = {"1h": 252 * 7, "1d": 252}


def bars_per_year(interval: str) -> float:
    """Approximate number of bars in one trading year for ``interval``."""
    if interval not in BARS_PER_YEAR:
        raise ValueError(f"unknown interval '{interval}'; known: {sorted(BARS_PER_YEAR)}")
    return float(BARS_PER_YEAR[interval])


def annual_turnover(turnover_per_bar: float, interval: str) -> float:
    """Convert per-bar position-change rate to changes per year."""
    return float(turnover_per_bar) * bars_per_year(interval)


def match_by_turnover(
    rows: list[dict[str, Any]], target_annual_turnover: float, max_abs_diff: float
) -> list[dict[str, Any]]:
    """Rows whose annual turnover is within ``max_abs_diff`` of the target.

    Each input row needs ``interval`` and ``turnover`` keys. Returns one row per
    (ticker, rule, params, interval, hold) group — i.e. filters, not dedups;
    callers aggregate across tickers themselves. Rows are returned sorted by
    absolute distance from the target.
    """
    out = []
    for r in rows:
        dist = abs(annual_turnover(r["turnover"], r["interval"]) - target_annual_turnover)
        if dist <= max_abs_diff:
            out.append({**r, "_turnover_distance": dist})
    return sorted(out, key=lambda r: r["_turnover_distance"])
