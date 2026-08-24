"""Tests for B3 timescale utilities."""

from __future__ import annotations

import pytest

from wavecast.signals.timescale import (
    annual_turnover,
    bars_per_year,
    match_by_turnover,
)


def test_bars_per_year_known_intervals():
    assert bars_per_year("1d") == 252.0
    assert bars_per_year("1h") == 252.0 * 7
    with pytest.raises(ValueError, match="unknown interval"):
        bars_per_year("5m")


def test_annual_turnover_scales_by_interval():
    # same per-bar turnover is ~7x more active annually on hourly
    assert annual_turnover(0.1, "1h") == pytest.approx(0.7 * 252)
    assert annual_turnover(0.1, "1d") == pytest.approx(25.2)


def _row(turnover: float, interval: str) -> dict:
    return {"ticker": "X", "interval": interval, "turnover": turnover}


def test_match_by_turnover_filters_and_sorts():
    rows = [
        _row(0.12, "1d"),   # 30.24/yr -> dist 9.76
        _row(0.02, "1h"),   # 35.28/yr -> dist 4.72
        _row(0.01, "1d"),   # 2.52/yr  -> excluded
        _row(0.15, "1d"),   # 37.8/yr  -> dist 2.2
    ]
    out = match_by_turnover(rows, target_annual_turnover=35.0, max_abs_diff=10.0)
    assert [r["turnover"] for r in out] == [0.02, 0.15, 0.12]
    assert all("_turnover_distance" in r for r in out)


def test_match_by_turnover_empty_when_nothing_close():
    assert match_by_turnover([_row(0.01, "1d")], 100.0, 5.0) == []
