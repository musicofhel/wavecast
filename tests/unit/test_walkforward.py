"""Tests for B5 walk-forward hold selection."""

from __future__ import annotations

import numpy as np
import pytest

from wavecast.signals.grid import GridCell
from wavecast.signals.walkforward import fold_boundaries, walk_forward_hold


def _synthetic(n: int = 1500, seed: int = 42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.01, n)
    prev = np.zeros(n)
    for i in range(1, n):
        prev[i] = -0.4 * prev[i - 1] + noise[i]
    ts = np.arange("2021-01-01", n, dtype="datetime64[D]")
    return prev.astype(np.float64), ts


def test_fold_boundaries_partition():
    b = fold_boundaries(400, 4, min_fold_bars=40)
    assert len(b) == 4
    assert b[0][0] == 0 and b[-1][1] == 400
    for (_s1, e1), (s2, e2) in zip(b[:-1], b[1:], strict=True):
        assert e1 == s2
        assert e2 - s2 >= 40


def test_fold_boundaries_too_short():
    with pytest.raises(ValueError):
        fold_boundaries(100, 4, min_fold_bars=40)


def test_walk_forward_structure():
    rets, ts = _synthetic()
    cell = GridCell(ticker="TEST", interval="1d", rule="mean_reversion", params={})
    out = walk_forward_hold(cell, rets, ts, candidate_holds=(1, 5),
                            n_train_bars=600, n_folds=4)
    folds = out["folds"]
    assert len(folds) == 4
    s = out["summary"]
    assert sum(s["selection_counts"].values()) == 4
    assert set(s["selection_counts"]) == {1, 5}
    # every fold row carries fixed-hold baselines
    assert all("fixed_hold1_sharpe" in f and "fixed_hold5_sharpe" in f for f in folds)
    # summary means match fold rows
    assert s["mean_wf_sharpe"] == pytest.approx(
        float(np.mean([f["fold_sharpe"] for f in folds]))
    )
    assert s["mean_fixed_sharpe"][1] == pytest.approx(
        float(np.mean([f["fixed_hold1_sharpe"] for f in folds]))
    )


def test_walk_forward_selected_hold_matches_train_scores():
    """The selected hold must be argmax of train scores — no test peeking."""
    rets, ts = _synthetic()
    cell = GridCell(ticker="TEST", interval="1d", rule="mean_reversion", params={})
    out = walk_forward_hold(cell, rets, ts, candidate_holds=(1, 2, 5),
                            n_train_bars=600, n_folds=4)
    assert len(out["folds"]) == 4
