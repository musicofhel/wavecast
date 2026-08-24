"""Per-ticker performance stability (Phase B / B2).

Splits each ticker's return series into train/test windows, evaluates grid
cells on both, and measures whether per-ticker backtest quality is stable:
Spearman rank correlation of train vs test Sharpe, and whether a top-K
sub-universe picked on train holds up out-of-sample versus the full universe.

Uses the B1 grid harness for cell evaluation; rule-based only (no model).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from wavecast.signals.grid import GridCell, run_cell


def split_window(
    timestamps: NDArray[np.datetime64],
    returns: NDArray[np.float64],
    train_end: str,
) -> tuple[tuple[NDArray[np.datetime64], NDArray[np.float64]], ...]:
    """Split a series into (train, test) windows at ``train_end`` (inclusive)."""
    cutoff = np.datetime64(train_end)
    mask = timestamps <= cutoff
    if mask.all() or not mask.any():
        raise ValueError(f"train_end {train_end} leaves an empty window")
    return (timestamps[mask], returns[mask]), (timestamps[~mask], returns[~mask])


def spearman(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Spearman rank correlation with tie-averaged ranks."""

    def ranks(a: NDArray[np.float64]) -> NDArray[np.float64]:
        a = np.asarray(a, dtype=np.float64)
        order = np.argsort(a, kind="stable")
        r = np.empty(len(a), dtype=np.float64)
        r[order] = np.arange(1.0, len(a) + 1.0)
        for v in np.unique(a):
            idx = a == v
            if idx.sum() > 1:
                r[idx] = r[idx].mean()
        return r

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("spearman needs equal-length arrays of length >= 2")
    rx, ry = ranks(x), ranks(y)
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = np.sqrt(float((rx_c**2).sum()) * float((ry_c**2).sum()))
    if denom == 0.0:
        return 0.0
    return float((rx_c * ry_c).sum() / denom)


@dataclass
class StabilityResult:
    """Train→test stability summary for one (interval, rule, params) config."""

    interval: str
    rule: str
    params: dict = field(default_factory=dict)
    n_tickers: int = 0
    spearman_sharpe: float = 0.0
    train_mean_sharpe: float = 0.0
    test_mean_sharpe: float = 0.0
    top_k_train_sharpe: float = 0.0
    top_k_test_sharpe: float = 0.0
    full_test_sharpe: float = 0.0
    n_positive_train: int = 0
    n_positive_test: int = 0


def evaluate_stability(
    tickers: list[str],
    interval: str,
    rule: str,
    data: dict[tuple[str, str], tuple[NDArray[np.datetime64], NDArray[np.float64]]],
    params: dict | None = None,
    train_end: str = "2024-06-30",
    cost_bps: float = 7.0,
    top_k: int = 5,
) -> StabilityResult:
    """Run one config per ticker on train and test windows; summarize stability."""
    train_sharpes: list[float] = []
    test_sharpes: list[float] = []
    for ticker in tickers:
        key = (ticker, interval)
        if key not in data:
            continue
        ts, rets = data[key]
        try:
            (tr_ts, tr_rets), (te_ts, te_rets) = split_window(ts, rets, train_end)
        except ValueError:
            continue
        cell = GridCell(ticker=ticker, interval=interval, rule=rule, params=params or {})
        train_sharpes.append(run_cell(cell, tr_rets, tr_ts, cost_bps=cost_bps)["sharpe"])
        test_sharpes.append(run_cell(cell, te_rets, te_ts, cost_bps=cost_bps)["sharpe"])

    tr = np.asarray(train_sharpes, dtype=np.float64)
    te = np.asarray(test_sharpes, dtype=np.float64)
    rho = spearman(tr, te) if len(tr) >= 2 else float("nan")
    k = min(top_k, len(tr))
    order = np.argsort(tr)[::-1]
    return StabilityResult(
        interval=interval,
        rule=rule,
        params=dict(params or {}),
        n_tickers=len(tr),
        spearman_sharpe=rho,
        train_mean_sharpe=float(tr.mean()) if len(tr) else 0.0,
        test_mean_sharpe=float(te.mean()) if len(te) else 0.0,
        top_k_train_sharpe=float(tr[order[:k]].mean()),
        top_k_test_sharpe=float(te[order[:k]].mean()),
        full_test_sharpe=float(te.mean()) if len(te) else 0.0,
        n_positive_train=int((tr > 0).sum()),
        n_positive_test=int((te > 0).sum()),
    )
