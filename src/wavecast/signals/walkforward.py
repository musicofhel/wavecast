"""Walk-forward parameter stability (Phase B / B5).

B4's headline (hold=5 rescues daily mean-reversion) was selected on the same
window it was reported on — classic peeking. This module answers honestly:
split the OOS region into folds; for each fold pick the hold that maximized
train Sharpe on everything BEFORE the fold, then score that choice on the
fold. Reports how often each hold is re-selected and the honest walk-forward
Sharpe, alongside fixed-hold baselines computed on identical folds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wavecast.signals.grid import COST_BPS_ROUND_TRIP, GridCell
from wavecast.signals.trade_mgmt import run_mgmt_cell


def fold_boundaries(
    n_test: int, n_folds: int, min_fold_bars: int = 40
) -> list[tuple[int, int]]:
    """Equal-width ``(start, end)`` boundaries covering ``[0, n_test)``.

    Raises ValueError if any fold would be shorter than ``min_fold_bars``.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    width = n_test // n_folds
    if width < min_fold_bars:
        raise ValueError(
            f"fold too short: n_test={n_test} / n_folds={n_folds} "
            f"= {width} < min_fold_bars={min_fold_bars}"
        )
    return [
        (i * width, (i + 1) * width if i + 1 < n_folds else n_test)
        for i in range(n_folds)
    ]


def walk_forward_hold(
    cell: GridCell,
    returns: NDArray[np.float64],
    timestamps: NDArray[np.datetime64],
    candidate_holds: tuple[int, ...] = (1, 2, 3, 5, 10),
    n_train_bars: int = 1000,
    n_folds: int = 4,
    cost_bps: float = COST_BPS_ROUND_TRIP,
) -> dict:
    """Expanding-window hold selection over the region after ``n_train_bars``.

    For fold k the training window is ``[: n_train_bars + fold_start]``
    (expanding) and the evaluation window is the fold itself. The hold with
    the best train Sharpe is applied to the fold untouched. Returns fold rows
    plus a summary with per-hold selection counts and mean fold Sharpes for
    the walk-forward choice vs every fixed hold (same folds, no peeking).
    """
    n = len(returns)
    if n_train_bars >= n - 2 * candidate_holds[-1]:
        raise ValueError(f"not enough data: n={n}, n_train_bars={n_train_bars}")
    bounds = fold_boundaries(n - n_train_bars, n_folds)

    # Pre-compute per-hold Sharpe once per contiguous window via run_mgmt_cell;
    # cache keyed by (hold, end_idx) since expanding trains share prefixes.
    def sharpe_for(hold: int, start: int, end: int) -> float:
        r = run_mgmt_cell(cell, returns[start:end], timestamps[start:end],
                          hold=hold, sizing="flat", cost_bps=cost_bps)
        return r["sharpe"]

    folds = []
    wf_sharpes: dict[int, float] = {}
    selection_counts: dict[int, int] = {h: 0 for h in candidate_holds}
    for k, (fs, fe) in enumerate(bounds):
        tr_start = max(0, n_train_bars - 500)
        tr_end = n_train_bars + fs
        train_scores = {
            h: sharpe_for(h, tr_start, tr_end) for h in candidate_holds
        }
        best_hold = max(train_scores, key=lambda h: train_scores[h])
        selection_counts[best_hold] += 1
        row = {
            "fold": k,
            "train_end": str(timestamps[tr_end - 1]),
            "test_start": str(timestamps[n_train_bars + fs]),
            "selected_hold": best_hold,
            "train_best_sharpe": train_scores[best_hold],
            "fold_sharpe": sharpe_for(best_hold, n_train_bars + fs, n_train_bars + fe),
        }
        for h in candidate_holds:
            key = f"fixed_hold{h}_sharpe"
            row[key] = sharpe_for(h, n_train_bars + fs, n_train_bars + fe)
            wf_sharpes.setdefault(key, []).append(row[key])
        wf_sharpes.setdefault("wf", []).append(row["fold_sharpe"])
        folds.append(row)

    summary = {
        "ticker": cell.ticker,
        "rule": cell.rule,
        "params": dict(cell.params),
        "interval": cell.interval,
        "candidate_holds": list(candidate_holds),
        "n_folds": n_folds,
        "selection_counts": selection_counts,
        "mean_wf_sharpe": float(np.mean(wf_sharpes["wf"])),
        "mean_fixed_sharpe": {
            h: float(np.mean(wf_sharpes[f"fixed_hold{h}_sharpe"]))
            for h in candidate_holds
        },
    }
    return {"folds": folds, "summary": summary}
