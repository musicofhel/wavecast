"""B5 walk-forward hold-stability sweep — entry point.

Expanding-window hold selection on daily mean_reversion across the universe:
for each of 4 OOS folds, the hold maximizing train Sharpe on everything before
the fold is applied to that fold. Reports per-hold selection counts (how often
hold=5 is re-selected) and honest walk-forward Sharpe vs fixed-hold baselines
computed on identical folds (no peeking anywhere).

Appends fold rows to research/results.jsonl with "task": "B5".

Usage:
    python scripts/hold_walkforward.py [--tickers ...] [--n-folds 4] \
        [--train-bars 1000] [--out research/results.jsonl]
"""

from __future__ import annotations

import argparse

import numpy as np
from scripts.backtest_grid import DEFAULT_UNIVERSE, load_series

from wavecast.signals.grid import GridCell, append_results
from wavecast.signals.walkforward import walk_forward_hold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=",".join(DEFAULT_UNIVERSE))
    parser.add_argument("--rule", default="mean_reversion")
    parser.add_argument("--holds", default="1,2,3,5,10")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--train-bars", type=int, default=1000)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--cost-bps", type=float, default=7.0)
    parser.add_argument("--out", default="research/results.jsonl")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    holds = tuple(int(h) for h in args.holds.split(","))
    rows = []
    summaries = []
    for ticker in tickers:
        series = load_series(ticker, args.interval)
        if series is None:
            print(f"skip {ticker}: no cache parquet")
            continue
        ts, rets = series
        cell = GridCell(ticker=ticker, interval=args.interval, rule=args.rule, params={})
        try:
            out = walk_forward_hold(
                cell, rets, ts, candidate_holds=holds,
                n_train_bars=args.train_bars, n_folds=args.n_folds,
                cost_bps=args.cost_bps,
            )
        except ValueError as exc:
            print(f"skip {ticker}: {exc}")
            continue
        summaries.append(out["summary"])
        for f in out["folds"]:
            row = dict(f)
            row.update({
                "task": "B5",
                "ticker": ticker,
                "interval": args.interval,
                "rule": args.rule,
                "candidate_holds": list(holds),
                "cost_bps_round_trip": args.cost_bps,
            })
            rows.append(row)

    # aggregate: selection counts + Sharpe comparison across all tickers/folds
    counts = {h: 0 for h in holds}
    wf: list[float] = []
    fixed: dict[int, list[float]] = {h: [] for h in holds}
    for s in summaries:
        for h, c in s["selection_counts"].items():
            counts[h] += c
    for r in rows:
        wf.append(r["fold_sharpe"])
        for h in holds:
            fixed[h].append(r[f"fixed_hold{h}_sharpe"])

    n_folds_total = len(rows)
    print(f"\nselection counts over {n_folds_total} folds "
          f"({len(summaries)} tickers):")
    for h in holds:
        pct = 100 * counts[h] / max(n_folds_total, 1)
        print(f"  hold={h:<3d} selected {counts[h]:>4d} ({pct:.0f}%)")
    print(f"\n{'strategy':>14s} {'mean fold Sharpe':>18s}")
    print(f"{'walk-forward':>14s} {np.mean(wf):>+18.3f}")
    for h in holds:
        print(f"{'fixed h=' + str(h):>14s} {np.mean(fixed[h]):>+18.3f}")

    append_results(rows, args.out)
    print(f"\nwrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
