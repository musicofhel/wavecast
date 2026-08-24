"""B3 timescale sweep — entry point.

Runs the rule-based configs through BOTH intervals ({1h, 1d}) x holding-period
overlays, so timescales are compared at comparable ANNUAL turnover rather than
raw per-bar flipping (B1: hourly per-bar is cost-dominated at 7bps; B4: holding
overlays cut turnover). Sizing is flat only — B4 found vol-scaling neutral.

For every cell we record ``turnover`` (per bar) as usual; annualized turnover
is derivable via signals.timescale.annual_turnover. Appends rows to
research/results.jsonl with "task": "B3".

Usage:
    python scripts/timescale_sweep.py [--intervals 1h,1d] \
        [--holds 1,2,3,5,10] [--train-end 2024-06-30] [--out research/results.jsonl]
"""

from __future__ import annotations

import argparse

import numpy as np
from scripts.backtest_grid import DEFAULT_UNIVERSE, load_series

from wavecast.signals.grid import GridCell, append_results
from wavecast.signals.stability import split_window
from wavecast.signals.timescale import annual_turnover
from wavecast.signals.trade_mgmt import run_mgmt_cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=",".join(DEFAULT_UNIVERSE))
    parser.add_argument("--rules", default="persistence,mean_reversion")
    parser.add_argument("--intervals", default="1h,1d")
    parser.add_argument("--holds", default="1,2,3,5,10")
    parser.add_argument("--train-end", default="2024-06-30")
    parser.add_argument("--cost-bps", type=float, default=7.0)
    parser.add_argument("--out", default="research/results.jsonl")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    rules = [r.strip() for r in args.rules.split(",")]
    intervals = [i.strip() for i in args.intervals.split(",")]
    holds = [int(h) for h in args.holds.split(",")]

    rows = []
    for interval in intervals:
        for ticker in tickers:
            series = load_series(ticker, interval)
            if series is None:
                print(f"skip {ticker} {interval}: no cache parquet")
                continue
            ts, rets = series
            try:
                (tr_ts, tr_rets), (te_ts, te_rets) = split_window(ts, rets, args.train_end)
            except ValueError:
                print(f"skip {ticker} {interval}: empty split window")
                continue
            for rule in rules:
                cell = GridCell(ticker=ticker, interval=interval, rule=rule, params={})
                for hold in holds:
                    tr = run_mgmt_cell(cell, tr_rets, tr_ts, hold=hold,
                                       cost_bps=args.cost_bps)
                    te = run_mgmt_cell(cell, te_rets, te_ts, hold=hold,
                                       cost_bps=args.cost_bps)
                    row = dict(te)
                    row.update({
                        "task": "B3",
                        "annual_turnover": annual_turnover(row["turnover"], interval),
                        "train_sharpe": tr["sharpe"],
                        "train_total_return": tr["total_return"],
                        "train_end": args.train_end,
                    })
                    rows.append(row)

    # aggregate per config across tickers: test Sharpe + mean annual turnover
    agg: dict[tuple, list[dict]] = {}
    for r in rows:
        agg.setdefault((r["interval"], r["rule"], r["hold"]), []).append(r)
    print(f"{'ivl':4s} {'rule':16s} {'hold':>4s} {'test_mean':>9s} {'pos':>4s} "
          f"{'ann_turn':>9s} n")
    for key in sorted(agg):
        v = np.asarray([r["sharpe"] for r in agg[key]])
        turn = np.mean([r["annual_turnover"] for r in agg[key]])
        print(f"{key[0]:4s} {key[1]:16s} {key[2]:4d} {v.mean():+9.3f} "
              f"{int((v > 0).sum()):4d} {turn:9.1f} {len(v)}")

    append_results(rows, args.out)
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
