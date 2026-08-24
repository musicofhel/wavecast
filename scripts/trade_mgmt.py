"""B4 trade-management sweep — entry point.

Runs the daily persistence / mean_reversion configs (the only positive-OOS
members per the B2 report) through holding-period and sizing variants:
{rule} x hold in {1,2,3,5,10} x sizing in {flat, vol_scaled} on train and
test windows (train_end 2024-06-30, same split as B2), at 7bps round-trip.

Appends rows to research/results.jsonl with "task": "B4".

Usage:
    python scripts/trade_mgmt.py [--tickers ...] [--holds 1,2,3,5,10] \
        [--train-end 2024-06-30] [--out research/results.jsonl]
"""

from __future__ import annotations

import argparse

import numpy as np
from scripts.backtest_grid import DEFAULT_UNIVERSE, load_series

from wavecast.signals.grid import GridCell, append_results
from wavecast.signals.stability import split_window
from wavecast.signals.trade_mgmt import run_mgmt_cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=",".join(DEFAULT_UNIVERSE))
    parser.add_argument("--rules", default="persistence,mean_reversion")
    parser.add_argument("--holds", default="1,2,3,5,10")
    parser.add_argument("--sizings", default="flat,vol_scaled")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--train-end", default="2024-06-30")
    parser.add_argument("--cost-bps", type=float, default=7.0)
    parser.add_argument("--out", default="research/results.jsonl")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    rules = [r.strip() for r in args.rules.split(",")]
    holds = [int(h) for h in args.holds.split(",")]
    sizings = [s.strip() for s in args.sizings.split(",")]

    data = {}
    for ticker in tickers:
        series = load_series(ticker, args.interval)
        if series is None:
            print(f"skip {ticker}: no cache parquet")
            continue
        data[ticker] = series

    rows = []
    for ticker, (ts, rets) in data.items():
        try:
            (tr_ts, tr_rets), (te_ts, te_rets) = split_window(ts, rets, args.train_end)
        except ValueError:
            print(f"skip {ticker}: empty split window")
            continue
        for rule in rules:
            params = {"lookback": 3} if rule == "momentum" else {}
            cell = GridCell(ticker=ticker, interval=args.interval, rule=rule, params=params)
            for hold in holds:
                for sizing in sizings:
                    tr = run_mgmt_cell(cell, tr_rets, tr_ts, hold=hold, sizing=sizing,
                                       cost_bps=args.cost_bps)
                    te = run_mgmt_cell(cell, te_rets, te_ts, hold=hold, sizing=sizing,
                                       cost_bps=args.cost_bps)
                    row = dict(te)
                    row.update({
                        "task": "B4",
                        "train_sharpe": tr["sharpe"],
                        "train_total_return": tr["total_return"],
                        "train_end": args.train_end,
                    })
                    rows.append(row)

    # aggregate per config across tickers
    agg: dict[tuple, list[float]] = {}
    for r in rows:
        agg.setdefault((r["rule"], r["hold"], r["sizing"]), []).append(r["sharpe"])
    print(f"{'rule':16s} {'hold':>4s} {'sizing':10s} {'test_mean':>9s} {'pos':>4s} n")
    for key in sorted(agg):
        v = np.asarray(agg[key])
        print(f"{key[0]:16s} {key[1]:4d} {key[2]:10s} {v.mean():+9.3f} "
              f"{int((v > 0).sum()):4d} {len(v)}")

    append_results(rows, args.out)
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
