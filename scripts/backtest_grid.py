"""B1 backtest grid harness — entry point.

Runs {ticker x interval x trading rule} cells over cached OHLCV parquets
(~/.wavecast/cache/{TICKER}_{interval}_ohlcv.parquet) through the signals
framework at 7bps round-trip costs, and appends results to
research/results.jsonl.

Rule-based cells (persistence / mean_reversion / momentum) need no model.
Model cells require predictions passed via --predictions (a JSON file mapping
"TICKER|interval" -> {"directions": [...], "confidences": [...]}).

Usage:
    python scripts/backtest_grid.py --tickers SPY,AAPL --intervals 1h,1d \
        --rules persistence,mean_reversion,momentum --out research/results.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wavecast.signals.grid import GridCell, append_results, load_results, run_grid

CACHE_DIR = Path.home() / ".wavecast" / "cache"
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "JPM", "GS", "BAC",
    "XOM", "CVX", "COP",
    "JNJ", "UNH", "PFE",
    "SPY", "QQQ",
    "GLD", "SLV", "USO", "UNG",
]


def load_series(ticker: str, interval: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load close-to-close log returns from the OHLCV cache; None if absent."""
    path = CACHE_DIR / f"{ticker}_{interval}_ohlcv.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    closes = df["close"].to_numpy(dtype=np.float64)
    ts = pd.to_datetime(df["timestamp"]).to_numpy(dtype="datetime64[ns]")
    returns = np.diff(np.log(closes))
    # align: return[i] is realized over bar i -> i+1; signal at bar i predicts it,
    # so drop the first timestamp to keep arrays equal length.
    return ts[1:], returns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=",".join(DEFAULT_UNIVERSE))
    parser.add_argument("--intervals", default="1h")
    parser.add_argument(
        "--rules", default="persistence,mean_reversion",
        help="comma list of rules; momentum takes optional :<lookback>",
    )
    parser.add_argument("--cost-bps", type=float, default=7.0)
    parser.add_argument("--predictions", default=None,
                        help="JSON file of model predictions for 'model' rule cells")
    parser.add_argument("--out", default="research/results.jsonl")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]

    data = {}
    for ticker in tickers:
        for interval in intervals:
            series = load_series(ticker, interval)
            if series is None:
                print(f"skip {ticker} {interval}: no cache parquet")
                continue
            data[(ticker, interval)] = series

    cells = []
    for rule_spec in args.rules.split(","):
        rule_spec = rule_spec.strip()
        params: dict = {}
        rule = rule_spec
        if ":" in rule_spec:
            rule, _, value = rule_spec.partition(":")
            params["lookback"] = int(value)
        for ticker in tickers:
            for interval in intervals:
                if (ticker, interval) in data:
                    cells.append(GridCell(ticker=ticker, interval=interval, rule=rule, params=params))

    predictions = None
    if args.predictions:
        raw = json.loads(Path(args.predictions).read_text())
        predictions = {
            tuple(k.split("|")): (
                np.asarray(v["directions"], dtype=np.float64),
                np.asarray(v.get("confidences", [1.0] * len(v["directions"])), dtype=np.float64),
            )
            for k, v in raw.items()
        }

    rows = run_grid(cells, data, predictions=predictions, cost_bps=args.cost_bps)

    prior = {(r["cell_key"]) for r in load_results(args.out)}
    fresh = [r for r in rows if r["cell_key"] not in prior]
    append_results(fresh, args.out)
    done = len(rows)
    print(f"ran {done} cells ({len(rows) - len(fresh)} already in ledger), "
          f"wrote {len(fresh)} new rows -> {args.out}")

    best = sorted(rows, key=lambda r: r["sharpe"], reverse=True)[:5]
    for r in best:
        base = r.get("baseline_persistence_sharpe")
        base_s = f"{base:+.2f}" if base is not None else "  n/a"
        print(f"{r['ticker']:5s} {r['interval']} {r['rule']:15s} "
              f"sharpe {r['sharpe']:+6.2f} (persistence {base_s}) "
              f"flat {r['flat_rate']:.2%} trades {r['num_trades']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
