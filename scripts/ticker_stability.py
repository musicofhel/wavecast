"""B2 per-ticker stability — entry point.

Loads cached OHLCV parquets, splits each ticker at --train-end, evaluates each
(interval, rule) config per ticker on both windows, and reports Spearman rank
correlation train→test plus top-K-selection OOS performance. Appends summary
rows to the results ledger with kind="stability".

Usage:
    python scripts/ticker_stability.py --intervals 1h,1d \
        --rules persistence,mean_reversion,momentum:3,momentum:8
"""

from __future__ import annotations

import argparse

from scripts.backtest_grid import DEFAULT_UNIVERSE, load_series

from wavecast.signals.grid import load_results
from wavecast.signals.stability import evaluate_stability


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=",".join(DEFAULT_UNIVERSE))
    parser.add_argument("--intervals", default="1h,1d")
    parser.add_argument("--rules", default="persistence,mean_reversion,momentum:3,momentum:8")
    parser.add_argument("--train-end", default="2024-06-30")
    parser.add_argument("--cost-bps", type=float, default=7.0)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]

    data = {}
    for ticker in tickers:
        for interval in intervals:
            series = load_series(ticker, interval)
            if series is None:
                continue
            data[(ticker, interval)] = series
    print(f"loaded {len(data)} series; train_end={args.train_end}")

    prior_keys = {
        (r.get("kind"), r.get("config_key"))
        for r in load_results("research/results.jsonl")
    }

    print(f"{'interval':6s} {'rule':15s} {'rho':>6s} {'n':>3s} "
          f"{'tr+':>3s} {'te+':>3s} {'topK_te':>8s} {'full_te':>8s}")
    for interval in intervals:
        for rule_spec in args.rules.split(","):
            rule_spec = rule_spec.strip()
            rule, _, value = rule_spec.partition(":")
            params = {"lookback": int(value)} if value else {}
            res = evaluate_stability(
                tickers,
                interval,
                rule,
                data,
                params=params,
                train_end=args.train_end,
                cost_bps=args.cost_bps,
                top_k=args.top_k,
            )
            if res.n_tickers == 0:
                continue
            import hashlib
            import json

            config_key = hashlib.sha256(json.dumps(
                {"interval": interval, "rule": rule, "params": params,
                 "train_end": args.train_end}, sort_keys=True).encode()
            ).hexdigest()[:12]
            if ("stability", config_key) not in prior_keys:
                from wavecast.signals.grid import append_results

                append_results([{
                    "kind": "stability",
                    "config_key": config_key,
                    "interval": interval,
                    "rule": rule,
                    "params": params,
                    "train_end": args.train_end,
                    "cost_bps_round_trip": args.cost_bps,
                    "top_k": args.top_k,
                    **{f.name: getattr(res, f.name) for f in res.__dataclass_fields__.values()
                       if f.name not in ("interval", "rule", "params")},
                }])
            print(f"{interval:6s} {rule_spec:15s} {res.spearman_sharpe:+6.2f} "
                  f"{res.n_tickers:3d} {res.n_positive_train:3d} {res.n_positive_test:3d} "
                  f"{res.top_k_test_sharpe:+8.2f} {res.full_test_sharpe:+8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
