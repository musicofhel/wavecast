#!/usr/bin/env python3
"""A5 end-to-end forward-test cycle proof (scratch only).

Phase 1: ForwardTestRunner.run_once() on the latest available bars —
         fetch → predict → log PENDING rows into loop_scratch.
Phase 2: CatchupRunner.run_for_session() on a past session — historical
         bars resolve immediately, exercising resolution with A2 fixes.

Never touches d1_forward_v1: the log dir is forced to loop_scratch.

Usage:
    python scripts/a5_e2e_cycle.py --session 2026-08-21
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from wavecast.core.universe import DEFAULT_UNIVERSE
from wavecast.forward.catchup import CatchupRunner
from wavecast.forward.config import ForwardTestConfig
from wavecast.forward.runner import ForwardTestRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_MODEL = str(Path.home() / ".wavecast" / "models" / "d1_augmented_v1")
DEFAULT_TICKERS = [a.ticker for a in DEFAULT_UNIVERSE.assets]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--vocab-path", default="")
    p.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    p.add_argument(
        "--log-dir",
        type=Path,
        default=Path.home() / ".wavecast" / "forward_tests" / "loop_scratch",
    )
    p.add_argument("--test-name", default="a5_e2e_cycle")
    p.add_argument("--session", required=True, help="past session for resolve phase (YYYY-MM-DD)")
    args = p.parse_args()

    if "d1_forward_v1" in str(args.log_dir):
        raise SystemExit("refusing to write the production ledger; use a scratch dir")

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    def config(test_name: str) -> ForwardTestConfig:
        return ForwardTestConfig(
            test_name=test_name,
            model_path=args.model_path,
            vocab_path=args.vocab_path,
            tickers=tickers,
            intervals=["1h"],
            horizons=[1],
            lookback_bars=300,
            dwt_levels=[1, 2, 5],
            log_dir=args.log_dir,
        )

    # Phase 1: live cycle — pending predictions from latest bars
    t0 = time.time()
    runner = ForwardTestRunner(config(args.test_name), pacing_seconds=0.5)
    summary = runner.run_once()
    print(f"\n[phase 1 run_once] {summary.total_predictions} predictions "
          f"in {time.time() - t0:.1f}s -> {args.log_dir / args.test_name}")

    # Phase 2: past-session catch-up — resolves against historical bars
    import datetime

    from wavecast.data.sources import fetch_massive_ohlcv
    from wavecast.forward.tracker import ForwardTestTracker

    t0 = time.time()
    resolve_name = f"{args.test_name}_resolve"
    catcher = CatchupRunner(config(resolve_name), pacing_seconds=0.5)
    n = catcher.run_for_session(args.session)
    print(f"[phase 2 catch-up {args.session}] {n} predictions in {time.time() - t0:.1f}s")

    # Phase 3: resolve the catch-up ledger against historical bars (A2 logic)
    tracker = ForwardTestTracker(
        log_dir=args.log_dir,
        test_name=resolve_name,
    )
    session_date = datetime.date.fromisoformat(args.session)
    start = (session_date - datetime.timedelta(days=70)).isoformat()
    end_exclusive = (session_date + datetime.timedelta(days=3)).isoformat()
    total_resolved = 0
    for ticker in tickers:
        try:
            prices = fetch_massive_ohlcv(
                ticker=ticker, start=start, end=end_exclusive, interval="1h"
            )
        except Exception as e:  # noqa: BLE001 — one ticker's API failure must not kill the cycle
            logging.warning("fetch failed for %s, skipping resolve: %s", ticker, e)
            continue
        if prices is None or prices.empty:
            logging.warning("no bars for %s; skipping resolve", ticker)
            continue
        total_resolved += tracker.resolve_pending(ticker, "1h", prices)
        time.sleep(0.5)
    resolved = tracker.get_resolved()
    correct = sum(1 for r in resolved if r.correct)
    print(f"[phase 3 resolve] resolved={total_resolved} "
          f"correct={correct}/{len(resolved)}")



if __name__ == "__main__":
    main()
