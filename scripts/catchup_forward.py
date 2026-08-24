#!/usr/bin/env python3
"""Gap census + catch-up for a forward-test ledger (A3).

Default: read-only census of the production d1_forward_v1 ledger.
With --session DATE: backfill that day's predictions from historical bars
into the given scratch log dir (NEVER the production ledger).

Usage:
    python scripts/catchup_forward.py                        # census only
    python scripts/catchup_forward.py --session 2026-06-17 \
        --test-name a3_catchup_20260617 \
        --log-dir ~/.wavecast/forward_tests/loop_scratch
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from wavecast.core.universe import DEFAULT_UNIVERSE
from wavecast.forward.catchup import CatchupRunner, census_gaps
from wavecast.forward.config import ForwardTestConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_LEDGER = Path.home() / ".wavecast" / "forward_tests" / "d1_forward_v1" / "predictions.jsonl"
DEFAULT_MODEL = str(Path.home() / ".wavecast" / "models" / "d1_augmented_v1")
DEFAULT_TICKERS = [a.ticker for a in DEFAULT_UNIVERSE.assets]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--end-date", default=None, help="census end date (YYYY-MM-DD)")
    p.add_argument("--session", default=None, help="backfill this day (YYYY-MM-DD)")
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--vocab-path", default="")
    p.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    p.add_argument("--test-name", default="loop_scratch_catchup")
    p.add_argument(
        "--log-dir",
        type=Path,
        default=Path.home() / ".wavecast" / "forward_tests" / "loop_scratch",
    )
    args = p.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    census = census_gaps(args.ledger, tickers=tickers, end_date=args.end_date)
    print(census.summary())
    if census.missing_days:
        print(f"missing days: {', '.join(census.missing_days)}")

    if not args.session:
        return

    if "d1_forward_v1" in str(args.log_dir):
        raise SystemExit("refusing to write the production ledger; use a scratch dir")

    config = ForwardTestConfig(
        test_name=args.test_name,
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        tickers=tickers,
        intervals=["1h"],
        horizons=[1],
        lookback_bars=300,
        dwt_levels=[1, 2, 5],
        log_dir=args.log_dir,
    )
    runner = CatchupRunner(config, pacing_seconds=1.0)
    n = runner.run_for_session(args.session)
    out = args.log_dir / args.test_name / "predictions.jsonl"
    print(f"\ncatch-up {args.session}: {n} predictions -> {out}")


if __name__ == "__main__":
    main()
