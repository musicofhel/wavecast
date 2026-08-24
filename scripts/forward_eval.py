#!/usr/bin/env python3
"""Honest evaluation of a forward-test prediction ledger (READ-ONLY).

Economic directional accuracy (sign of actual_return), Wilson CIs, per-ticker
and per-month breakdowns, trade stats vs always-up/always-down/persistence
baselines at 7bps round-trip cost, and the flat-prediction rate.

Usage:
    python scripts/forward_eval.py [--ledger PATH] [--cost-bps 7.0]
"""

import argparse

from wavecast.evaluation.forward_ledger import evaluate, format_report, load_ledger

DEFAULT_LEDGER = "~/.wavecast/forward_tests/d1_forward_v1/predictions.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    parser.add_argument("--cost-bps", type=float, default=7.0)
    args = parser.parse_args()

    rows = load_ledger(args.ledger)
    if not rows:
        raise SystemExit(f"no resolved predictions in {args.ledger}")
    rep = evaluate(rows, cost_bps=args.cost_bps)
    print(format_report(rep))


if __name__ == "__main__":
    main()
