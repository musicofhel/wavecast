#!/usr/bin/env python3
"""Run D1 forward test: fetch latest 2026 bars, predict, log to JSONL.

Can be called repeatedly (cron or manual) to accumulate predictions.

Usage:
    python scripts/forward_test_d1.py [OPTIONS]

    --model-path PATH   Model directory (default: ~/.wavecast/models/d1_augmented_v1)
    --tickers T1,T2     Comma-separated tickers (default: all 20)
    --test-name NAME    Forward test name (default: d1_forward_v1)
    --report            Print summary report after prediction cycle
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from wavecast.core.universe import DEFAULT_UNIVERSE
from wavecast.forward.config import ForwardTestConfig
from wavecast.forward.runner import ForwardTestRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = str(Path.home() / ".wavecast" / "models" / "d1_augmented_v1")
DEFAULT_TICKERS = [a.ticker for a in DEFAULT_UNIVERSE.assets]


def main() -> None:
    model_path = DEFAULT_MODEL
    tickers = DEFAULT_TICKERS
    test_name = "d1_forward_v1"
    show_report = False

    for arg in sys.argv[1:]:
        if arg.startswith("--model-path="):
            model_path = arg.split("=", 1)[1]
        elif arg.startswith("--tickers="):
            tickers = arg.split("=", 1)[1].split(",")
        elif arg.startswith("--test-name="):
            test_name = arg.split("=", 1)[1]
        elif arg == "--report":
            show_report = True

    # Verify model exists
    model_dir = Path(model_path)
    if not (model_dir / "config.json").exists():
        logger.error("Model not found at %s — run scripts/train_d1_model.py first", model_path)
        sys.exit(1)

    config = ForwardTestConfig(
        test_name=test_name,
        model_path=model_path,
        vocab_path="",  # not needed for continuous models
        tickers=tickers,
        intervals=["1h"],
        horizons=[1],
        lookback_bars=300,
        dwt_levels=[1, 2, 5],
    )

    runner = ForwardTestRunner(config)
    logger.info("Starting D1 forward test: %d tickers, test=%s", len(tickers), test_name)
    summary = runner.run_once()

    print(f"\nForward test cycle complete: {test_name}")
    print(f"  Total predictions: {summary.total_predictions}")
    print(f"  Resolved: {summary.resolved_predictions}")
    if summary.total_predictions > 0 and summary.resolved_predictions > 0:
        print(f"  Accuracy: {summary.accuracy:.1%}")
        print(f"  Dir accuracy: {summary.directional_accuracy:.1%}")
        print(f"  Cumulative PnL: {summary.cumulative_pnl:+.4f}")

    if show_report:
        print("\n--- Per-ticker breakdown ---")
        for ticker_name, ticker_summary in sorted(summary.per_ticker.items()):
            n = ticker_summary.get("n_predictions", 0)
            n_res = ticker_summary.get("n_resolved", 0)
            acc = ticker_summary.get("directional_accuracy", 0)
            pnl = ticker_summary.get("cumulative_pnl", 0)
            if n > 0:
                print(f"  {ticker_name:6s}  preds={n:3d}  resolved={n_res:3d}  dir_acc={acc:.1%}  pnl={pnl:+.4f}")

    log_dir = Path.home() / ".wavecast" / "forward_tests" / test_name
    print(f"\n  Predictions logged to: {log_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
