"""Fetch Phase 3 universe data (20 assets, hourly + daily)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from wavecast.core.universe import PHASE3_UNIVERSE
from wavecast.data.sources import fetch_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

FALLBACKS = {
    "GLD": "IAU",
    "SLV": "SIVR",
    "USO": "USO",
    "UNG": "UNG",
}

MIN_HOURLY_BARS = 3000


def main() -> None:
    tickers = PHASE3_UNIVERSE.tickers
    logger.info("Fetching %d assets: %s", len(tickers), tickers)
    logger.info("Intervals: 1h, 1d | Date range: 2021-02-01 to 2025-12-31")

    results = fetch_universe(
        tickers=tickers,
        start="2021-02-01",
        end="2025-12-31",
        intervals=["1h", "1d"],
        rate_limit_pause=12.5,
        fallbacks=FALLBACKS,
    )

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Ticker':<10} {'Hourly Bars':>12} {'Daily Bars':>12} {'Status':>10}")
    print("-" * 70)

    all_ok = True
    for ticker in tickers:
        hourly_bars = len(results[ticker].get("1h", []))
        daily_bars = len(results[ticker].get("1d", []))
        status = "OK" if hourly_bars >= MIN_HOURLY_BARS else "LOW"
        if hourly_bars < MIN_HOURLY_BARS:
            all_ok = False
        print(f"{ticker:<10} {hourly_bars:>12,} {daily_bars:>12,} {status:>10}")

    print("=" * 70)
    if all_ok:
        print("All assets meet minimum bar threshold.")
    else:
        print(f"WARNING: Some assets have < {MIN_HOURLY_BARS} hourly bars.")

    cache_dir = Path.home() / ".wavecast" / "cache"
    total_size = sum(f.stat().st_size for f in cache_dir.glob("*.parquet"))
    print(f"\nCache directory: {cache_dir}")
    print(f"Total cache size: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
