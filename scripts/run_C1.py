"""C1: SAX alphabet granularity sweep.

Research Q1: Is alphabet_size=7 optimal? Does it vary by sector?

Sweeps alphabet_size in {3, 5, 7, 9, 11}, holding constant:
  n_segments=256, word_length=4, context_length=16, all 20 assets, hourly bars.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wavecast.core.universe import PHASE3_UNIVERSE
from wavecast.experiments.config import ExperimentConfig
from wavecast.experiments.runner import ExperimentRunner
from wavecast.experiments.storage import save_results, compare_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)

TICKERS = PHASE3_UNIVERSE.tickers
CACHE_DIR = Path.home() / ".wavecast" / "cache"
OUTPUT_PATH = Path.home() / ".wavecast" / "experiments" / "C1_granularity.json"

ALPHABET_SIZES = [3, 5, 7, 9, 11]

# Sector groupings for per-sector analysis
SECTORS = {
    "tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    "finance": ["JPM", "GS", "BAC"],
    "energy": ["XOM", "CVX", "COP"],
    "healthcare": ["JNJ", "UNH", "PFE"],
    "broad_etf": ["SPY", "QQQ"],
    "commodity_etf": ["GLD", "SLV", "USO", "UNG"],
}


def main() -> None:
    runner = ExperimentRunner(cache_dir=CACHE_DIR)

    configs = [
        ExperimentConfig(
            name=f"C1_alpha{a}",
            tickers=TICKERS,
            alphabet_size=a,
            n_segments=256,
            word_length=4,
            context_length=16,
            interval="1h",
        )
        for a in ALPHABET_SIZES
    ]

    logger.info("=" * 70)
    logger.info("C1: SAX Alphabet Granularity Sweep")
    logger.info("Alphabet sizes: %s", ALPHABET_SIZES)
    logger.info("Tickers: %d assets", len(TICKERS))
    logger.info("=" * 70)

    results = runner.run_sweep(configs)

    # Save results
    save_results(results, OUTPUT_PATH)
    logger.info("Results saved to %s", OUTPUT_PATH)

    # Print comparison table
    df = compare_results(results, metric="directional_accuracy")
    print("\n" + "=" * 90)
    print("C1 RESULTS: Alphabet Size Sweep (sorted by directional accuracy)")
    print("=" * 90)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Print per-asset breakdown for the best config
    best = max(results, key=lambda r: r.directional_accuracy)
    print(f"\nBest config: {best.config.name} (alphabet_size={best.config.alphabet_size})")
    print(f"  Token accuracy: {best.token_accuracy:.4f} (CI: {best.token_accuracy_ci[0]:.4f}-{best.token_accuracy_ci[1]:.4f})")
    print(f"  Directional accuracy: {best.directional_accuracy:.4f} (CI: {best.directional_accuracy_ci[0]:.4f}-{best.directional_accuracy_ci[1]:.4f})")
    print(f"  Top-3 accuracy: {best.top3_accuracy:.4f}")
    print(f"  Baseline (persistence): {best.baseline_persistence:.4f}")
    print(f"  Baseline (most frequent): {best.baseline_most_frequent:.4f}")
    print(f"  Baseline (momentum): {best.baseline_momentum:.4f}")
    print(f"  Lift over persistence: {best.directional_accuracy - best.baseline_persistence:+.4f}")

    # Per-sector accuracy for best config
    if best.per_asset_accuracy:
        print(f"\nPer-sector accuracy (best config: {best.config.name}):")
        for sector_name, sector_tickers in SECTORS.items():
            accs = [best.per_asset_accuracy[t] for t in sector_tickers if t in best.per_asset_accuracy]
            if accs:
                mean_acc = sum(accs) / len(accs)
                print(f"  {sector_name:<15} {mean_acc:.4f}  ({', '.join(f'{t}={best.per_asset_accuracy.get(t, 0):.3f}' for t in sector_tickers)})")

    # Cross-alphabet sector comparison
    print("\nAlphabet x Sector directional accuracy (approximated from per-asset token accuracy):")
    header = f"{'Alpha':>6}"
    for sector_name in SECTORS:
        header += f"  {sector_name:>14}"
    print(header)
    for r in sorted(results, key=lambda x: x.config.alphabet_size):
        row = f"{r.config.alphabet_size:>6}"
        for sector_name, sector_tickers in SECTORS.items():
            accs = [r.per_asset_accuracy.get(t, 0) for t in sector_tickers]
            mean_acc = sum(accs) / len(accs) if accs else 0
            row += f"  {mean_acc:>14.4f}"
        print(row)

    # Per-level accuracy for best config
    if best.per_level_accuracy:
        print(f"\nPer-level accuracy (best config: {best.config.name}):")
        for lvl in sorted(best.per_level_accuracy.keys()):
            print(f"  Level {lvl}: {best.per_level_accuracy[lvl]:.4f}")


if __name__ == "__main__":
    main()
