"""C3: Cross-sector transfer matrix.

Research Q3: Does multi-sector training help?

Uses optimal settings from C1 (alphabet_size=7) and C2 (levels=[1,2,5]).
Runs:
  - 1 all-asset baseline
  - 5 per-sector isolation (train+test within sector)
  - 5 leave-one-sector-out
  - 20 single-asset
Total: 31 experiments.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

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

CACHE_DIR = Path.home() / ".wavecast" / "cache"
OUTPUT_PATH = Path.home() / ".wavecast" / "experiments" / "C3_transfer.json"

# Optimal from C1 and C2
OPTIMAL_ALPHABET = 7
OPTIMAL_LEVELS = [1, 2, 5]

# Sector definitions
SECTORS = {
    "tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    "finance": ["JPM", "GS", "BAC"],
    "energy": ["XOM", "CVX", "COP"],
    "healthcare": ["JNJ", "UNH", "PFE"],
    "broad_etf": ["SPY", "QQQ"],
    "commodity_etf": ["GLD", "SLV", "USO", "UNG"],
}

ALL_TICKERS = PHASE3_UNIVERSE.tickers


def _make_config(name: str, tickers: list[str]) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        tickers=tickers,
        alphabet_size=OPTIMAL_ALPHABET,
        n_segments=256,
        word_length=4,
        context_length=16,
        interval="1h",
        dwt_levels=OPTIMAL_LEVELS,
    )


def main() -> None:
    runner = ExperimentRunner(cache_dir=CACHE_DIR)

    configs: list[ExperimentConfig] = []

    # 1 all-asset baseline
    configs.append(_make_config("C3_all_assets", ALL_TICKERS))

    # 5 per-sector isolation
    for sector_name, sector_tickers in SECTORS.items():
        configs.append(_make_config(f"C3_sector_{sector_name}", sector_tickers))

    # 5 leave-one-sector-out
    for sector_name, sector_tickers in SECTORS.items():
        remaining = [t for t in ALL_TICKERS if t not in sector_tickers]
        configs.append(_make_config(f"C3_leave_out_{sector_name}", remaining))

    # 20 single-asset
    for ticker in ALL_TICKERS:
        configs.append(_make_config(f"C3_single_{ticker}", [ticker]))

    logger.info("=" * 70)
    logger.info("C3: Cross-Sector Transfer Matrix")
    logger.info("Optimal alphabet_size=%d, levels=%s (from C1+C2)", OPTIMAL_ALPHABET, OPTIMAL_LEVELS)
    logger.info("Experiments: %d total", len(configs))
    logger.info("=" * 70)

    results = runner.run_sweep(configs)

    # Save results
    save_results(results, OUTPUT_PATH)
    logger.info("Results saved to %s", OUTPUT_PATH)

    # Print comparison table
    df = compare_results(results, metric="token_accuracy")
    print("\n" + "=" * 100)
    print("C3 RESULTS: Cross-Sector Transfer (sorted by token accuracy)")
    print("=" * 100)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Named result lookup
    result_map = {r.config.name: r for r in results}

    # All-asset baseline
    baseline = result_map.get("C3_all_assets")
    if baseline:
        print(f"\nAll-asset baseline: token_acc={baseline.token_accuracy:.4f}, "
              f"dir_acc={baseline.directional_accuracy:.4f}, "
              f"n_test={baseline.n_test_samples}")

    # Per-sector isolation vs all-asset
    print("\nSector isolation vs all-asset baseline:")
    print(f"  {'Sector':<16} {'Isolated':>10} {'All-Asset':>10} {'Delta':>10} {'N_test':>8}")
    for sector_name in SECTORS:
        r = result_map.get(f"C3_sector_{sector_name}")
        if r and baseline:
            # Get all-asset accuracy for this sector's tickers from per_asset_accuracy
            sector_tickers = SECTORS[sector_name]
            all_asset_accs = [baseline.per_asset_accuracy.get(t, 0) for t in sector_tickers]
            all_asset_mean = np.mean(all_asset_accs) if all_asset_accs else 0
            delta = r.token_accuracy - all_asset_mean
            print(f"  {sector_name:<16} {r.token_accuracy:>10.4f} {all_asset_mean:>10.4f} {delta:>+10.4f} {r.n_test_samples:>8}")

    # Leave-one-sector-out
    print("\nLeave-one-sector-out vs all-asset baseline:")
    print(f"  {'Left Out':<16} {'Token Acc':>10} {'All-Asset':>10} {'Delta':>10}")
    if baseline:
        for sector_name in SECTORS:
            r = result_map.get(f"C3_leave_out_{sector_name}")
            if r:
                delta = r.token_accuracy - baseline.token_accuracy
                print(f"  {sector_name:<16} {r.token_accuracy:>10.4f} {baseline.token_accuracy:>10.4f} {delta:>+10.4f}")

    # Single-asset results (median +/- IQR)
    single_accs = []
    for ticker in ALL_TICKERS:
        r = result_map.get(f"C3_single_{ticker}")
        if r:
            single_accs.append((ticker, r.token_accuracy, r.n_test_samples))

    if single_accs:
        accs = [a[1] for a in single_accs]
        median = np.median(accs)
        q25, q75 = np.percentile(accs, [25, 75])
        print(f"\nSingle-asset results (n={len(single_accs)}):")
        print(f"  Median token accuracy: {median:.4f} (IQR: {q25:.4f}-{q75:.4f})")
        print(f"  Warning: Small sample sizes ({single_accs[0][2]} samples) likely overfit")
        print(f"\n  {'Ticker':<8} {'Token Acc':>10} {'N_test':>8}")
        for ticker, acc, n_test in sorted(single_accs, key=lambda x: -x[1]):
            print(f"  {ticker:<8} {acc:>10.4f} {n_test:>8}")

    # Transfer matrix: does training on more data help?
    if baseline:
        print("\n--- TRANSFER ANALYSIS ---")
        helps_count = 0
        hurts_count = 0
        for sector_name in SECTORS:
            r = result_map.get(f"C3_sector_{sector_name}")
            if r:
                sector_tickers = SECTORS[sector_name]
                all_asset_accs = [baseline.per_asset_accuracy.get(t, 0) for t in sector_tickers]
                all_asset_mean = np.mean(all_asset_accs) if all_asset_accs else 0
                if r.token_accuracy > all_asset_mean:
                    helps_count += 1
                    print(f"  {sector_name}: isolation BETTER than multi-asset ({r.token_accuracy:.4f} vs {all_asset_mean:.4f})")
                else:
                    hurts_count += 1
                    print(f"  {sector_name}: multi-asset BETTER than isolation ({all_asset_mean:.4f} vs {r.token_accuracy:.4f})")

        print(f"\n  Multi-sector training helps: {hurts_count}/{hurts_count + helps_count} sectors")
        print(f"  Sector isolation better: {helps_count}/{hurts_count + helps_count} sectors")


if __name__ == "__main__":
    main()
