"""C2: DWT level contribution analysis.

Research Q2: Which DWT levels are most predictable?

Uses optimal alphabet_size=7 from C1. Runs:
  - 5 single-level models (level 1 through 5)
  - 1 all-levels-combined model (levels [1,2,3,4,5])
  - 5 ablation models (drop one level each)
Total: 11 experiments.
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
OUTPUT_PATH = Path.home() / ".wavecast" / "experiments" / "C2_levels.json"

# Optimal from C1
OPTIMAL_ALPHABET = 7
ALL_LEVELS = [1, 2, 3, 4, 5]


def main() -> None:
    runner = ExperimentRunner(cache_dir=CACHE_DIR)

    configs: list[ExperimentConfig] = []

    # 5 single-level models
    for lvl in ALL_LEVELS:
        configs.append(
            ExperimentConfig(
                name=f"C2_level{lvl}_only",
                tickers=TICKERS,
                alphabet_size=OPTIMAL_ALPHABET,
                n_segments=256,
                word_length=4,
                context_length=16,
                interval="1h",
                dwt_levels=[lvl],
            )
        )

    # 1 all-levels-combined
    configs.append(
        ExperimentConfig(
            name="C2_all_levels",
            tickers=TICKERS,
            alphabet_size=OPTIMAL_ALPHABET,
            n_segments=256,
            word_length=4,
            context_length=16,
            interval="1h",
            dwt_levels=ALL_LEVELS,
        )
    )

    # 5 ablation models (drop one level each)
    for drop_lvl in ALL_LEVELS:
        remaining = [l for l in ALL_LEVELS if l != drop_lvl]
        configs.append(
            ExperimentConfig(
                name=f"C2_drop_level{drop_lvl}",
                tickers=TICKERS,
                alphabet_size=OPTIMAL_ALPHABET,
                n_segments=256,
                word_length=4,
                context_length=16,
                interval="1h",
                dwt_levels=remaining,
            )
        )

    logger.info("=" * 70)
    logger.info("C2: DWT Level Contribution Analysis")
    logger.info("Optimal alphabet_size=%d (from C1)", OPTIMAL_ALPHABET)
    logger.info("Experiments: %d total", len(configs))
    logger.info("=" * 70)

    results = runner.run_sweep(configs)

    # Save results
    save_results(results, OUTPUT_PATH)
    logger.info("Results saved to %s", OUTPUT_PATH)

    # Print comparison table
    df = compare_results(results, metric="token_accuracy")
    print("\n" + "=" * 90)
    print("C2 RESULTS: DWT Level Contribution (sorted by token accuracy)")
    print("=" * 90)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Single-level results
    single_results = {r.config.name: r for r in results if r.config.name.startswith("C2_level")}
    print("\nPer-level token accuracy (single-level models):")
    for lvl in ALL_LEVELS:
        r = single_results.get(f"C2_level{lvl}_only")
        if r:
            print(f"  Level {lvl}: token_acc={r.token_accuracy:.4f}, dir_acc={r.directional_accuracy:.4f}, "
                  f"samples={r.n_test_samples}, vocab={r.vocab_size}")

    # All-levels combined
    all_levels_result = next((r for r in results if r.config.name == "C2_all_levels"), None)
    if all_levels_result:
        print(f"\nAll levels combined: token_acc={all_levels_result.token_accuracy:.4f}, "
              f"dir_acc={all_levels_result.directional_accuracy:.4f}")
        if all_levels_result.per_level_accuracy:
            print("  Per-level breakdown within combined model:")
            for lvl in sorted(all_levels_result.per_level_accuracy.keys()):
                print(f"    Level {lvl}: {all_levels_result.per_level_accuracy[lvl]:.4f}")

    # Ablation impact
    print("\nAblation impact (drop one level, measure accuracy change):")
    if all_levels_result:
        base_acc = all_levels_result.token_accuracy
        for drop_lvl in ALL_LEVELS:
            r = next((r for r in results if r.config.name == f"C2_drop_level{drop_lvl}"), None)
            if r:
                delta = r.token_accuracy - base_acc
                pct = delta / base_acc * 100
                print(f"  Drop level {drop_lvl}: token_acc={r.token_accuracy:.4f} "
                      f"(delta={delta:+.4f}, {pct:+.1f}%)")

    # Recommendation
    if all_levels_result:
        base_acc = all_levels_result.token_accuracy
        exclude_levels = []
        for drop_lvl in ALL_LEVELS:
            r = next((r for r in results if r.config.name == f"C2_drop_level{drop_lvl}"), None)
            if r:
                delta = r.token_accuracy - base_acc
                if delta >= 0 or abs(delta / base_acc * 100) < 1.0:
                    exclude_levels.append(drop_lvl)

        if exclude_levels:
            recommended = [l for l in ALL_LEVELS if l not in exclude_levels]
            print(f"\nRecommendation: Exclude levels {exclude_levels} (<1% ablation improvement)")
            print(f"  Use levels: {recommended}")
        else:
            print(f"\nRecommendation: All levels contribute >1%. Use all levels: {ALL_LEVELS}")


if __name__ == "__main__":
    main()
