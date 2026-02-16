"""F4: Optuna HPO for SAX parameters.

Search space:
  n_segments: [64, 128, 256, 512]
  word_length: [3, 4, 5, 6]
  word_stride: [1, 2]

Fixed: alphabet=7, context=16, dwt_levels=[1,2,5]
Objective: maximize directional accuracy on walk-forward test set
Tickers: [AAPL, SPY, GLD, JPM, XOM] (one per sector + broad ETF)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wavecast.experiments.hpo import (
    run_sax_hpo,
    save_study_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path.home() / ".wavecast" / "experiments" / "F4_sax_hpo.json"
N_TRIALS = 30


def main() -> None:
    logger.info("=" * 70)
    logger.info("F4: Optuna HPO for SAX Parameters")
    logger.info("Trials: %d", N_TRIALS)
    logger.info("Search: n_segments=[64,128,256,512], word_length=[3,4,5,6], word_stride=[1,2]")
    logger.info("Fixed: alphabet=7, context=16, levels=[1,2,5]")
    logger.info("Tickers: AAPL, SPY, GLD, JPM, XOM")
    logger.info("=" * 70)

    study = run_sax_hpo(n_trials=N_TRIALS)

    # Save results
    save_study_results(study, OUTPUT_PATH)
    logger.info("Results saved to %s", OUTPUT_PATH)

    # Print summary
    print("\n" + "=" * 70)
    print("F4 RESULTS: SAX Parameter HPO")
    print("=" * 70)
    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best directional accuracy: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.params}")

    print(f"\nOptimal n_segments: {study.best_trial.params['n_segments']}")
    print(f"Optimal word_length: {study.best_trial.params['word_length']}")
    print(f"Optimal word_stride: {study.best_trial.params['word_stride']}")

    # Top 5 trials
    print("\nTop 5 trials:")
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=True,
    )
    for i, t in enumerate(sorted_trials[:5]):
        print(
            f"  {i + 1}. Trial #{t.number}: "
            f"dir_acc={t.value:.4f} | "
            f"n_seg={t.params['n_segments']}, "
            f"wl={t.params['word_length']}, "
            f"ws={t.params['word_stride']}"
        )

    print(f"\nTotal trials: {len(study.trials)}")
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    print(f"Completed: {len(completed)}")
    if completed:
        values = [t.value for t in completed if t.value is not None]
        print(f"Mean directional accuracy: {sum(values) / len(values):.4f}")
        print(f"Std directional accuracy: {(sum((v - sum(values)/len(values))**2 for v in values) / len(values))**0.5:.4f}")


if __name__ == "__main__":
    main()
