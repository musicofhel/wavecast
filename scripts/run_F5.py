"""F5: Optuna HPO for model architecture.

Search space:
  embed_dim: [32, 64, 128]
  num_heads: [2, 4, 8]
  num_layers: [2, 3, 4, 6]
  dropout: [0.05, 0.1, 0.15, 0.2]

Constraint: embed_dim must be divisible by num_heads
Fixed: SAX params from F4 results (or Phase 3 defaults)
Objective: maximize directional accuracy on walk-forward test set
Tickers: [AAPL, SPY, GLD, JPM, XOM]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wavecast.experiments.hpo import (
    run_architecture_hpo,
    save_study_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)

F4_PATH = Path.home() / ".wavecast" / "experiments" / "F4_sax_hpo.json"
OUTPUT_PATH = Path.home() / ".wavecast" / "experiments" / "F5_arch_hpo.json"
N_TRIALS = 25


def load_f4_best_params() -> dict:
    """Load best SAX params from F4, fall back to Phase 3 defaults."""
    if F4_PATH.exists():
        data = json.loads(F4_PATH.read_text())
        params = data.get("best_trial", {}).get("params", {})
        if params:
            logger.info("Loaded F4 best params: %s", params)
            return {
                "n_segments": params.get("n_segments", 256),
                "word_length": params.get("word_length", 4),
                "word_stride": params.get("word_stride", 1),
            }
    logger.info("F4 results not found, using Phase 3 defaults")
    return {"n_segments": 256, "word_length": 4, "word_stride": 1}


def main() -> None:
    sax_params = load_f4_best_params()

    logger.info("=" * 70)
    logger.info("F5: Optuna HPO for Model Architecture")
    logger.info("Trials: %d", N_TRIALS)
    logger.info("Search: embed_dim=[32,64,128], num_heads=[2,4,8], num_layers=[2,3,4,6], dropout=[0.05-0.2]")
    logger.info("SAX params (from F4): %s", sax_params)
    logger.info("Tickers: AAPL, SPY, GLD, JPM, XOM")
    logger.info("=" * 70)

    study = run_architecture_hpo(
        n_trials=N_TRIALS,
        n_segments=sax_params["n_segments"],
        word_length=sax_params["word_length"],
        word_stride=sax_params["word_stride"],
    )

    # Save results
    save_study_results(study, OUTPUT_PATH)
    logger.info("Results saved to %s", OUTPUT_PATH)

    # Print summary
    print("\n" + "=" * 70)
    print("F5 RESULTS: Architecture HPO")
    print("=" * 70)
    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best directional accuracy: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.params}")

    print(f"\nOptimal embed_dim: {study.best_trial.params.get('embed_dim')}")
    print(f"Optimal num_heads: {study.best_trial.params.get('num_heads')}")
    print(f"Optimal num_layers: {study.best_trial.params.get('num_layers')}")
    print(f"Optimal dropout: {study.best_trial.params.get('dropout')}")

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
            f"ed={t.params.get('embed_dim')}, "
            f"nh={t.params.get('num_heads')}, "
            f"nl={t.params.get('num_layers')}, "
            f"do={t.params.get('dropout')}"
        )

    print(f"\nTotal trials: {len(study.trials)}")
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned = [t for t in study.trials if t.state.name == "PRUNED"]
    print(f"Completed: {len(completed)}, Pruned (invalid combos): {len(pruned)}")
    if completed:
        values = [t.value for t in completed if t.value is not None]
        print(f"Mean directional accuracy: {sum(values) / len(values):.4f}")


if __name__ == "__main__":
    main()
