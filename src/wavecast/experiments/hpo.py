"""Hyperparameter optimization with Optuna for SAX and architecture parameters."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import optuna

from wavecast.experiments.config import ExperimentConfig
from wavecast.experiments.runner import ExperimentRunner

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".wavecast" / "cache"
DEFAULT_EXPERIMENTS_DIR = Path.home() / ".wavecast" / "experiments"

# Fast HPO subset: one per sector + broad ETF
HPO_TICKERS = ["AAPL", "SPY", "GLD", "JPM", "XOM"]

# Phase 3 optimal defaults for fixed params
OPTIMAL_ALPHABET = 7
OPTIMAL_CONTEXT = 16
OPTIMAL_DWT_LEVELS = [1, 2, 5]


class SAXObjective:
    """Optuna objective for SAX parameter optimization.

    Maximizes directional accuracy on walk-forward test set (train 2021-2023,
    test 2024) using a 5-ticker subset for speed.
    """

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        tickers: list[str] | None = None,
    ) -> None:
        self.runner = ExperimentRunner(cache_dir=cache_dir)
        self.tickers = tickers or HPO_TICKERS

    def __call__(self, trial: optuna.Trial) -> float:
        n_segments = trial.suggest_categorical("n_segments", [64, 128, 256, 512])
        word_length = trial.suggest_categorical("word_length", [3, 4, 5, 6])
        word_stride = trial.suggest_categorical("word_stride", [1, 2])

        config = ExperimentConfig(
            name=f"sax_hpo_trial_{trial.number}",
            tickers=self.tickers,
            interval="1h",
            train_end="2023-12-31",
            test_start="2024-01-01",
            n_segments=n_segments,
            alphabet_size=OPTIMAL_ALPHABET,
            word_length=word_length,
            word_stride=word_stride,
            min_word_freq=1,
            max_vocab_size=100,
            context_length=OPTIMAL_CONTEXT,
            embed_dim=64,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
            epochs=80,
            batch_size=64,
            learning_rate=0.0005,
            patience=15,
            dwt_levels=OPTIMAL_DWT_LEVELS,
        )

        try:
            result = self.runner.run(config)
            return result.directional_accuracy
        except Exception:
            logger.exception("Trial %d failed", trial.number)
            return 0.0


class ArchitectureObjective:
    """Optuna objective for model architecture optimization.

    Maximizes directional accuracy using the best SAX params (from F4 or
    Phase 3 defaults) on a 5-ticker subset.
    """

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        tickers: list[str] | None = None,
        n_segments: int = 256,
        word_length: int = 4,
        word_stride: int = 1,
    ) -> None:
        self.runner = ExperimentRunner(cache_dir=cache_dir)
        self.tickers = tickers or HPO_TICKERS
        self.n_segments = n_segments
        self.word_length = word_length
        self.word_stride = word_stride

    def __call__(self, trial: optuna.Trial) -> float:
        embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128])
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        num_layers = trial.suggest_categorical("num_layers", [2, 3, 4, 6])
        dropout = trial.suggest_float("dropout", 0.05, 0.2, step=0.05)

        # Constraint: embed_dim must be divisible by num_heads
        if embed_dim % num_heads != 0:
            raise optuna.TrialPruned()

        config = ExperimentConfig(
            name=f"arch_hpo_trial_{trial.number}",
            tickers=self.tickers,
            interval="1h",
            train_end="2023-12-31",
            test_start="2024-01-01",
            n_segments=self.n_segments,
            alphabet_size=OPTIMAL_ALPHABET,
            word_length=self.word_length,
            word_stride=self.word_stride,
            min_word_freq=1,
            max_vocab_size=100,
            context_length=OPTIMAL_CONTEXT,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            epochs=80,
            batch_size=64,
            learning_rate=0.0005,
            patience=15,
            dwt_levels=OPTIMAL_DWT_LEVELS,
        )

        try:
            result = self.runner.run(config)
            return result.directional_accuracy
        except optuna.TrialPruned:
            raise
        except Exception:
            logger.exception("Trial %d failed", trial.number)
            return 0.0


def run_sax_hpo(
    n_trials: int = 30,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    tickers: list[str] | None = None,
) -> optuna.Study:
    """Run Optuna HPO for SAX parameters.

    Args:
        n_trials: Number of optimization trials.
        cache_dir: Path to cached price data.
        tickers: Ticker subset (defaults to HPO_TICKERS).

    Returns:
        Completed Optuna study.
    """
    study = optuna.create_study(
        study_name="sax_hpo",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    objective = SAXObjective(cache_dir=cache_dir, tickers=tickers)
    study.optimize(objective, n_trials=n_trials)
    return study


def run_architecture_hpo(
    n_trials: int = 25,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    tickers: list[str] | None = None,
    n_segments: int = 256,
    word_length: int = 4,
    word_stride: int = 1,
) -> optuna.Study:
    """Run Optuna HPO for model architecture parameters.

    Args:
        n_trials: Number of optimization trials.
        cache_dir: Path to cached price data.
        tickers: Ticker subset (defaults to HPO_TICKERS).
        n_segments: SAX n_segments (from F4 results or default).
        word_length: SAX word_length (from F4 results or default).
        word_stride: SAX word_stride (from F4 results or default).

    Returns:
        Completed Optuna study.
    """
    study = optuna.create_study(
        study_name="architecture_hpo",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    objective = ArchitectureObjective(
        cache_dir=cache_dir,
        tickers=tickers,
        n_segments=n_segments,
        word_length=word_length,
        word_stride=word_stride,
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def save_study_results(study: optuna.Study, path: Path) -> None:
    """Save Optuna study results as JSON.

    Args:
        study: Completed Optuna study.
        path: Output JSON file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    trials_data = []
    for t in study.trials:
        trials_data.append({
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": t.state.name,
            "duration_seconds": (
                (t.datetime_complete - t.datetime_start).total_seconds()
                if t.datetime_complete and t.datetime_start
                else None
            ),
        })

    data = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "best_trial": {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
        },
        "n_trials": len(study.trials),
        "trials": trials_data,
    }

    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("Study results saved to %s", path)
