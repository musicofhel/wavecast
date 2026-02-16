"""Experiment result container."""

from __future__ import annotations

from dataclasses import dataclass, field

from wavecast.experiments.config import ExperimentConfig


@dataclass
class ExperimentResult:
    """Results from a single experiment run.

    Contains accuracy metrics with bootstrap confidence intervals,
    baselines, per-asset/sector/level breakdowns, and metadata.
    """

    config: ExperimentConfig

    # Primary metrics
    token_accuracy: float
    token_accuracy_ci: tuple[float, float]
    top3_accuracy: float
    directional_accuracy: float
    directional_accuracy_ci: tuple[float, float]

    # Baselines
    baseline_most_frequent: float
    baseline_persistence: float
    baseline_momentum: float

    # Per-dimension breakdowns
    per_asset_accuracy: dict[str, float] = field(default_factory=dict)
    per_sector_accuracy: dict[str, float] = field(default_factory=dict)
    per_level_accuracy: dict[int, float] = field(default_factory=dict)

    # Dataset info
    vocab_size: int = 0
    unk_rate: float = 0.0
    n_train_samples: int = 0
    n_test_samples: int = 0

    # Timing
    training_time_seconds: float = 0.0
    timestamp: str = ""

    # Multi-split aggregation (populated when split_mode != "single")
    n_splits: int = 1
    token_accuracy_std: float = 0.0
    directional_accuracy_std: float = 0.0
