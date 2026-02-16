"""Results storage and comparison utilities for experiments."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from wavecast.experiments.config import ExperimentConfig
from wavecast.experiments.result import ExperimentResult

DEFAULT_EXPERIMENTS_DIR = Path.home() / ".wavecast" / "experiments"


def _serialize_result(result: ExperimentResult) -> dict:
    """Convert an ExperimentResult to a JSON-serializable dict.

    Handles:
    - tuple CIs -> list (JSON has no tuple type)
    - int dict keys -> str keys (JSON requires string keys)
    - nested dataclass (ExperimentConfig) -> dict
    """
    d = asdict(result)
    # tuple CIs become lists via asdict, which is fine for JSON
    # int dict keys need string conversion
    if "per_level_accuracy" in d:
        d["per_level_accuracy"] = {
            str(k): v for k, v in d["per_level_accuracy"].items()
        }
    if "config" in d and "dwt_levels" in d["config"]:
        # dwt_levels is list[int] | None — JSON handles both fine
        pass
    return d


def _deserialize_result(d: dict) -> ExperimentResult:
    """Reconstruct an ExperimentResult from a deserialized dict.

    Reverses _serialize_result:
    - list CIs -> tuple
    - str dict keys for per_level_accuracy -> int keys
    - nested dict -> ExperimentConfig
    """
    config_dict = d.pop("config")
    config = ExperimentConfig(**config_dict)

    # Restore tuple CIs
    for ci_field in ("token_accuracy_ci", "directional_accuracy_ci"):
        if ci_field in d and isinstance(d[ci_field], list):
            d[ci_field] = tuple(d[ci_field])

    # Restore int keys for per_level_accuracy
    if "per_level_accuracy" in d:
        d["per_level_accuracy"] = {
            int(k): v for k, v in d["per_level_accuracy"].items()
        }

    return ExperimentResult(config=config, **d)


def save_results(results: list[ExperimentResult], path: Path) -> None:
    """Save experiment results to a JSON file.

    Args:
        results: List of ExperimentResult to save.
        path: Output file path (.json).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized = [_serialize_result(r) for r in results]
    path.write_text(json.dumps(serialized, indent=2, default=str))


def load_results(path: Path) -> list[ExperimentResult]:
    """Load experiment results from a JSON file.

    Args:
        path: Path to a JSON file written by save_results.

    Returns:
        List of ExperimentResult objects.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    return [_deserialize_result(d) for d in data]


def compare_results(
    results: list[ExperimentResult],
    metric: str = "directional_accuracy",
) -> pd.DataFrame:
    """Build a comparison table across experiment results.

    Args:
        results: List of ExperimentResult to compare.
        metric: Primary metric column to highlight. Must be an attribute
            of ExperimentResult (e.g. "directional_accuracy", "token_accuracy").

    Returns:
        DataFrame with one row per experiment and columns for key metrics.
    """
    rows = []
    for r in results:
        row = {
            "name": r.config.name,
            "token_accuracy": r.token_accuracy,
            "top3_accuracy": r.top3_accuracy,
            "directional_accuracy": r.directional_accuracy,
            "baseline_most_frequent": r.baseline_most_frequent,
            "baseline_persistence": r.baseline_persistence,
            "baseline_momentum": r.baseline_momentum,
            "vocab_size": r.vocab_size,
            "unk_rate": r.unk_rate,
            "n_train_samples": r.n_train_samples,
            "n_test_samples": r.n_test_samples,
            "training_time_seconds": r.training_time_seconds,
        }
        # Add CI bounds for the primary metric if available
        ci_field = f"{metric}_ci"
        ci_val = getattr(r, ci_field, None)
        if ci_val is not None:
            row[f"{metric}_ci_lower"] = ci_val[0]
            row[f"{metric}_ci_upper"] = ci_val[1]
        rows.append(row)

    df = pd.DataFrame(rows)
    if metric in df.columns:
        df = df.sort_values(metric, ascending=False).reset_index(drop=True)
    return df
