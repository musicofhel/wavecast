"""Tests for experiment results storage and comparison."""

import json

import pandas as pd

from wavecast.experiments.config import ExperimentConfig
from wavecast.experiments.result import ExperimentResult
from wavecast.experiments.storage import (
    compare_results,
    load_results,
    save_results,
)


def _make_result(name: str = "test", **overrides) -> ExperimentResult:
    """Create a minimal ExperimentResult for testing."""
    config = ExperimentConfig(name=name, tickers=["AAPL"])
    defaults = dict(
        config=config,
        token_accuracy=0.35,
        token_accuracy_ci=(0.30, 0.40),
        top3_accuracy=0.65,
        directional_accuracy=0.58,
        directional_accuracy_ci=(0.52, 0.64),
        baseline_most_frequent=0.15,
        baseline_persistence=0.45,
        baseline_momentum=0.50,
        per_asset_accuracy={"AAPL": 0.36},
        per_sector_accuracy={"tech": 0.36},
        per_level_accuracy={1: 0.30, 2: 0.35, 3: 0.40},
        vocab_size=150,
        unk_rate=0.05,
        n_train_samples=5000,
        n_test_samples=1200,
        training_time_seconds=12.5,
        timestamp="2026-02-16T00:00:00",
    )
    defaults.update(overrides)
    return ExperimentResult(**defaults)


def test_save_load_roundtrip(tmp_path):
    results = [_make_result("exp1"), _make_result("exp2")]
    path = tmp_path / "results.json"
    save_results(results, path)

    loaded = load_results(path)
    assert len(loaded) == 2
    assert loaded[0].config.name == "exp1"
    assert loaded[1].config.name == "exp2"


def test_roundtrip_preserves_metrics(tmp_path):
    original = _make_result()
    path = tmp_path / "result.json"
    save_results([original], path)
    loaded = load_results(path)[0]

    assert loaded.token_accuracy == original.token_accuracy
    assert loaded.top3_accuracy == original.top3_accuracy
    assert loaded.directional_accuracy == original.directional_accuracy
    assert loaded.baseline_most_frequent == original.baseline_most_frequent
    assert loaded.baseline_persistence == original.baseline_persistence
    assert loaded.baseline_momentum == original.baseline_momentum
    assert loaded.vocab_size == original.vocab_size
    assert loaded.unk_rate == original.unk_rate
    assert loaded.n_train_samples == original.n_train_samples
    assert loaded.n_test_samples == original.n_test_samples
    assert loaded.timestamp == original.timestamp


def test_roundtrip_preserves_ci_as_tuples(tmp_path):
    original = _make_result()
    path = tmp_path / "result.json"
    save_results([original], path)
    loaded = load_results(path)[0]

    assert isinstance(loaded.token_accuracy_ci, tuple)
    assert loaded.token_accuracy_ci == original.token_accuracy_ci
    assert isinstance(loaded.directional_accuracy_ci, tuple)
    assert loaded.directional_accuracy_ci == original.directional_accuracy_ci


def test_roundtrip_preserves_int_keys(tmp_path):
    original = _make_result()
    path = tmp_path / "result.json"
    save_results([original], path)
    loaded = load_results(path)[0]

    assert all(isinstance(k, int) for k in loaded.per_level_accuracy)
    assert loaded.per_level_accuracy == original.per_level_accuracy


def test_roundtrip_preserves_config(tmp_path):
    original = _make_result()
    path = tmp_path / "result.json"
    save_results([original], path)
    loaded = load_results(path)[0]

    assert loaded.config.name == original.config.name
    assert loaded.config.tickers == original.config.tickers
    assert loaded.config.interval == original.config.interval
    assert loaded.config.alphabet_size == original.config.alphabet_size


def test_save_creates_parent_dirs(tmp_path):
    path = tmp_path / "sub" / "dir" / "results.json"
    save_results([_make_result()], path)
    assert path.exists()


def test_saved_json_is_valid(tmp_path):
    path = tmp_path / "results.json"
    save_results([_make_result()], path)
    data = json.loads(path.read_text())
    assert isinstance(data, list)
    assert len(data) == 1
    assert "config" in data[0]
    assert "token_accuracy" in data[0]


def test_roundtrip_none_fields(tmp_path):
    result = _make_result(
        per_asset_accuracy={},
        per_sector_accuracy={},
        per_level_accuracy={},
    )
    path = tmp_path / "result.json"
    save_results([result], path)
    loaded = load_results(path)[0]
    assert loaded.per_asset_accuracy == {}
    assert loaded.per_sector_accuracy == {}
    assert loaded.per_level_accuracy == {}


def test_roundtrip_config_with_none_dwt_levels(tmp_path):
    config = ExperimentConfig(name="none_levels", tickers=["SPY"], dwt_levels=None)
    result = _make_result()
    # Override config
    result.config = config
    path = tmp_path / "result.json"
    save_results([result], path)
    loaded = load_results(path)[0]
    assert loaded.config.dwt_levels is None


def test_compare_results_basic():
    results = [
        _make_result("exp1", directional_accuracy=0.55),
        _make_result("exp2", directional_accuracy=0.60),
        _make_result("exp3", directional_accuracy=0.52),
    ]
    df = compare_results(results)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    # Sorted by directional_accuracy descending
    assert df.iloc[0]["name"] == "exp2"
    assert df.iloc[1]["name"] == "exp1"
    assert df.iloc[2]["name"] == "exp3"


def test_compare_results_has_expected_columns():
    results = [_make_result()]
    df = compare_results(results)
    expected_cols = {
        "name", "token_accuracy", "top3_accuracy", "directional_accuracy",
        "baseline_most_frequent", "baseline_persistence", "baseline_momentum",
        "vocab_size", "unk_rate", "n_train_samples", "n_test_samples",
        "training_time_seconds",
    }
    assert expected_cols.issubset(set(df.columns))


def test_compare_results_ci_columns():
    results = [_make_result()]
    df = compare_results(results, metric="directional_accuracy")
    assert "directional_accuracy_ci_lower" in df.columns
    assert "directional_accuracy_ci_upper" in df.columns


def test_compare_results_custom_metric():
    results = [
        _make_result("a", token_accuracy=0.40),
        _make_result("b", token_accuracy=0.30),
    ]
    df = compare_results(results, metric="token_accuracy")
    # Sorted by token_accuracy descending
    assert df.iloc[0]["name"] == "a"
    assert df.iloc[1]["name"] == "b"
