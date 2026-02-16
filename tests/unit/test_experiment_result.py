"""Tests for ExperimentResult."""

from wavecast.experiments.config import ExperimentConfig
from wavecast.experiments.result import ExperimentResult


def _make_result(**overrides) -> ExperimentResult:
    """Create a minimal ExperimentResult for testing."""
    config = ExperimentConfig(name="test", tickers=["AAPL"])
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


def test_result_creation():
    r = _make_result()
    assert r.token_accuracy == 0.35
    assert r.directional_accuracy == 0.58
    assert r.config.name == "test"


def test_ci_fields_are_tuples():
    r = _make_result()
    assert isinstance(r.token_accuracy_ci, tuple)
    assert len(r.token_accuracy_ci) == 2
    assert r.token_accuracy_ci[0] <= r.token_accuracy_ci[1]
    assert isinstance(r.directional_accuracy_ci, tuple)
    assert len(r.directional_accuracy_ci) == 2


def test_per_level_accuracy_int_keys():
    r = _make_result()
    assert all(isinstance(k, int) for k in r.per_level_accuracy)
    assert r.per_level_accuracy[1] == 0.30


def test_defaults_for_optional_fields():
    config = ExperimentConfig(name="minimal", tickers=["SPY"])
    r = ExperimentResult(
        config=config,
        token_accuracy=0.3,
        token_accuracy_ci=(0.25, 0.35),
        top3_accuracy=0.6,
        directional_accuracy=0.5,
        directional_accuracy_ci=(0.45, 0.55),
        baseline_most_frequent=0.1,
        baseline_persistence=0.4,
        baseline_momentum=0.45,
    )
    assert r.per_asset_accuracy == {}
    assert r.per_sector_accuracy == {}
    assert r.per_level_accuracy == {}
    assert r.vocab_size == 0
    assert r.unk_rate == 0.0
    assert r.n_train_samples == 0
    assert r.n_test_samples == 0
    assert r.training_time_seconds == 0.0
    assert r.timestamp == ""


def test_nested_config_accessible():
    r = _make_result()
    assert r.config.tickers == ["AAPL"]
    assert r.config.interval == "1h"
    assert r.config.alphabet_size == 7
