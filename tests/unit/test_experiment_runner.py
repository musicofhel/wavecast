"""Tests for ExperimentRunner — full pipeline with tiny synthetic data."""

import numpy as np
import pytest

from wavecast.core.types import TimeSeries
from wavecast.data.cache import ParquetCache
from wavecast.experiments.config import ExperimentConfig
from wavecast.experiments.runner import ExperimentRunner


def _seed_cache(cache: ParquetCache, tickers: list[str], interval: str = "1h") -> None:
    """Seed a ParquetCache with synthetic time series for testing.

    Creates 2000 hourly bars starting 2021-01-01 (~83 days).
    Split dates in tests should fall within this range.
    """
    rng = np.random.default_rng(42)
    n = 2000
    base = np.datetime64("2021-01-01")
    timestamps = np.array([base + np.timedelta64(i, "h") for i in range(n)])
    for ticker in tickers:
        values = 100 + np.cumsum(rng.standard_normal(n) * 0.3)
        ts = TimeSeries(
            values=values,
            timestamps=timestamps,
            ticker=ticker,
            interval=interval,
        )
        cache.put(ts)


@pytest.fixture
def runner_with_cache(tmp_path):
    """Create an ExperimentRunner with seeded cache."""
    cache_dir = tmp_path / "cache"
    cache = ParquetCache(cache_dir)
    tickers = ["AAPL", "MSFT"]
    _seed_cache(cache, tickers, interval="1h")
    runner = ExperimentRunner(cache_dir=cache_dir)
    return runner, tickers


def test_runner_completes(runner_with_cache):
    """Full pipeline: seed data -> split -> DWT -> SAX -> train -> evaluate."""
    runner, tickers = runner_with_cache
    config = ExperimentConfig(
        name="test_small",
        tickers=tickers,
        interval="1h",
        train_end="2021-02-15",
        test_start="2021-02-16",
        # Small params for fast test
        n_segments=32,
        alphabet_size=5,
        word_length=3,
        word_stride=1,
        min_word_freq=1,
        max_vocab_size=100,
        context_length=4,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        epochs=2,
        batch_size=32,
        learning_rate=0.001,
        patience=2,
        dwt_levels=[1, 2],
    )
    result = runner.run(config)

    assert result.config.name == "test_small"
    assert 0.0 <= result.token_accuracy <= 1.0
    assert 0.0 <= result.top3_accuracy <= 1.0
    assert 0.0 <= result.directional_accuracy <= 1.0
    assert result.n_train_samples > 0
    assert result.n_test_samples > 0
    assert result.vocab_size > 2  # At least PAD + UNK + some words
    assert result.training_time_seconds > 0
    assert result.timestamp != ""
    assert isinstance(result.token_accuracy_ci, tuple)
    assert isinstance(result.directional_accuracy_ci, tuple)


def test_runner_baselines_populated(runner_with_cache):
    runner, tickers = runner_with_cache
    config = ExperimentConfig(
        name="test_baselines",
        tickers=tickers,
        interval="1h",
        train_end="2021-02-15",
        test_start="2021-02-16",
        n_segments=32,
        alphabet_size=5,
        word_length=3,
        min_word_freq=1,
        max_vocab_size=100,
        context_length=4,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        epochs=2,
        batch_size=32,
        patience=2,
        dwt_levels=[1],
    )
    result = runner.run(config)

    assert 0.0 <= result.baseline_most_frequent <= 1.0
    assert 0.0 <= result.baseline_persistence <= 1.0
    assert 0.0 <= result.baseline_momentum <= 1.0


def test_runner_per_level_accuracy(runner_with_cache):
    runner, tickers = runner_with_cache
    config = ExperimentConfig(
        name="test_levels",
        tickers=tickers,
        interval="1h",
        train_end="2021-02-15",
        test_start="2021-02-16",
        n_segments=32,
        alphabet_size=5,
        word_length=3,
        min_word_freq=1,
        max_vocab_size=100,
        context_length=4,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        epochs=2,
        batch_size=32,
        patience=2,
        dwt_levels=[1, 2],
    )
    result = runner.run(config)

    # Should have accuracy for each level used
    assert len(result.per_level_accuracy) > 0
    for lvl, acc in result.per_level_accuracy.items():
        assert isinstance(lvl, int)
        assert 0.0 <= acc <= 1.0


def test_runner_sweep(runner_with_cache):
    runner, tickers = runner_with_cache
    configs = [
        ExperimentConfig(
            name=f"sweep_{i}",
            tickers=tickers,
            interval="1h",
            train_end="2021-02-15",
            test_start="2021-02-16",
            n_segments=32,
            alphabet_size=a,
            word_length=3,
            min_word_freq=1,
            max_vocab_size=100,
            context_length=4,
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            epochs=2,
            batch_size=32,
            patience=2,
            dwt_levels=[1],
        )
        for i, a in enumerate([3, 5])
    ]
    results = runner.run_sweep(configs)
    assert len(results) == 2
    assert results[0].config.name == "sweep_0"
    assert results[1].config.name == "sweep_1"


def test_runner_missing_ticker_raises(tmp_path):
    cache_dir = tmp_path / "empty_cache"
    cache_dir.mkdir()
    runner = ExperimentRunner(cache_dir=cache_dir)
    config = ExperimentConfig(
        name="missing",
        tickers=["NONEXISTENT"],
        interval="1h",
    )
    from wavecast.core.exceptions import DataNotFoundError

    with pytest.raises(DataNotFoundError):
        runner.run(config)
