"""Tests for ExperimentRunner with expanding/rolling split modes."""

import numpy as np
import pytest

from wavecast.core.types import TimeSeries
from wavecast.data.cache import ParquetCache
from wavecast.experiments.config import ExperimentConfig
from wavecast.experiments.runner import ExperimentRunner


def _seed_cache(cache: ParquetCache, tickers: list[str], interval: str = "1h") -> None:
    """Seed a ParquetCache with synthetic time series for testing.

    Creates 2000 hourly bars starting 2021-01-01.
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


def test_expanding_split_mode(runner_with_cache):
    """ExperimentRunner with expanding split_mode runs and returns aggregated results."""
    runner, tickers = runner_with_cache
    config = ExperimentConfig(
        name="test_expanding",
        tickers=tickers,
        interval="1h",
        # These are ignored for expanding mode; splits are index-based
        train_end="2021-02-15",
        test_start="2021-02-16",
        split_mode="expanding",
        initial_train_size=800,
        test_window_size=400,
        step_size=400,
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
        dwt_levels=[1],
    )
    result = runner.run(config)

    assert result.n_splits > 1
    assert 0.0 <= result.token_accuracy <= 1.0
    assert result.token_accuracy_std >= 0.0
    assert 0.0 <= result.directional_accuracy <= 1.0
    assert result.directional_accuracy_std >= 0.0
    assert result.n_train_samples > 0
    assert result.n_test_samples > 0


def test_rolling_split_mode(runner_with_cache):
    """ExperimentRunner with rolling split_mode runs and returns aggregated results."""
    runner, tickers = runner_with_cache
    config = ExperimentConfig(
        name="test_rolling",
        tickers=tickers,
        interval="1h",
        train_end="2021-02-15",
        test_start="2021-02-16",
        split_mode="rolling",
        train_window_size=800,
        test_window_size=400,
        step_size=400,
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
        dwt_levels=[1],
    )
    result = runner.run(config)

    assert result.n_splits > 1
    assert 0.0 <= result.token_accuracy <= 1.0
    assert result.token_accuracy_std >= 0.0


def test_single_split_mode_backward_compat(runner_with_cache):
    """Default split_mode='single' still works as before."""
    runner, tickers = runner_with_cache
    config = ExperimentConfig(
        name="test_single",
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

    assert result.n_splits == 1
    assert result.token_accuracy_std == 0.0
    assert 0.0 <= result.token_accuracy <= 1.0


def test_invalid_split_mode_raises(runner_with_cache):
    runner, tickers = runner_with_cache
    config = ExperimentConfig(
        name="test_bad_mode",
        tickers=tickers,
        interval="1h",
        split_mode="invalid",
    )
    from wavecast.core.exceptions import ConfigError
    with pytest.raises(ConfigError, match="Unknown split_mode"):
        runner.run(config)
