"""Deterministic synthetic data generators for testing."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wavecast.core.types import MarketLabel, TimeSeries


def make_timestamps(n: int, start: str = "2020-01-01") -> NDArray[np.datetime64]:
    """Generate daily timestamps."""
    return np.arange(start, n, dtype="datetime64[D]")[:n]


def make_sine_series(
    n: int = 500, freq: float = 0.05, noise: float = 0.1, seed: int = 42
) -> TimeSeries:
    """Deterministic sine wave + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    values = 100 + 10 * np.sin(2 * np.pi * freq * t) + noise * rng.standard_normal(n)
    return TimeSeries(
        values=values,
        timestamps=make_timestamps(n),
        ticker="SINE",
        interval="1d",
    )


def make_random_walk(n: int = 500, seed: int = 42) -> TimeSeries:
    """Random walk with fixed seed."""
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal(n) * 0.01
    values = 100 * np.exp(np.cumsum(steps))
    return TimeSeries(
        values=values,
        timestamps=make_timestamps(n),
        ticker="WALK",
        interval="1d",
    )


def make_trending_series(
    n: int = 500, trend: float = 0.001, seed: int = 42
) -> TimeSeries:
    """Upward-trending random walk."""
    rng = np.random.default_rng(seed)
    steps = trend + rng.standard_normal(n) * 0.005
    values = 100 * np.exp(np.cumsum(steps))
    return TimeSeries(
        values=values,
        timestamps=make_timestamps(n),
        ticker="TREND",
        interval="1d",
    )


def make_mean_reverting_series(
    n: int = 500, theta: float = 0.1, mu: float = 100.0, sigma: float = 2.0, seed: int = 42
) -> TimeSeries:
    """Ornstein-Uhlenbeck mean-reverting process."""
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    values[0] = mu
    for i in range(1, n):
        values[i] = values[i - 1] + theta * (mu - values[i - 1]) + sigma * rng.standard_normal()
    return TimeSeries(
        values=values,
        timestamps=make_timestamps(n),
        ticker="MREV",
        interval="1d",
    )


def make_labels(n: int = 500, seed: int = 42) -> NDArray:
    """Random MarketLabel array."""
    rng = np.random.default_rng(seed)
    choices = [MarketLabel.UP.value, MarketLabel.DOWN.value, MarketLabel.FLAT.value]
    return rng.choice(choices, size=n)


def make_feature_matrix(
    n: int = 100, d: int = 40, seed: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate synthetic feature matrix and target."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    # Target is a noisy linear combination of first few features
    w = rng.standard_normal(d) * 0.1
    y = X @ w + rng.standard_normal(n) * 0.01
    return X, y
