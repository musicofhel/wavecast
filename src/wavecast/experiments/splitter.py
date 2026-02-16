"""Walk-forward train/test splitter at the raw price level."""

from __future__ import annotations

import numpy as np

from wavecast.core.exceptions import ConfigError
from wavecast.core.types import TimeSeries


def walk_forward_split(
    price_series: dict[str, TimeSeries],
    train_end: str,
    test_start: str,
) -> tuple[dict[str, TimeSeries], dict[str, TimeSeries]]:
    """Split raw price series into train and test sets by date.

    CRITICAL: This split happens BEFORE any DWT/SAX transformation.
    Each split is then independently decomposed, normalized, and tokenized
    by the ExperimentRunner to prevent information leakage.

    Args:
        price_series: Mapping of ticker -> TimeSeries with raw prices.
        train_end: End date for training data (inclusive), e.g. "2023-12-31".
        test_start: Start date for test data (inclusive), e.g. "2024-01-01".

    Returns:
        Tuple of (train_series, test_series) dicts, each mapping
        ticker -> TimeSeries. Tickers with no data in either split
        are excluded from that split.

    Raises:
        ConfigError: If train_end >= test_start or no data remains after split.
    """
    train_end_dt = np.datetime64(train_end)
    test_start_dt = np.datetime64(test_start)

    if train_end_dt >= test_start_dt:
        raise ConfigError(
            f"train_end ({train_end}) must be before test_start ({test_start})"
        )

    train_series: dict[str, TimeSeries] = {}
    test_series: dict[str, TimeSeries] = {}

    for ticker, ts in price_series.items():
        timestamps = ts.timestamps

        # Train: all points with timestamp <= train_end
        train_mask = timestamps <= train_end_dt
        # Test: all points with timestamp >= test_start
        test_mask = timestamps >= test_start_dt

        if np.any(train_mask):
            train_series[ticker] = TimeSeries(
                values=ts.values[train_mask].copy(),
                timestamps=timestamps[train_mask].copy(),
                ticker=ts.ticker,
                interval=ts.interval,
                column=ts.column,
            )

        if np.any(test_mask):
            test_series[ticker] = TimeSeries(
                values=ts.values[test_mask].copy(),
                timestamps=timestamps[test_mask].copy(),
                ticker=ts.ticker,
                interval=ts.interval,
                column=ts.column,
            )

    if not train_series:
        raise ConfigError(
            f"No training data: all timestamps are after train_end={train_end}"
        )
    if not test_series:
        raise ConfigError(
            f"No test data: all timestamps are before test_start={test_start}"
        )

    return train_series, test_series


def _split_by_index(
    price_series: dict[str, TimeSeries],
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
) -> tuple[dict[str, TimeSeries], dict[str, TimeSeries]]:
    """Split price series by integer index boundaries.

    All tickers are assumed to have the same length (aligned timestamps).
    """
    train_series: dict[str, TimeSeries] = {}
    test_series: dict[str, TimeSeries] = {}

    for ticker, ts in price_series.items():
        train_series[ticker] = TimeSeries(
            values=ts.values[train_start:train_end].copy(),
            timestamps=ts.timestamps[train_start:train_end].copy(),
            ticker=ts.ticker,
            interval=ts.interval,
            column=ts.column,
        )
        test_series[ticker] = TimeSeries(
            values=ts.values[test_start:test_end].copy(),
            timestamps=ts.timestamps[test_start:test_end].copy(),
            ticker=ts.ticker,
            interval=ts.interval,
            column=ts.column,
        )

    return train_series, test_series


def expanding_window_split(
    price_series: dict[str, TimeSeries],
    initial_train_end: int,
    test_window_size: int,
    step_size: int,
) -> list[tuple[dict[str, TimeSeries], dict[str, TimeSeries]]]:
    """Generate expanding-window train/test splits at the raw price level.

    The training set grows with each split while the test window slides forward.

    Args:
        price_series: Mapping of ticker -> TimeSeries with raw prices.
            All tickers must have the same length.
        initial_train_end: Index of the last sample in the first training set
            (exclusive upper bound, so train is [0, initial_train_end)).
        test_window_size: Number of samples in each test window.
        step_size: How many samples the boundary moves forward per split.

    Returns:
        List of (train_series, test_series) dict pairs.

    Raises:
        ConfigError: If parameters are invalid or data is too short for one split.
    """
    if not price_series:
        raise ConfigError("price_series must not be empty")

    n = len(next(iter(price_series.values())).values)

    if initial_train_end <= 0:
        raise ConfigError(f"initial_train_end must be > 0, got {initial_train_end}")
    if test_window_size <= 0:
        raise ConfigError(f"test_window_size must be > 0, got {test_window_size}")
    if step_size <= 0:
        raise ConfigError(f"step_size must be > 0, got {step_size}")
    if initial_train_end + test_window_size > n:
        raise ConfigError(
            f"Data too short for even one split: need {initial_train_end + test_window_size} "
            f"samples, have {n}"
        )

    splits: list[tuple[dict[str, TimeSeries], dict[str, TimeSeries]]] = []
    train_end = initial_train_end

    while train_end + test_window_size <= n:
        test_start = train_end
        test_end = test_start + test_window_size

        train_dict, test_dict = _split_by_index(
            price_series,
            train_start=0,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        splits.append((train_dict, test_dict))
        train_end += step_size

    return splits


def rolling_window_split(
    price_series: dict[str, TimeSeries],
    train_window_size: int,
    test_window_size: int,
    step_size: int,
) -> list[tuple[dict[str, TimeSeries], dict[str, TimeSeries]]]:
    """Generate rolling (fixed-size) window train/test splits at the raw price level.

    Both train and test windows slide forward together. Train window is fixed size.

    Args:
        price_series: Mapping of ticker -> TimeSeries with raw prices.
            All tickers must have the same length.
        train_window_size: Fixed number of samples in each training window.
        test_window_size: Number of samples in each test window.
        step_size: How many samples the window slides forward per split.

    Returns:
        List of (train_series, test_series) dict pairs.

    Raises:
        ConfigError: If parameters are invalid or data is too short for one split.
    """
    if not price_series:
        raise ConfigError("price_series must not be empty")

    n = len(next(iter(price_series.values())).values)

    if train_window_size <= 0:
        raise ConfigError(f"train_window_size must be > 0, got {train_window_size}")
    if test_window_size <= 0:
        raise ConfigError(f"test_window_size must be > 0, got {test_window_size}")
    if step_size <= 0:
        raise ConfigError(f"step_size must be > 0, got {step_size}")
    if train_window_size + test_window_size > n:
        raise ConfigError(
            f"Data too short for even one split: need {train_window_size + test_window_size} "
            f"samples, have {n}"
        )

    splits: list[tuple[dict[str, TimeSeries], dict[str, TimeSeries]]] = []
    offset = 0

    while offset + train_window_size + test_window_size <= n:
        train_start = offset
        train_end = offset + train_window_size
        test_start = train_end
        test_end = test_start + test_window_size

        train_dict, test_dict = _split_by_index(
            price_series,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        splits.append((train_dict, test_dict))
        offset += step_size

    return splits
