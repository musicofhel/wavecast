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
