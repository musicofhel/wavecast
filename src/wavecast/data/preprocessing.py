"""Data preprocessing utilities for financial time series."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wavecast.core.exceptions import DataError
from wavecast.core.types import MarketLabel, TimeSeries


def normalize_zscore(ts: TimeSeries) -> TimeSeries:
    """Z-score normalize a time series to zero mean, unit variance.

    Args:
        ts: Input time series.

    Returns:
        New TimeSeries with z-score normalized values.

    Raises:
        DataError: If the series has zero standard deviation.
    """
    std = np.std(ts.values)
    if std == 0.0:
        raise DataError(
            f"Cannot z-score normalize {ts.ticker}: standard deviation is zero"
        )
    mean = np.mean(ts.values)
    normalized = (ts.values - mean) / std
    return TimeSeries(
        values=normalized,
        timestamps=ts.timestamps.copy(),
        ticker=ts.ticker,
        interval=ts.interval,
        column=ts.column,
    )


def log_returns(ts: TimeSeries) -> TimeSeries:
    """Compute log returns: ln(p_t / p_{t-1}).

    The resulting series is one element shorter than the input.

    Args:
        ts: Input time series of prices.

    Returns:
        New TimeSeries of log returns.

    Raises:
        DataError: If the series is too short or contains non-positive values.
    """
    if ts.length < 2:
        raise DataError(
            f"Cannot compute log returns for {ts.ticker}: need at least 2 values"
        )
    if np.any(ts.values <= 0):
        raise DataError(
            f"Cannot compute log returns for {ts.ticker}: "
            "values must be strictly positive"
        )
    returns = np.diff(np.log(ts.values))
    return TimeSeries(
        values=returns,
        timestamps=ts.timestamps[1:].copy(),
        ticker=ts.ticker,
        interval=ts.interval,
        column="log_return",
    )


def handle_nans(ts: TimeSeries, method: str = "ffill") -> TimeSeries:
    """Handle NaN values in a time series.

    Args:
        ts: Input time series.
        method: Fill method - 'ffill' (forward fill), 'bfill' (backward fill),
            'interpolate' (linear interpolation), or 'drop' (remove NaN rows).

    Returns:
        New TimeSeries with NaNs handled.

    Raises:
        DataError: If method is unknown or all values are NaN.
    """
    values = ts.values.copy().astype(np.float64)
    timestamps = ts.timestamps.copy()
    nan_mask = np.isnan(values)

    if not np.any(nan_mask):
        return TimeSeries(
            values=values,
            timestamps=timestamps,
            ticker=ts.ticker,
            interval=ts.interval,
            column=ts.column,
        )

    if np.all(nan_mask):
        raise DataError(f"All values are NaN for {ts.ticker}")

    if method == "ffill":
        # Forward fill: propagate last valid observation
        for i in range(1, len(values)):
            if np.isnan(values[i]):
                values[i] = values[i - 1]
        # If leading NaNs remain, backfill them with first valid value
        first_valid = np.argmax(~np.isnan(values))
        values[:first_valid] = values[first_valid]
    elif method == "bfill":
        # Backward fill: propagate next valid observation
        for i in range(len(values) - 2, -1, -1):
            if np.isnan(values[i]):
                values[i] = values[i + 1]
        # If trailing NaNs remain, forward fill them
        last_valid_idx = len(values) - 1 - np.argmax(~np.isnan(values[::-1]))
        values[last_valid_idx + 1 :] = values[last_valid_idx]
    elif method == "interpolate":
        valid_idx = np.where(~nan_mask)[0]
        valid_vals = values[valid_idx]
        all_idx = np.arange(len(values))
        values = np.interp(all_idx, valid_idx, valid_vals)
    elif method == "drop":
        keep = ~nan_mask
        values = values[keep]
        timestamps = timestamps[keep]
        if len(values) == 0:
            raise DataError(f"No values remain after dropping NaNs for {ts.ticker}")
    else:
        raise DataError(
            f"Unknown NaN handling method '{method}'. "
            "Use 'ffill', 'bfill', 'interpolate', or 'drop'."
        )

    return TimeSeries(
        values=values,
        timestamps=timestamps,
        ticker=ts.ticker,
        interval=ts.interval,
        column=ts.column,
    )


def label_returns(
    ts: TimeSeries, threshold: float = 0.001
) -> NDArray:
    """Label each return as UP, DOWN, or FLAT based on threshold.

    Computes simple returns from prices, then labels each return.

    Args:
        ts: Input time series of prices.
        threshold: Absolute return threshold for FLAT classification.

    Returns:
        Array of MarketLabel values (length = ts.length - 1).

    Raises:
        DataError: If the series is too short.
    """
    if ts.length < 2:
        raise DataError(
            f"Cannot label returns for {ts.ticker}: need at least 2 values"
        )

    returns = np.diff(ts.values) / ts.values[:-1]

    labels = np.empty(len(returns), dtype=object)
    for i, r in enumerate(returns):
        if r > threshold:
            labels[i] = MarketLabel.UP
        elif r < -threshold:
            labels[i] = MarketLabel.DOWN
        else:
            labels[i] = MarketLabel.FLAT

    return labels
