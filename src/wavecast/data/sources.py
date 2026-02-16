"""Data source loaders for financial time series."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from wavecast.core.exceptions import DataError, DataNotFoundError
from wavecast.core.types import TimeSeries

# Load .env if not already loaded (covers notebooks, scripts, direct imports)
load_dotenv()

# Massive API timespan mapping from wavecast intervals
_INTERVAL_TO_TIMESPAN = {
    "1m": ("minute", 1),
    "5m": ("minute", 5),
    "15m": ("minute", 15),
    "30m": ("minute", 30),
    "1h": ("hour", 1),
    "1d": ("day", 1),
    "1w": ("week", 1),
    "1mo": ("month", 1),
}


def _get_massive_client():  # type: ignore[no-untyped-def]
    """Get an authenticated Massive REST client."""
    api_key = os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        raise DataError(
            "MASSIVE_API_KEY environment variable is not set. "
            "Export it with: export MASSIVE_API_KEY='your_key'"
        )
    from massive import RESTClient

    return RESTClient(api_key=api_key)


def fetch_massive(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> TimeSeries:
    """Fetch OHLCV data from Massive.com API.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').
        start: Start date (YYYY-MM-DD). Defaults to 5 years ago.
        end: End date (YYYY-MM-DD). Defaults to today.
        interval: Data interval ('1d', '1h', '5m', etc.).

    Returns:
        TimeSeries with close prices and timestamps.

    Raises:
        DataNotFoundError: If ticker returns no data.
        DataError: On API failure.
    """
    client = _get_massive_client()

    if start is None:
        start = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    timespan_info = _INTERVAL_TO_TIMESPAN.get(interval)
    if timespan_info is None:
        raise DataError(
            f"Unsupported interval '{interval}'. "
            f"Supported: {list(_INTERVAL_TO_TIMESPAN.keys())}"
        )
    timespan, multiplier = timespan_info

    try:
        aggs = []
        for a in client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start,
            to=end,
            limit=50000,
        ):
            aggs.append(a)
    except Exception as e:
        raise DataError(f"Massive API error for {ticker}: {e}") from e

    if not aggs:
        raise DataNotFoundError(
            f"No data returned for ticker '{ticker}' "
            f"(start={start}, end={end}, interval={interval})"
        )

    closes = np.array([a.close for a in aggs], dtype=np.float64)
    # Massive timestamps are Unix ms
    timestamps = np.array(
        [np.datetime64(int(a.timestamp), "ms") for a in aggs],
        dtype="datetime64[ns]",
    )

    return TimeSeries(
        values=closes,
        timestamps=timestamps,
        ticker=ticker,
        interval=interval,
        column="close",
    )


# Keep fetch_massive as the primary, alias for backward compat
fetch = fetch_massive


def load_csv(
    path: Path,
    ticker: str,
    column: str = "close",
    date_column: str = "date",
) -> TimeSeries:
    """Load a time series from a CSV file.

    Args:
        path: Path to the CSV file.
        ticker: Ticker label to assign.
        column: Name of the value column (case-insensitive).
        date_column: Name of the date column (case-insensitive).

    Returns:
        TimeSeries sorted by date ascending.

    Raises:
        DataNotFoundError: If the file does not exist.
        DataError: On parse or column errors.
    """
    path = Path(path)
    if not path.exists():
        raise DataNotFoundError(f"CSV file not found: {path}")

    try:
        df = pd.read_csv(path, parse_dates=True)
    except Exception as e:
        raise DataError(f"Failed to read CSV {path}: {e}") from e

    col_map = {c.lower(): c for c in df.columns}

    date_col_actual = col_map.get(date_column.lower())
    if date_col_actual is None:
        raise DataError(
            f"Date column '{date_column}' not found in {path}. "
            f"Available: {list(df.columns)}"
        )

    value_col_actual = col_map.get(column.lower())
    if value_col_actual is None:
        raise DataError(
            f"Value column '{column}' not found in {path}. "
            f"Available: {list(df.columns)}"
        )

    df[date_col_actual] = pd.to_datetime(df[date_col_actual])
    df = df.sort_values(date_col_actual).reset_index(drop=True)

    values = df[value_col_actual].to_numpy(dtype=np.float64)
    timestamps = df[date_col_actual].to_numpy(dtype="datetime64[ns]")

    return TimeSeries(
        values=values,
        timestamps=timestamps,
        ticker=ticker,
        interval="1d",
        column=column.lower(),
    )


def load_parquet(
    path: Path,
    ticker: str,
    column: str = "close",
    date_column: str = "date",
) -> TimeSeries:
    """Load a time series from a Parquet file.

    Args:
        path: Path to the Parquet file.
        ticker: Ticker label to assign.
        column: Name of the value column (case-insensitive).
        date_column: Name of the date column (case-insensitive).

    Returns:
        TimeSeries sorted by date ascending.

    Raises:
        DataNotFoundError: If the file does not exist.
        DataError: On parse or column errors.
    """
    path = Path(path)
    if not path.exists():
        raise DataNotFoundError(f"Parquet file not found: {path}")

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        raise DataError(f"Failed to read Parquet {path}: {e}") from e

    col_map = {c.lower(): c for c in df.columns}

    date_col_actual = col_map.get(date_column.lower())
    if date_col_actual is None:
        raise DataError(
            f"Date column '{date_column}' not found in {path}. "
            f"Available: {list(df.columns)}"
        )

    value_col_actual = col_map.get(column.lower())
    if value_col_actual is None:
        raise DataError(
            f"Value column '{column}' not found in {path}. "
            f"Available: {list(df.columns)}"
        )

    df[date_col_actual] = pd.to_datetime(df[date_col_actual])
    df = df.sort_values(date_col_actual).reset_index(drop=True)

    values = df[value_col_actual].to_numpy(dtype=np.float64)
    timestamps = df[date_col_actual].to_numpy(dtype="datetime64[ns]")

    return TimeSeries(
        values=values,
        timestamps=timestamps,
        ticker=ticker,
        interval="1d",
        column=column.lower(),
    )
