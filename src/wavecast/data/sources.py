"""Data source loaders for financial time series."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from wavecast.core.exceptions import DataError, DataNotFoundError
from wavecast.core.types import TimeSeries

logger = logging.getLogger(__name__)

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


def fetch_massive_ohlcv(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch full OHLCV data from Massive.com as a DataFrame.

    Returns DataFrame with columns: timestamp, open, high, low, close, volume.
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

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [int(a.timestamp) for a in aggs], unit="ms", utc=True
            ),
            "open": [a.open for a in aggs],
            "high": [a.high for a in aggs],
            "low": [a.low for a in aggs],
            "close": [a.close for a in aggs],
            "volume": [a.volume for a in aggs],
        }
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_latest_bars(
    ticker: str,
    interval: str = "1h",
    n_bars: int = 200,
) -> pd.DataFrame:
    """Fetch the most recent N bars for a ticker.

    Always fetches fresh from API (no Parquet cache).
    Computes start date as now - n_bars * interval_duration * 1.5 to buffer for
    weekends, holidays, and trading gaps.

    Args:
        ticker: Stock ticker symbol.
        interval: Data interval (must be in _INTERVAL_TO_TIMESPAN).
        n_bars: Number of recent bars to fetch.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
        Sorted by timestamp ascending, limited to last n_bars rows.

    Raises:
        DataError: If interval is unsupported or API fails.
        DataNotFoundError: If no data returned.
    """
    interval_hours: dict[str, float] = {
        "1m": 1 / 60,
        "5m": 5 / 60,
        "15m": 0.25,
        "30m": 0.5,
        "1h": 1.0,
        "1d": 24.0,
        "1w": 168.0,
        "1mo": 720.0,
    }

    hours_per_bar = interval_hours.get(interval)
    if hours_per_bar is None:
        raise DataError(
            f"Unsupported interval '{interval}'. "
            f"Supported: {list(_INTERVAL_TO_TIMESPAN.keys())}"
        )

    # For intraday intervals, account for limited trading hours (~7h/day, 5d/week).
    # Stocks trade ~7 hours/day on 5 of 7 calendar days, so each trading-hour bar
    # corresponds to ~24/7 * 7/5 ≈ 4.8 calendar hours. Use 5x buffer for safety.
    buffer_multiplier = 5.0 if hours_per_bar < 24 else 1.5
    total_hours = n_bars * hours_per_bar * buffer_multiplier
    start = (datetime.now() - timedelta(hours=total_hours)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")

    df = fetch_massive_ohlcv(ticker, start=start, end=end, interval=interval)

    # Return only the last n_bars
    return df.tail(n_bars).reset_index(drop=True)


def fetch_universe(
    tickers: list[str],
    start: str = "2021-02-01",
    end: str = "2025-12-31",
    intervals: list[str] | None = None,
    cache_dir: Path | None = None,
    rate_limit_pause: float = 12.5,
    fallbacks: dict[str, str] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Fetch OHLCV data for multiple tickers and intervals, caching to Parquet.

    Args:
        tickers: List of ticker symbols.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        intervals: List of intervals to fetch (default: ['1h', '1d']).
        cache_dir: Directory for Parquet cache (default: ~/.wavecast/cache).
        rate_limit_pause: Seconds between API calls (Massive free tier: 5/min).
        fallbacks: Mapping of ticker -> fallback ticker if primary fails.

    Returns:
        Nested dict: {ticker: {interval: DataFrame}}.
    """
    if intervals is None:
        intervals = ["1h", "1d"]
    if cache_dir is None:
        cache_dir = Path.home() / ".wavecast" / "cache"
    if fallbacks is None:
        fallbacks = {}
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, pd.DataFrame]] = {}

    for ticker in tickers:
        results[ticker] = {}
        for interval in intervals:
            cache_path = cache_dir / f"{ticker}_{interval}_ohlcv.parquet"

            if cache_path.exists():
                logger.info("Cache hit: %s %s", ticker, interval)
                results[ticker][interval] = pd.read_parquet(cache_path)
                continue

            actual_ticker = ticker
            try:
                logger.info("Fetching %s %s ...", ticker, interval)
                df = fetch_massive_ohlcv(
                    ticker, start=start, end=end, interval=interval
                )
            except (DataError, DataNotFoundError) as e:
                if ticker in fallbacks:
                    actual_ticker = fallbacks[ticker]
                    logger.warning(
                        "%s failed (%s), trying fallback %s",
                        ticker,
                        e,
                        actual_ticker,
                    )
                    df = fetch_massive_ohlcv(
                        actual_ticker, start=start, end=end, interval=interval
                    )
                else:
                    raise

            # Forward-fill missing bars within trading sessions only
            if interval == "1h":
                df = _ffill_hourly(df)

            df.to_parquet(cache_path, index=False)
            results[ticker][interval] = df
            logger.info(
                "  %s %s: %d bars (saved to %s)",
                actual_ticker,
                interval,
                len(df),
                cache_path.name,
            )
            time.sleep(rate_limit_pause)

    return results


def _ffill_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill gaps within regular trading hours (9:30-16:00 ET).

    Only fills gaps that fall within trading hours on trading days.
    Does not create bars outside the existing date range.
    """
    if df.empty:
        return df

    # Ensure UTC timestamps
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    # Convert to Eastern for trading hour detection
    ts_et = df["timestamp"].dt.tz_convert("US/Eastern")

    # Trading hours: 9:30 to 16:00 ET (Massive uses bar open time, so 9:00 to 15:00)
    trading_mask = (ts_et.dt.hour >= 9) & (ts_et.dt.hour <= 15)

    # For commodity ETFs and extended-hours data, keep all rows
    # Only filter if we detect regular equity trading pattern
    if trading_mask.sum() > 0.5 * len(df):
        # Primarily regular-hours data, forward-fill within trading hours
        df = df.set_index("timestamp")
        df = df.ffill(limit=2)  # Fill up to 2 consecutive missing bars
        df = df.reset_index()

    return df


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
