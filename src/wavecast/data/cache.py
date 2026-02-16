"""Parquet-based cache for fetched time series data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wavecast.core.types import TimeSeries


class ParquetCache:
    """Disk-backed cache storing time series as Parquet files.

    Files are stored as {cache_dir}/{ticker}_{interval}.parquet with columns
    'timestamp' and 'value'.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, ticker: str, interval: str = "1d") -> Path:
        safe_ticker = ticker.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_ticker}_{interval}.parquet"

    def get(
        self, ticker: str, interval: str = "1d", column: str = "close"
    ) -> TimeSeries | None:
        """Retrieve a cached time series, or None if not cached.

        Args:
            ticker: Ticker symbol.
            interval: Data interval.
            column: Column label to assign.

        Returns:
            TimeSeries if cached, else None.
        """
        path = self._key_path(ticker, interval)
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
        except Exception:
            return None

        if "timestamp" not in df.columns or "value" not in df.columns:
            return None

        return TimeSeries(
            values=df["value"].to_numpy(dtype=np.float64),
            timestamps=df["timestamp"].to_numpy(dtype="datetime64[ns]"),
            ticker=ticker,
            interval=interval,
            column=column,
        )

    def put(self, ts: TimeSeries) -> Path:
        """Store a time series to the cache.

        Args:
            ts: TimeSeries to cache.

        Returns:
            Path to the written Parquet file.
        """
        path = self._key_path(ts.ticker, ts.interval)
        df = pd.DataFrame(
            {
                "timestamp": pd.DatetimeIndex(ts.timestamps),
                "value": ts.values,
            }
        )
        df.to_parquet(path, index=False)
        return path

    def list_cached(self) -> list[tuple[str, str]]:
        """List all cached (ticker, interval) pairs.

        Returns:
            List of (ticker, interval) tuples.
        """
        results: list[tuple[str, str]] = []
        for f in sorted(self.cache_dir.glob("*.parquet")):
            stem = f.stem
            # Split on last underscore: ticker may contain underscores
            parts = stem.rsplit("_", 1)
            if len(parts) == 2:
                results.append((parts[0], parts[1]))
        return results

    def clear(self) -> int:
        """Remove all cached Parquet files.

        Returns:
            Number of files deleted.
        """
        count = 0
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
            count += 1
        return count
