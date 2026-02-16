"""Precomputed DWT decomposition cache for exploration/visualization.

NOT used by experiments — experiments decompose per-split internally.
This cache is for interactive exploration and visualization only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wavecast.core.types import TimeSeries, WaveletDecomposition
from wavecast.wavelets.dwt import decompose

logger = logging.getLogger(__name__)


class DecompositionCache:
    """Disk-backed cache for precomputed DWT decompositions.

    Stores decomposition coefficients as .npz files:
        {cache_dir}/{ticker}_{interval}_{wavelet}_L{level}.npz
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = Path.home() / ".wavecast" / "cache" / "decompositions"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(
        self, ticker: str, interval: str, wavelet: str, level: int
    ) -> Path:
        safe_ticker = ticker.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_ticker}_{interval}_{wavelet}_L{level}.npz"

    def get(
        self,
        ticker: str,
        interval: str = "1h",
        wavelet: str = "db4",
        level: int = 5,
    ) -> WaveletDecomposition | None:
        """Retrieve a cached decomposition, or None if not cached."""
        path = self._key_path(ticker, interval, wavelet, level)
        if not path.exists():
            return None

        try:
            data = np.load(path, allow_pickle=False)
            coefficients = [data[f"level_{i}"] for i in range(level + 1)]
            original_length = int(data["original_length"])
            return WaveletDecomposition(
                coefficients=coefficients,
                wavelet=wavelet,
                level=level,
                original_length=original_length,
                ticker=ticker,
            )
        except Exception:
            logger.warning("Corrupt cache file: %s", path)
            return None

    def put(self, decomp: WaveletDecomposition, interval: str = "1h") -> Path:
        """Store a decomposition to the cache."""
        path = self._key_path(
            decomp.ticker, interval, decomp.wavelet, decomp.level
        )
        save_dict = {
            f"level_{i}": c for i, c in enumerate(decomp.coefficients)
        }
        save_dict["original_length"] = np.array(decomp.original_length)
        np.savez_compressed(path, **save_dict)
        return path

    def decompose_and_cache(
        self,
        ts: TimeSeries,
        wavelet: str = "db4",
        level: int = 5,
    ) -> WaveletDecomposition:
        """Decompose a time series and cache the result.

        Returns cached version if available.
        """
        cached = self.get(ts.ticker, ts.interval, wavelet, level)
        if cached is not None:
            return cached

        decomp = decompose(ts, wavelet=wavelet, level=level)
        self.put(decomp, interval=ts.interval)
        return decomp

    def list_cached(self) -> list[tuple[str, str, str, int]]:
        """List all cached (ticker, interval, wavelet, level) tuples."""
        results: list[tuple[str, str, str, int]] = []
        for f in sorted(self.cache_dir.glob("*.npz")):
            stem = f.stem
            parts = stem.rsplit("_", 2)
            if len(parts) >= 3 and parts[-1].startswith("L"):
                level = int(parts[-1][1:])
                wavelet = parts[-2]
                # Remaining is ticker_interval
                rest = "_".join(parts[:-2])
                rest_parts = rest.rsplit("_", 1)
                if len(rest_parts) == 2:
                    results.append(
                        (rest_parts[0], rest_parts[1], wavelet, level)
                    )
        return results


def precompute_decompositions(
    tickers: list[str],
    ohlcv_cache_dir: Path | None = None,
    decomp_cache_dir: Path | None = None,
    intervals: list[str] | None = None,
    wavelet: str = "db4",
    level: int = 5,
) -> dict[str, dict[str, WaveletDecomposition]]:
    """Precompute DWT decompositions for all assets in the universe.

    Reads close prices from the OHLCV Parquet cache and decomposes each.

    Args:
        tickers: List of ticker symbols.
        ohlcv_cache_dir: Where OHLCV Parquets live (default: ~/.wavecast/cache).
        decomp_cache_dir: Where to store decompositions.
        intervals: Intervals to decompose (default: ['1h']).
        wavelet: Wavelet family.
        level: Decomposition depth.

    Returns:
        Nested dict: {ticker: {interval: WaveletDecomposition}}.
    """
    if ohlcv_cache_dir is None:
        ohlcv_cache_dir = Path.home() / ".wavecast" / "cache"
    if intervals is None:
        intervals = ["1h"]

    cache = DecompositionCache(decomp_cache_dir)
    results: dict[str, dict[str, WaveletDecomposition]] = {}

    for ticker in tickers:
        results[ticker] = {}
        for interval in intervals:
            # Check decomposition cache first
            cached = cache.get(ticker, interval, wavelet, level)
            if cached is not None:
                logger.info("Decomposition cache hit: %s %s", ticker, interval)
                results[ticker][interval] = cached
                continue

            # Load OHLCV from parquet
            ohlcv_path = ohlcv_cache_dir / f"{ticker}_{interval}_ohlcv.parquet"
            if not ohlcv_path.exists():
                logger.warning("No OHLCV data for %s %s, skipping", ticker, interval)
                continue

            df = pd.read_parquet(ohlcv_path)
            close_prices = df["close"].to_numpy(dtype=np.float64)
            timestamps = pd.to_datetime(df["timestamp"]).to_numpy(
                dtype="datetime64[ns]"
            )

            ts = TimeSeries(
                values=close_prices,
                timestamps=timestamps,
                ticker=ticker,
                interval=interval,
                column="close",
            )

            decomp = cache.decompose_and_cache(ts, wavelet=wavelet, level=level)
            results[ticker][interval] = decomp
            logger.info(
                "Decomposed %s %s: %d points -> %d levels",
                ticker,
                interval,
                len(close_prices),
                level,
            )

    return results
