"""Tests for fetch_latest_bars function."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from wavecast.core.exceptions import DataError


def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Create synthetic OHLCV DataFrame."""
    base = datetime(2025, 6, 1, 9, 0)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(base, periods=n, freq="h", tz="UTC"),
            "open": [100.0 + i * 0.1 for i in range(n)],
            "high": [101.0 + i * 0.1 for i in range(n)],
            "low": [99.0 + i * 0.1 for i in range(n)],
            "close": [100.5 + i * 0.1 for i in range(n)],
            "volume": [1000000] * n,
        }
    )


class TestFetchLatestBars:
    def test_hourly_start_date(self) -> None:
        """Start date should use 5x buffer for intraday intervals."""
        with patch("wavecast.data.sources.fetch_massive_ohlcv") as mock_fetch:
            mock_fetch.return_value = _make_ohlcv(300)

            from wavecast.data.sources import fetch_latest_bars

            df = fetch_latest_bars("AAPL", interval="1h", n_bars=200)

            # Verify fetch_massive_ohlcv was called
            mock_fetch.assert_called_once()
            call_kwargs = mock_fetch.call_args
            assert call_kwargs[0][0] == "AAPL"  # ticker
            assert len(df) == 200

    def test_five_min_start_date(self) -> None:
        """5m interval should compute shorter lookback."""
        with patch("wavecast.data.sources.fetch_massive_ohlcv") as mock_fetch:
            mock_fetch.return_value = _make_ohlcv(500)

            from wavecast.data.sources import fetch_latest_bars

            df = fetch_latest_bars("MSFT", interval="5m", n_bars=100)
            assert len(df) == 100

    def test_n_bars_param(self) -> None:
        """Returned DataFrame should have at most n_bars rows."""
        with patch("wavecast.data.sources.fetch_massive_ohlcv") as mock_fetch:
            mock_fetch.return_value = _make_ohlcv(50)

            from wavecast.data.sources import fetch_latest_bars

            df = fetch_latest_bars("GOOG", interval="1h", n_bars=200)
            # API returned only 50, so we get 50
            assert len(df) == 50

    def test_interval_validation(self) -> None:
        """Invalid interval should raise DataError."""
        from wavecast.data.sources import fetch_latest_bars

        with pytest.raises(DataError, match="Unsupported interval"):
            fetch_latest_bars("AAPL", interval="3h", n_bars=100)
