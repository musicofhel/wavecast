"""Tests for 429-aware retry in Massive data sources."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from wavecast.core.exceptions import DataError
from wavecast.data.sources import (
    _BACKOFF_BASE_SECONDS,
    _MAX_RETRIES,
    _fetch_aggs_with_retry,
    _is_rate_limit_error,
    _retry_after_seconds,
    fetch_massive_ohlcv,
)


def _agg(ts: int = 1700000000000) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=ts, open=1.0, high=1.1, low=0.9, close=1.05, volume=1000
    )


class TestRateLimitDetection:
    def test_status_code_attribute(self) -> None:
        exc = Exception("err")
        exc.status_code = 429
        assert _is_rate_limit_error(exc)

    def test_message_match(self) -> None:
        assert _is_rate_limit_error(Exception("HTTP 429 Too Many Requests"))
        assert _is_rate_limit_error(Exception("rate limit exceeded"))
        assert not _is_rate_limit_error(Exception("404 Not Found"))

    def test_retry_after_header(self) -> None:
        exc = Exception("429")
        exc.headers = {"Retry-After": "7"}
        assert _retry_after_seconds(exc) == 7.0

    def test_retry_after_missing(self) -> None:
        assert _retry_after_seconds(Exception("429")) is None
        exc = Exception("429")
        exc.headers = {"Retry-After": "soon"}
        assert _retry_after_seconds(exc) is None


class TestFetchAggsWithRetry:
    def test_succeeds_after_429s(self) -> None:
        client = MagicMock()
        responses = [Exception("HTTP 429"), Exception("429 too many requests")]
        client.list_aggs.side_effect = [
            *responses,
            iter([_agg()]),
        ]
        with (
            patch("wavecast.data.sources.time.sleep") as mock_sleep,
            patch(
                "wavecast.data.sources._retry_after_seconds", return_value=None
            ),
        ):
            aggs = _fetch_aggs_with_retry(client, "AAPL", 1, "hour", "2026-01-01", "2026-02-01")

        assert len(aggs) == 1
        # exponential backoff: base * 2**attempt
        expected_first = _BACKOFF_BASE_SECONDS
        assert mock_sleep.call_args_list[0][0][0] == pytest.approx(expected_first)
        assert client.list_aggs.call_count == 3

    def test_honors_retry_after(self) -> None:
        client = MagicMock()
        err = Exception("429")
        err.headers = {"Retry-After": "13"}
        client.list_aggs.side_effect = [err, iter([_agg()])]
        with patch("wavecast.data.sources.time.sleep") as mock_sleep:
            aggs = _fetch_aggs_with_retry(client, "AAPL", 1, "hour", "2026-01-01", "2026-02-01")
        assert len(aggs) == 1
        assert mock_sleep.call_args_list[0][0][0] == pytest.approx(13.0)

    def test_raises_data_error_on_persistent_429(self) -> None:
        client = MagicMock()
        client.list_aggs.side_effect = [Exception("429")] * _MAX_RETRIES
        with (
            patch("wavecast.data.sources.time.sleep"),
            pytest.raises(DataError, match="AAPL"),
        ):
            _fetch_aggs_with_retry(client, "AAPL", 1, "hour", "2026-01-01", "2026-02-01")
        assert client.list_aggs.call_count == _MAX_RETRIES

    def test_non_429_error_not_retried(self) -> None:
        client = MagicMock()
        client.list_aggs.side_effect = Exception("500 server error")
        with (
            patch("wavecast.data.sources.time.sleep") as mock_sleep,
            pytest.raises(DataError, match="500"),
        ):
            _fetch_aggs_with_retry(client, "AAPL", 1, "hour", "2026-01-01", "2026-02-01")
        assert client.list_aggs.call_count == 1
        mock_sleep.assert_not_called()


class TestFetchOhlcvRetryIntegration:
    def test_fetch_massive_ohlcv_recovers_from_429(self) -> None:
        client = MagicMock()
        client.list_aggs.side_effect = [Exception("HTTP 429"), iter([_agg()])]
        with (
            patch("wavecast.data.sources._get_massive_client", return_value=client),
            patch("wavecast.data.sources.time.sleep"),
            patch("wavecast.data.sources._retry_after_seconds", return_value=None),
        ):
            df = fetch_massive_ohlcv("AAPL", start="2026-01-01", end="2026-02-01", interval="1h")
        assert len(df) == 1
        assert list(df.columns) == [
            "timestamp", "open", "high", "low", "close", "volume",
        ]
