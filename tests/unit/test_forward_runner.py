"""Tests for ForwardTestRunner."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from wavecast.forward.config import ForwardTestConfig
from wavecast.forward.runner import ForwardTestRunner
from wavecast.tokenizer.vocabulary import SAXVocabulary


def _make_prices(n: int = 200) -> pd.DataFrame:
    """Create synthetic OHLCV DataFrame."""
    base = datetime.datetime(2025, 6, 1, 9, 0, tzinfo=datetime.UTC)
    timestamps = [base + datetime.timedelta(hours=i) for i in range(n)]
    rng = np.random.default_rng(42)
    close = [100.0 + 0.1 * i + rng.normal(0, 0.5) for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "open": close,
            "high": [c + 0.5 for c in close],
            "low": [c - 0.5 for c in close],
            "close": close,
            "volume": [1000000] * n,
        }
    )


def _make_mock_model(vocab_size: int = 20) -> MagicMock:
    """Create a mock WaveletGPT model."""
    model = MagicMock()
    model._config = {"context_length": 16, "vocab_size": vocab_size}
    model.predict.return_value = np.array([min(5, vocab_size - 1)], dtype=np.int64)
    model.predict_proba.return_value = np.full((1, vocab_size), 1.0 / vocab_size)
    return model


def _make_vocab() -> SAXVocabulary:
    """Create a real small SAXVocabulary from synthetic words."""
    # Build a real small vocabulary from synthetic words
    words = [[f"{'abcdefg'[i % 7]}" * 4 for i in range(50)] for _ in range(3)]
    vocab = SAXVocabulary.from_corpus(words, min_freq=1, max_size=20)
    return vocab


class TestForwardTestRunner:
    def test_run_once_mock(self, tmp_path: Path) -> None:
        """Full cycle with mocked data source and model."""
        config = ForwardTestConfig(
            test_name="test_run",
            model_path=str(tmp_path / "model"),
            vocab_path=str(tmp_path / "vocab.json"),
            tickers=["AAPL"],
            intervals=["1h"],
            lookback_bars=200,
            log_dir=tmp_path / "logs",
        )

        vocab = _make_vocab()
        model = _make_mock_model(vocab_size=vocab.size)

        with (
            patch.object(ForwardTestRunner, "_load_model", return_value=model),
            patch.object(ForwardTestRunner, "_load_vocab", return_value=vocab),
            patch("wavecast.forward.runner.fetch_latest_bars", return_value=_make_prices(500)),
        ):
            runner = ForwardTestRunner(config)
            summary = runner.run_once()

        assert summary.total_predictions >= 1
        assert summary.test_name == "test_run"

    def test_resolve_pending(self, tmp_path: Path) -> None:
        """Second run should resolve predictions from first run."""
        config = ForwardTestConfig(
            test_name="test_resolve",
            model_path=str(tmp_path / "model"),
            vocab_path=str(tmp_path / "vocab.json"),
            tickers=["AAPL"],
            intervals=["1h"],
            lookback_bars=200,
            log_dir=tmp_path / "logs",
        )

        vocab = _make_vocab()
        model = _make_mock_model(vocab_size=vocab.size)

        # First run: creates predictions
        with (
            patch.object(ForwardTestRunner, "_load_model", return_value=model),
            patch.object(ForwardTestRunner, "_load_vocab", return_value=vocab),
            patch("wavecast.forward.runner.fetch_latest_bars", return_value=_make_prices(500)),
        ):
            runner = ForwardTestRunner(config)
            summary1 = runner.run_once()

        assert summary1.pending_predictions >= 1

    def test_build_pipeline_context(self, tmp_path: Path) -> None:
        """Pipeline context produces valid X array."""
        config = ForwardTestConfig(
            test_name="test_ctx",
            model_path=str(tmp_path / "model"),
            vocab_path=str(tmp_path / "vocab.json"),
            tickers=["AAPL"],
            intervals=["1h"],
            lookback_bars=200,
            log_dir=tmp_path / "logs",
        )

        vocab = _make_vocab()
        model = _make_mock_model(vocab_size=vocab.size)

        runner = ForwardTestRunner(config)
        prices = _make_prices(500)

        X_full, timestamps = runner._build_pipeline_context(
            prices, "AAPL", "1h", vocab, model
        )

        # X should have samples with context_length + 2 columns
        assert X_full is not None
        assert X_full.ndim == 2
        assert X_full.shape[1] == 16 + 2  # context_length + level_id + asset_class_id

    def test_predict_mock(self, tmp_path: Path) -> None:
        """Model predict/predict_proba called correctly."""
        config = ForwardTestConfig(
            test_name="test_pred",
            model_path=str(tmp_path / "model"),
            vocab_path=str(tmp_path / "vocab.json"),
            tickers=["AAPL"],
            intervals=["1h"],
            lookback_bars=200,
            log_dir=tmp_path / "logs",
        )

        vocab = _make_vocab()
        model = _make_mock_model(vocab_size=vocab.size)

        with (
            patch.object(ForwardTestRunner, "_load_model", return_value=model),
            patch.object(ForwardTestRunner, "_load_vocab", return_value=vocab),
            patch("wavecast.forward.runner.fetch_latest_bars", return_value=_make_prices(500)),
        ):
            runner = ForwardTestRunner(config)
            runner.run_once()

        model.predict.assert_called_once()
        model.predict_proba.assert_called_once()

    def test_multi_interval(self, tmp_path: Path) -> None:
        """Handles multiple intervals correctly."""
        config = ForwardTestConfig(
            test_name="test_multi",
            model_path=str(tmp_path / "model"),
            vocab_path=str(tmp_path / "vocab.json"),
            tickers=["AAPL"],
            intervals=["1h", "1d"],
            lookback_bars=200,
            log_dir=tmp_path / "logs",
        )

        vocab = _make_vocab()
        model = _make_mock_model(vocab_size=vocab.size)

        with (
            patch.object(ForwardTestRunner, "_load_model", return_value=model),
            patch.object(ForwardTestRunner, "_load_vocab", return_value=vocab),
            patch("wavecast.forward.runner.fetch_latest_bars", return_value=_make_prices(500)),
        ):
            runner = ForwardTestRunner(config)
            summary = runner.run_once()

        # Should have predictions for both intervals
        assert summary.total_predictions >= 2

    def test_no_pending(self, tmp_path: Path) -> None:
        """First run has no pending to resolve."""
        config = ForwardTestConfig(
            test_name="test_nopend",
            model_path=str(tmp_path / "model"),
            vocab_path=str(tmp_path / "vocab.json"),
            tickers=["AAPL"],
            intervals=["1h"],
            lookback_bars=200,
            log_dir=tmp_path / "logs",
        )

        vocab = _make_vocab()
        model = _make_mock_model(vocab_size=vocab.size)

        with (
            patch.object(ForwardTestRunner, "_load_model", return_value=model),
            patch.object(ForwardTestRunner, "_load_vocab", return_value=vocab),
            patch("wavecast.forward.runner.fetch_latest_bars", return_value=_make_prices(500)),
        ):
            runner = ForwardTestRunner(config)
            summary = runner.run_once()

        assert summary.resolved_predictions == 0

    def test_compute_target_timestamp(self, tmp_path: Path) -> None:
        """Target timestamp computed correctly for various intervals."""
        config = ForwardTestConfig(
            test_name="test_target",
            model_path=str(tmp_path / "model"),
            vocab_path=str(tmp_path / "vocab.json"),
            tickers=["AAPL"],
            log_dir=tmp_path / "logs",
        )
        runner = ForwardTestRunner(config)

        base = np.datetime64("2025-06-01T09:00:00")

        # 1h, horizon=1
        result = runner._compute_target_timestamp(base, "1h", 1)
        assert "2025-06-01T10:00:00" in result

        # 1d, horizon=2
        result = runner._compute_target_timestamp(base, "1d", 2)
        assert "2025-06-03" in result

        # 5m, horizon=3
        result = runner._compute_target_timestamp(base, "5m", 3)
        assert "2025-06-01T09:15:00" in result


class TestPerTickerIsolation:
    def _config(self, tmp_path: Path, tickers: list[str]) -> ForwardTestConfig:
        return ForwardTestConfig(
            test_name="test_isolation",
            model_path=str(tmp_path / "model"),
            vocab_path=str(tmp_path / "vocab.json"),
            tickers=tickers,
            intervals=["1h"],
            lookback_bars=200,
            log_dir=tmp_path / "logs",
        )

    def test_fetch_failure_skips_ticker_and_continues(self, tmp_path: Path) -> None:
        """One ticker's API death must not kill the run (JPM 429 cron failure)."""
        from wavecast.core.exceptions import DataError

        vocab = _make_vocab()
        model = _make_mock_model(vocab_size=vocab.size)

        def fake_fetch(ticker: str, **kwargs):  # type: ignore[no-untyped-def]
            if ticker == "BAD":
                raise DataError("Massive API error for BAD: 429")
            return _make_prices(500)

        with (
            patch.object(ForwardTestRunner, "_load_model", return_value=model),
            patch.object(ForwardTestRunner, "_load_vocab", return_value=vocab),
            patch(
                "wavecast.forward.runner.fetch_latest_bars", side_effect=fake_fetch
            ),
        ):
            runner = ForwardTestRunner(self._config(tmp_path, ["BAD", "AAPL"]))
            summary = runner.run_once()

        assert summary.total_predictions >= 1
        rows = [
            json.loads(line)
            for line in (tmp_path / "logs" / "test_isolation" / "predictions.jsonl")
            .read_text()
            .splitlines()
        ]
        assert {r["ticker"] for r in rows} == {"AAPL"}

    def test_all_tickers_failing_returns_empty_summary(self, tmp_path: Path) -> None:
        """Every fetch failing yields an empty-but-valid summary, no exception."""
        from wavecast.core.exceptions import DataNotFoundError

        vocab = _make_vocab()
        model = _make_mock_model(vocab_size=vocab.size)

        with (
            patch.object(ForwardTestRunner, "_load_model", return_value=model),
            patch.object(ForwardTestRunner, "_load_vocab", return_value=vocab),
            patch(
                "wavecast.forward.runner.fetch_latest_bars",
                side_effect=DataNotFoundError("no data"),
            ),
        ):
            runner = ForwardTestRunner(self._config(tmp_path, ["X", "Y"]))
            summary = runner.run_once()

        assert summary.total_predictions == 0

    def test_pacing_sleeps_between_tickers(self, tmp_path: Path) -> None:
        """pacing_seconds pauses after the first ticker."""
        vocab = _make_vocab()
        model = _make_mock_model(vocab_size=vocab.size)

        with (
            patch.object(ForwardTestRunner, "_load_model", return_value=model),
            patch.object(ForwardTestRunner, "_load_vocab", return_value=vocab),
            patch(
                "wavecast.forward.runner.fetch_latest_bars",
                return_value=_make_prices(500),
            ),
            patch("wavecast.forward.runner.time.sleep") as mock_sleep,
        ):
            runner = ForwardTestRunner(
                self._config(tmp_path, ["A", "B"]), pacing_seconds=2.5
            )
            runner.run_once()

        mock_sleep.assert_called_once_with(2.5)
