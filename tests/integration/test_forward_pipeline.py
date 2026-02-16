"""Integration tests for forward testing pipeline."""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from wavecast.forward.config import ForwardTestConfig
from wavecast.forward.runner import ForwardTestRunner
from wavecast.forward.tracker import ForwardTestTracker
from wavecast.tokenizer.vocabulary import SAXVocabulary


def _make_prices(n: int = 500, base_price: float = 100.0) -> pd.DataFrame:
    """Create synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    base_ts = datetime.datetime(2025, 6, 1, 9, 0, tzinfo=datetime.UTC)
    timestamps = [base_ts + datetime.timedelta(hours=i) for i in range(n)]
    close = base_price + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": [1000000] * n,
        }
    )


def _make_vocab() -> SAXVocabulary:
    """Build a real small vocabulary from synthetic SAX words."""
    words = []
    for _ in range(5):
        seq = [f"{'abcdefg'[i % 7]}" * 4 for i in range(30)]
        words.append(seq)
    return SAXVocabulary.from_corpus(words, min_freq=1, max_size=20)


def _make_mock_model(vocab_size: int) -> MagicMock:
    model = MagicMock()
    model._config = {"context_length": 16, "vocab_size": vocab_size}
    model.predict.return_value = np.array([vocab_size // 2], dtype=np.int64)
    # Return uniform probabilities
    proba = np.full((1, vocab_size), 1.0 / vocab_size)
    model.predict_proba.return_value = proba
    return model


class TestForwardPipelineIntegration:
    def test_end_to_end_synthetic(self, tmp_path: Path) -> None:
        """Full forward test cycle with mock API and synthetic model."""
        vocab = _make_vocab()
        model = _make_mock_model(vocab.size)
        prices = _make_prices()

        config = ForwardTestConfig(
            test_name="e2e_test",
            model_path=str(tmp_path / "model"),
            vocab_path=str(tmp_path / "vocab.json"),
            tickers=["AAPL"],
            intervals=["1h"],
            lookback_bars=500,
            log_dir=tmp_path / "logs",
        )

        with (
            patch.object(ForwardTestRunner, "_load_model", return_value=model),
            patch.object(ForwardTestRunner, "_load_vocab", return_value=vocab),
            patch("wavecast.forward.runner.fetch_latest_bars", return_value=prices),
        ):
            runner = ForwardTestRunner(config)
            summary = runner.run_once()

        assert summary.test_name == "e2e_test"
        assert summary.total_predictions >= 1
        assert summary.pending_predictions >= 1
        assert summary.resolved_predictions == 0  # First run, nothing to resolve

        # Verify JSONL was written
        pred_file = tmp_path / "logs" / "e2e_test" / "predictions.jsonl"
        assert pred_file.exists()
        lines = pred_file.read_text().strip().split("\n")
        assert len(lines) >= 1

    def test_full_resolve_cycle(self, tmp_path: Path) -> None:
        """Run twice: first creates predictions, second resolves them."""
        vocab = _make_vocab()
        model = _make_mock_model(vocab.size)

        config = ForwardTestConfig(
            test_name="resolve_test",
            model_path=str(tmp_path / "model"),
            vocab_path=str(tmp_path / "vocab.json"),
            tickers=["AAPL"],
            intervals=["1h"],
            horizons=[1],
            lookback_bars=500,
            log_dir=tmp_path / "logs",
        )

        # First run
        with (
            patch.object(ForwardTestRunner, "_load_model", return_value=model),
            patch.object(ForwardTestRunner, "_load_vocab", return_value=vocab),
            patch("wavecast.forward.runner.fetch_latest_bars", return_value=_make_prices()),
        ):
            runner1 = ForwardTestRunner(config)
            summary1 = runner1.run_once()

        assert summary1.pending_predictions >= 1

        # Manually resolve by modifying predictions to have past target timestamps
        tracker = ForwardTestTracker(
            log_dir=tmp_path / "logs",
            test_name="resolve_test",
        )
        for pred in tracker._predictions:
            # Set target to 2 hours ago so it can be resolved
            pred.target_timestamp = (
                datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=2)
            ).isoformat()
        tracker._save_all()

        # Second run with price data that covers the target timestamps
        prices2 = _make_prices(600, base_price=102.0)
        # Extend timestamps to cover current time
        now = datetime.datetime.now(datetime.UTC)
        timestamps = [now - datetime.timedelta(hours=600 - i) for i in range(600)]
        prices2["timestamp"] = pd.to_datetime(timestamps, utc=True)

        with (
            patch.object(ForwardTestRunner, "_load_model", return_value=model),
            patch.object(ForwardTestRunner, "_load_vocab", return_value=vocab),
            patch("wavecast.forward.runner.fetch_latest_bars", return_value=prices2),
        ):
            runner2 = ForwardTestRunner(config)
            summary2 = runner2.run_once()

        # Should have resolved some predictions from first run
        assert summary2.resolved_predictions >= 1

    def test_multi_ticker_cycle(self, tmp_path: Path) -> None:
        """Multiple tickers in a single run."""
        vocab = _make_vocab()
        model = _make_mock_model(vocab.size)

        config = ForwardTestConfig(
            test_name="multi_ticker",
            model_path=str(tmp_path / "model"),
            vocab_path=str(tmp_path / "vocab.json"),
            tickers=["AAPL", "MSFT", "GOOG"],
            intervals=["1h"],
            lookback_bars=500,
            log_dir=tmp_path / "logs",
        )

        with (
            patch.object(ForwardTestRunner, "_load_model", return_value=model),
            patch.object(ForwardTestRunner, "_load_vocab", return_value=vocab),
            patch("wavecast.forward.runner.fetch_latest_bars", return_value=_make_prices()),
        ):
            runner = ForwardTestRunner(config)
            summary = runner.run_once()

        # Should have predictions for all 3 tickers
        assert summary.total_predictions >= 3
        assert set(summary.tickers) == {"AAPL", "GOOG", "MSFT"}
