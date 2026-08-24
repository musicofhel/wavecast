"""Tests for forward gap census + catch-up runner."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from wavecast.forward.catchup import CatchupRunner, census_gaps
from wavecast.forward.config import ForwardTestConfig
from wavecast.forward.runner import ForwardTestRunner
from wavecast.tokenizer.vocabulary import SAXVocabulary


def _write_ledger(path: Path, rows: list[dict]) -> Path:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


class TestCensusGaps:
    def test_counts_and_missing_days(self, tmp_path: Path) -> None:
        ledger = _write_ledger(
            tmp_path / "predictions.jsonl",
            [
                {"timestamp": "2026-07-08T21:00:00+00:00", "ticker": "AAPL"},
                {"timestamp": "2026-07-08T21:00:00+00:00", "ticker": "MSFT"},
                # 2026-07-09 (Thu) fully missing
                {"timestamp": "2026-07-10T21:00:00+00:00", "ticker": "AAPL"},
                {"timestamp": "2026-07-11T21:00:00+0000".replace("0000", "00"), "ticker": "AAPL"},
            ],
        )
        c = census_gaps(ledger, tickers=["AAPL", "MSFT"], end_date="2026-07-10")
        assert c.first_day == "2026-07-08"
        assert c.per_day_counts["2026-07-08"] == 2
        assert "2026-07-09" in c.missing_days
        assert c.missing_tickers_by_day == {"2026-07-10": ["MSFT"]}
        assert c.total_missing_pairs == 2 * 1 + 1  # 1 missing day x2 tickers + partial

    def test_read_only_on_existing_ledger(self, tmp_path: Path) -> None:
        ledger = _write_ledger(
            tmp_path / "p.jsonl",
            [{"timestamp": "2026-07-08T21:00:00Z", "ticker": "SPY"}],
        )
        before = ledger.read_text()
        census_gaps(ledger)
        assert ledger.read_text() == before

    def test_weekend_not_counted_missing(self, tmp_path: Path) -> None:
        # 2026-07-11 is a Saturday; a gap there must not be flagged
        ledger = _write_ledger(
            tmp_path / "p.jsonl",
            [
                {"timestamp": "2026-07-10T21:00:00Z", "ticker": "SPY"},
                {"timestamp": "2026-07-13T21:00:00Z", "ticker": "SPY"},
            ],
        )
        c = census_gaps(ledger, tickers=["SPY"])
        assert c.missing_days == []
        assert c.missing_tickers_by_day == {}

    def test_empty_ledger(self, tmp_path: Path) -> None:
        c = census_gaps(tmp_path / "missing.jsonl")
        assert c.first_day is None
        assert c.total_missing_pairs == 0


def _make_prices(n: int = 300, start_day: str = "2026-06-01") -> pd.DataFrame:
    base = datetime.datetime.fromisoformat(start_day + "T09:00:00+00:00")
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


class TestCatchupRunner:
    def _config(self, tmp_path: Path) -> ForwardTestConfig:
        return ForwardTestConfig(
            test_name="loop_scratch_catchup",
            model_path=str(tmp_path / "model"),
            vocab_path=str(tmp_path / "vocab.json"),
            tickers=["AAPL"],
            intervals=["1h"],
            horizons=[1],
            lookback_bars=200,
            log_dir=tmp_path / "scratch_logs",
        )

    def test_run_for_session_logs_bar_timestamp(self, tmp_path: Path) -> None:
        config = self._config(tmp_path)
        words = [[f"{'abcdefg'[i % 7]}" * 4 for i in range(50)] for _ in range(3)]
        vocab = SAXVocabulary.from_corpus(words, min_freq=1, max_size=20)
        model = MagicMock()
        model._config = {"context_length": 16}
        model.input_mode = "tokenized"
        model.task = "token"
        model.predict.return_value = np.array([5], dtype=np.int64)
        model.predict_proba.return_value = np.full((1, 20), 0.05)

        prices = _make_prices(600, start_day="2026-05-20")

        with (
            patch.object(ForwardTestRunner, "_load_model", return_value=model),
            patch.object(ForwardTestRunner, "_load_vocab", return_value=vocab),
            patch(
                "wavecast.forward.catchup.fetch_massive_ohlcv",
                return_value=prices,
            ) as fake_fetch,
        ):
            runner = CatchupRunner(config, pacing_seconds=0)
            n = runner.run_for_session("2026-06-05")

        assert n >= 1
        # Historical window ends the day AFTER the session
        _, kwargs = fake_fetch.call_args
        assert kwargs["end"] == "2026-06-06"

        pred_files = list((tmp_path / "scratch_logs").rglob("*.jsonl"))
        assert pred_files, "prediction file must exist under scratch log dir"
        rows = [json.loads(line) for line in pred_files[0].read_text().splitlines() if line]
        assert all(r["ticker"] == "AAPL" for r in rows)
        # Prediction timestamp must be the BAR timestamp of the session day,
        # not wall-clock now.
        last = rows[-1]
        assert last["timestamp"].startswith("2026-06-")
        assert last["target_timestamp"] > last["timestamp"]

    def test_fetch_failure_skips_ticker(self, tmp_path: Path) -> None:
        from wavecast.core.exceptions import DataError

        config = self._config(tmp_path)
        model = MagicMock()
        model._config = {"context_length": 16}
        model.input_mode = "tokenized"
        model.task = "token"
        vocab = MagicMock()

        with (
            patch.object(ForwardTestRunner, "_load_model", return_value=model),
            patch.object(ForwardTestRunner, "_load_vocab", return_value=vocab),
            patch(
                "wavecast.forward.catchup.fetch_massive_ohlcv",
                side_effect=DataError("429 too many requests"),
            ),
        ):
            runner = CatchupRunner(config, pacing_seconds=0)
            n = runner.run_for_session("2026-06-05")

        assert n == 0
