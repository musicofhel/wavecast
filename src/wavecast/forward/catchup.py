"""Forward-run gap census + historical catch-up.

The census reads a predictions ledger READ-ONLY and reports which
(day, ticker) sessions are missing or partial (the production cron lost
15 of 20 tickers to JPM 429s after 2026-07-10). The CatchupRunner
re-predicts missed sessions from HISTORICAL bars into whatever log_dir
the caller points it at — point it at loop_scratch/, never the
production ledger.
"""

from __future__ import annotations

import datetime
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from wavecast.core.exceptions import DataError, DataNotFoundError
from wavecast.data.sources import fetch_massive_ohlcv
from wavecast.forward.runner import ForwardTestRunner, compute_signal_b
from wavecast.forward.types import ForwardPrediction
from wavecast.signals import ReturnSignalGenerator, SignalGenerator

logger = logging.getLogger(__name__)


@dataclass
class GapCensus:
    """Missing/partial coverage of a forward-test ledger."""

    first_day: str | None = None
    last_day: str | None = None
    per_day_counts: dict[str, int] = field(default_factory=dict)
    missing_days: list[str] = field(default_factory=list)
    # day -> tickers absent from an otherwise-covered day
    missing_tickers_by_day: dict[str, list[str]] = field(default_factory=dict)
    expected_tickers: set[str] = field(default_factory=set)

    @property
    def total_missing_pairs(self) -> int:
        """Missing pairs = fully-missing days x universe size + partial gaps."""
        n_tickers = len(self.expected_tickers)
        return len(self.missing_days) * n_tickers + sum(
            len(v) for v in self.missing_tickers_by_day.values()
        )

    def summary(self) -> str:
        lines = [
            f"ledger span: {self.first_day} -> {self.last_day}",
            f"days covered: {len(self.per_day_counts)}",
            f"fully missing days: {len(self.missing_days)}",
            f"partial days: {len(self.missing_tickers_by_day)}",
            f"missing (day, ticker) pairs: {self.total_missing_pairs}",
        ]
        return "\n".join(lines)


def census_gaps(
    ledger_path: Path,
    tickers: list[str] | None = None,
    end_date: str | None = None,
) -> GapCensus:
    """Compute which (day, ticker) sessions are absent from a ledger.

    Read-only: opens the JSONL for reading and never writes.
    """
    rows: list[dict] = []
    if ledger_path.exists():
        with open(ledger_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    observed: dict[str, set[str]] = {}
    for r in rows:
        day = str(r["timestamp"])[:10]
        observed.setdefault(day, set()).add(r["ticker"])

    census = GapCensus()
    if not observed:
        return census

    days = sorted(observed)
    census.first_day = days[0]
    census.last_day = days[-1]
    census.per_day_counts = {d: len(observed[d]) for d in days}

    expected = set(tickers) if tickers else {
        t for s in observed.values() for t in s
    }
    census.expected_tickers = expected
    if end_date is None:
        end = datetime.date.fromisoformat(census.last_day)
    else:
        end = datetime.date.fromisoformat(end_date)

    d = datetime.date.fromisoformat(census.first_day)
    while d <= end:
        key = d.isoformat()
        present = observed.get(key, set())
        if d.weekday() < 5:  # weekends are not sessions
            if not present:
                census.missing_days.append(key)
            else:
                missing = sorted(expected - present)
                if missing:
                    census.missing_tickers_by_day[key] = missing
        d += datetime.timedelta(days=1)

    return census


class CatchupRunner(ForwardTestRunner):
    """Re-run missed forward sessions from historical bars.

    Inherits model/vocab loading, pipeline context building, pacing and
    magnitude-threshold logic from ForwardTestRunner. Predictions carry
    the BAR timestamp (not wall-clock now) so resolution downstream sees
    the true session time. Log destination comes entirely from
    ``config.log_dir`` / ``config.test_name`` — callers must use a
    scratch directory when backfilling production gaps.
    """

    def run_for_session(
        self, session_end: str | datetime.date
    ) -> int:
        """Backfill one daily session; returns number of predictions logged."""
        if isinstance(session_end, str):
            session_end = datetime.date.fromisoformat(session_end)
        model = self._load_model()
        vocab = self._load_vocab() if model.input_mode != "continuous" else None
        tracker = self._get_tracker()

        is_return_task = model.task in ("return_quantile", "return_regression")
        if is_return_task:
            sig_gen = ReturnSignalGenerator(  # type: ignore[assignment]
                task=model.task,
                n_classes=model._config.get("n_output_classes", 5),
                confidence_threshold=self.config.signal.confidence_threshold,
            )
        else:
            assert vocab is not None
            sig_gen = SignalGenerator(
                vocab_size=vocab.size,
                alphabet_size=self.config.sax.alphabet_size,
                confidence_threshold=self.config.signal.confidence_threshold,
                calibration_method=self.config.signal.calibration_method,
                temperature=self.config.signal.temperature,
            )

        # Historical window: generous buffer so DWT level 5 has enough bars
        # even across weekends/holidays (mirrors fetch_latest_bars' 5x rule).
        start = session_end - datetime.timedelta(days=max(30, self.config.lookback_bars // 5))
        end_exclusive = session_end + datetime.timedelta(days=1)
        end_of_session = pd.Timestamp(session_end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)

        n_logged = 0
        first_ticker = True
        for ticker in self.config.tickers:
            if self.pacing_seconds > 0 and not first_ticker:
                time.sleep(self.pacing_seconds)
            first_ticker = False
            for interval in self.config.intervals:
                try:
                    prices = fetch_massive_ohlcv(
                        ticker=ticker,
                        start=start.isoformat(),
                        end=end_exclusive.isoformat(),
                        interval=interval,
                    )
                except (DataError, DataNotFoundError) as e:
                    logger.warning("Catchup fetch failed for %s %s: %s", ticker, interval, e)
                    continue

                prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
                prices = prices[prices["timestamp"] <= end_of_session].reset_index(drop=True)
                if prices.empty:
                    logger.warning("No bars on/before %s for %s", session_end, ticker)
                    continue

                X_full, timestamps = self._build_pipeline_context(
                    prices, ticker, interval, vocab, model
                )
                if X_full is None or len(X_full) == 0:
                    logger.warning("No context built for %s %s, skipping", ticker, interval)
                    continue

                # Predict on the LAST position (last in-session bar), matching
                # ForwardTestRunner.run_once's alignment convention.
                X_last = X_full[-1:]
                predicted = model.predict(X_last)
                proba = model.predict_proba(X_last)

                magnitude = None
                tercile = None
                softmax_list = None
                if proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
                    bin_midpoints, mag_thresholds = self._load_magnitude_config()
                    if len(bin_midpoints) == proba.shape[1]:
                        magnitude = compute_signal_b(proba[0], bin_midpoints)
                        t33, t67 = mag_thresholds
                        tercile = (
                            "large" if magnitude >= t67
                            else "medium" if magnitude >= t33
                            else "small"
                        )
                        softmax_list = proba[0].tolist()

                pred_ts = pd.Timestamp(timestamps[-1])
                for horizon in self.config.horizons:
                    if is_return_task:
                        series = sig_gen.generate(
                            predicted_classes=predicted,
                            timestamps=[pred_ts],
                            ticker=ticker,
                            horizon=horizon,
                            probabilities=proba if proba.ndim == 2 else None,
                        )
                    else:
                        series = sig_gen.generate(
                            probabilities=proba,
                            predicted_tokens=predicted,
                            timestamps=[pred_ts],
                            ticker=ticker,
                            horizon=horizon,
                        )
                    sig = series.signals[0]
                    target_ts = self._compute_target_timestamp(
                        np.datetime64(pred_ts.to_datetime64()), interval, horizon
                    )
                    pred = ForwardPrediction(
                        id=str(uuid.uuid4()),
                        timestamp=pred_ts.isoformat(),
                        target_timestamp=target_ts,
                        ticker=ticker,
                        interval=interval,
                        horizon=horizon,
                        predicted_direction=sig.direction,
                        predicted_confidence=sig.confidence,
                        predicted_token=sig.token_id,
                        softmax_probs=softmax_list,
                        predicted_magnitude=magnitude,
                        magnitude_tercile=tercile,
                        a2i_trade=(tercile == "large" and sig.direction != 0),
                    )
                    tracker.log_prediction(pred)
                    n_logged += 1
                    logger.info(
                        "Catchup prediction: %s %s @%s h=%d dir=%+d conf=%.3f",
                        ticker, interval, pred_ts.isoformat(), horizon,
                        sig.direction, sig.confidence,
                    )

        return n_logged
