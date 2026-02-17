"""Forward test runner: orchestrates prediction loop on live/recent data."""

from __future__ import annotations

import datetime
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from wavecast.core.types import MultiLevelTokenSequence, TimeSeries, TokenSequence
from wavecast.data.auxiliary_features import compute_detail_auxiliary_features
from wavecast.data.continuous_dataset import build_continuous_dataset
from wavecast.data.sources import fetch_latest_bars
from wavecast.forward.config import ForwardTestConfig
from wavecast.forward.tracker import ForwardTestTracker
from wavecast.forward.types import ForwardPrediction, ForwardTestSummary
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.sax.bow import extract_words
from wavecast.sax.sax import sax_transform
from wavecast.signals import ReturnSignalGenerator, SignalGenerator
from wavecast.tokenizer.dataset import build_sequence_dataset
from wavecast.tokenizer.vocabulary import SAXVocabulary
from wavecast.wavelets.dwt import decompose

logger = logging.getLogger(__name__)

# Reuse sector/asset class mappings from experiments.runner
SECTOR_ID_MAP: dict[str, int] = {
    "tech": 0,
    "finance": 1,
    "energy": 2,
    "healthcare": 3,
    "broad_etf": 4,
    "commodity_etf": 5,
}

ASSET_CLASS_ID_MAP: dict[str, int] = {
    "equity": 0,
    "crypto": 1,
    "forex": 2,
    "commodity": 3,
}

# Interval -> timedelta mapping for target timestamp computation
_INTERVAL_DELTAS: dict[str, datetime.timedelta] = {
    "1m": datetime.timedelta(minutes=1),
    "5m": datetime.timedelta(minutes=5),
    "15m": datetime.timedelta(minutes=15),
    "30m": datetime.timedelta(minutes=30),
    "1h": datetime.timedelta(hours=1),
    "1d": datetime.timedelta(days=1),
    "1w": datetime.timedelta(weeks=1),
}


class ForwardTestRunner:
    """Orchestrates forward test cycles: fetch -> resolve -> predict -> log.

    Each call to run_once():
    1. Loads model + vocabulary (cached after first load)
    2. For each ticker x interval:
       a. Fetches latest bars
       b. Resolves pending predictions against actual prices
       c. Builds pipeline context (DWT -> SAX -> tokenize)
       d. Runs model.predict_proba() on latest context
       e. Generates signal via SignalGenerator
       f. Logs new prediction
    3. Returns summary
    """

    def __init__(self, config: ForwardTestConfig) -> None:
        self.config = config
        self._model: WaveletGPT | None = None
        self._vocab: SAXVocabulary | None = None
        self._tracker: ForwardTestTracker | None = None

    def run_once(self) -> ForwardTestSummary:
        """Run a single forward test cycle."""
        model = self._load_model()
        # Continuous models don't need a SAX vocabulary
        vocab = self._load_vocab() if model.input_mode != "continuous" else None
        tracker = self._get_tracker()

        # Choose signal generator based on model task
        model_task = model.task
        is_return_task = model_task in ("return_quantile", "return_regression")

        if is_return_task:
            n_classes = model._config.get("n_output_classes", 5)
            return_signal_gen = ReturnSignalGenerator(
                task=model_task,
                n_classes=n_classes,
                confidence_threshold=self.config.signal.confidence_threshold,
            )
        else:
            signal_gen = SignalGenerator(
                vocab_size=vocab.size,
                alphabet_size=self.config.sax.alphabet_size,
                confidence_threshold=self.config.signal.confidence_threshold,
                calibration_method=self.config.signal.calibration_method,
                temperature=self.config.signal.temperature,
            )

        for ticker in self.config.tickers:
            for interval in self.config.intervals:
                logger.info("Processing %s %s", ticker, interval)

                # Fetch latest bars
                prices = fetch_latest_bars(
                    ticker=ticker,
                    interval=interval,
                    n_bars=self.config.lookback_bars,
                )

                # Resolve pending predictions
                n_resolved = tracker.resolve_pending(ticker, interval, prices)
                if n_resolved > 0:
                    logger.info(
                        "Resolved %d predictions for %s %s",
                        n_resolved,
                        ticker,
                        interval,
                    )

                # Build pipeline context
                X_full, timestamps = self._build_pipeline_context(
                    prices, ticker, interval, vocab, model
                )

                if X_full is None or len(X_full) == 0:
                    logger.warning(
                        "No context built for %s %s, skipping", ticker, interval
                    )
                    continue

                # Predict on the LAST position (latest bar)
                X_last = X_full[-1:]

                predicted = model.predict(X_last)
                proba = model.predict_proba(X_last)

                for horizon in self.config.horizons:
                    # Generate signal based on model task
                    if is_return_task:
                        signal_series = return_signal_gen.generate(
                            predicted_classes=predicted,
                            timestamps=timestamps[-1:],
                            ticker=ticker,
                            horizon=horizon,
                            probabilities=proba if proba.ndim == 2 else None,
                        )
                    else:
                        signal_series = signal_gen.generate(
                            probabilities=proba,
                            predicted_tokens=predicted,
                            timestamps=timestamps[-1:],
                            ticker=ticker,
                            horizon=horizon,
                        )

                    sig = signal_series.signals[0]

                    # Compute target timestamp
                    target_ts = self._compute_target_timestamp(
                        timestamps[-1], interval, horizon
                    )

                    pred = ForwardPrediction(
                        id=str(uuid.uuid4()),
                        timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
                        target_timestamp=target_ts,
                        ticker=ticker,
                        interval=interval,
                        horizon=horizon,
                        predicted_direction=sig.direction,
                        predicted_confidence=sig.confidence,
                        predicted_token=sig.token_id,
                    )

                    tracker.log_prediction(pred)
                    logger.info(
                        "Prediction: %s %s h=%d dir=%+d conf=%.3f",
                        ticker,
                        interval,
                        horizon,
                        sig.direction,
                        sig.confidence,
                    )

        return tracker.get_summary()

    def _build_pipeline_context(
        self,
        prices: pd.DataFrame,
        ticker: str,
        interval: str,
        vocab: SAXVocabulary,
        model: WaveletGPT,
    ) -> tuple[NDArray | None, NDArray]:
        """Build model input from price data.

        Auto-detects model input mode:
        - "tokenized": SAX pipeline (DWT -> SAX -> tokenize -> sequence dataset)
        - "continuous": D1 pipeline (DWT -> delta coefficients -> aux features -> windows)

        Returns:
            (X_full, timestamps) where X_full shape depends on input mode,
            or (None, timestamps) if insufficient data.
        """
        if model.input_mode == "continuous":
            return self._build_d1_pipeline_context(prices, ticker, interval, model)
        return self._build_sax_pipeline_context(prices, ticker, interval, vocab, model)

    def _build_sax_pipeline_context(
        self,
        prices: pd.DataFrame,
        ticker: str,
        interval: str,
        vocab: SAXVocabulary,
        model: WaveletGPT,
    ) -> tuple[NDArray | None, NDArray]:
        """Build SAX-based model input (original tokenized pipeline)."""
        close_values = prices["close"].to_numpy(dtype=np.float64)
        ts_values = prices["timestamp"].to_numpy(dtype="datetime64[ns]")

        if len(close_values) < 20:
            return None, ts_values

        ts = TimeSeries(
            values=close_values,
            timestamps=ts_values,
            ticker=ticker,
            interval=interval,
        )

        dwt_level = 5
        decomp = decompose(ts, level=dwt_level)

        sax_config = self.config.sax
        levels_to_use = self.config.dwt_levels

        level_words: dict[int, list[str]] = {}

        for lvl in levels_to_use:
            coeffs = decomp.detail_at_level(lvl)
            if len(coeffs) < 2:
                continue
            n_seg = min(sax_config.n_segments, len(coeffs))
            sax_rep = sax_transform(coeffs, n_seg, sax_config.alphabet_size)
            words = extract_words(
                sax_rep.symbols,
                sax_config.word_length,
                sax_config.word_stride,
            )
            level_words[lvl] = words

        if not level_words:
            return None, ts_values

        # Encode with pre-loaded vocabulary (no new vocab building)
        level_sequences: dict[int, TokenSequence] = {}
        for lvl, words in level_words.items():
            token_ids = vocab.encode_sequence(words)
            level_sequences[lvl] = TokenSequence(
                token_ids=token_ids,
                words=words,
                ticker=ticker,
                interval=interval,
                wavelet_level=lvl,
            )

        mlt = MultiLevelTokenSequence(
            ticker=ticker,
            interval=interval,
            level_sequences=level_sequences,
        )

        asset_class_map = self._build_asset_class_map([ticker])

        # Get context_length from model config
        context_length = model._config.get("context_length", 16)
        dataset = build_sequence_dataset(
            [mlt], vocab, context_length, asset_class_map
        )

        if len(dataset.samples) == 0:
            return None, ts_values

        X, _y, _, _ = dataset.to_arrays()
        levels = np.array(
            [s.level for s in dataset.samples], dtype=np.int64
        )
        ac = np.array(
            [s.asset_class_id for s in dataset.samples], dtype=np.int64
        )
        X_full = np.column_stack([X, levels, ac])

        return X_full, ts_values

    def _build_d1_pipeline_context(
        self,
        prices: pd.DataFrame,
        ticker: str,
        interval: str,
        model: WaveletGPT,
    ) -> tuple[NDArray | None, NDArray]:
        """Build D1 (Delta+Aux) continuous pipeline context.

        Pipeline: prices -> DWT -> np.diff(detail) -> aux features -> sliding windows -> X
        """
        close_values = prices["close"].to_numpy(dtype=np.float64)
        ts_values = prices["timestamp"].to_numpy(dtype="datetime64[ns]")

        if len(close_values) < 20:
            return None, ts_values

        ts = TimeSeries(
            values=close_values,
            timestamps=ts_values,
            ticker=ticker,
            interval=interval,
        )

        decomp = decompose(ts, level=5)
        context_length = model._config.get("context_length", 16)
        n_aux = model._config.get("n_aux_features", 0)
        levels_to_use = self.config.dwt_levels
        asset_class_map = self._build_asset_class_map([ticker])

        coeff_series: dict[tuple[str, int], NDArray] = {}
        aux_series: dict[tuple[str, int], NDArray] = {}

        for lvl in levels_to_use:
            detail = decomp.detail_at_level(lvl)
            if len(detail) < 2:
                continue
            deltas = np.diff(detail)
            if len(deltas) > context_length:
                coeff_series[(ticker, lvl)] = deltas
                if n_aux > 0:
                    aux_series[(ticker, lvl)] = compute_detail_auxiliary_features(
                        detail, decomp.approximation
                    )

        if not coeff_series:
            return None, ts_values

        ds = build_continuous_dataset(
            coeff_series, context_length, asset_class_map, normalize=True
        )

        if len(ds.windows) == 0:
            return None, ts_values

        ctx_arr, lvl_arr, ac_arr = ds.to_arrays()

        if n_aux > 0 and aux_series:
            aux_windows = self._build_aux_windows(aux_series, context_length, n_aux)
            aux_flat = aux_windows.reshape(len(aux_windows), -1)
            X_full = np.column_stack([ctx_arr, aux_flat, lvl_arr, ac_arr])
        else:
            X_full = np.column_stack([ctx_arr, lvl_arr, ac_arr])

        return X_full, ts_values

    @staticmethod
    def _build_aux_windows(
        aux_series: dict[tuple[str, int], NDArray],
        context_length: int,
        n_aux_features: int,
    ) -> NDArray:
        """Build sliding windows from auxiliary feature series (sorted order)."""
        windows: list[NDArray] = []
        for (_ticker, _level), aux in sorted(aux_series.items()):
            if len(aux) <= context_length:
                continue
            for i in range(len(aux) - context_length):
                windows.append(aux[i:i + context_length])
        if not windows:
            return np.empty((0, context_length, n_aux_features), dtype=np.float64)
        return np.array(windows, dtype=np.float64)

    def _compute_target_timestamp(
        self, last_ts: np.datetime64, interval: str, horizon: int
    ) -> str:
        """Compute the ISO8601 timestamp for when the predicted bar closes."""
        delta = _INTERVAL_DELTAS.get(interval, datetime.timedelta(hours=1))
        target = pd.Timestamp(last_ts) + delta * horizon
        return target.isoformat()

    def _load_model(self) -> WaveletGPT:
        """Load model from disk (cached after first call)."""
        if self._model is None:
            self._model = WaveletGPT.load(Path(self.config.model_path))
            logger.info("Loaded model from %s", self.config.model_path)
        return self._model

    def _load_vocab(self) -> SAXVocabulary:
        """Load vocabulary from disk (cached after first call)."""
        if self._vocab is None:
            self._vocab = SAXVocabulary.load(Path(self.config.vocab_path))
            logger.info(
                "Loaded vocabulary from %s (%d tokens)",
                self.config.vocab_path,
                self._vocab.size,
            )
        return self._vocab

    def _get_tracker(self) -> ForwardTestTracker:
        """Get or create the prediction tracker."""
        if self._tracker is None:
            self._tracker = ForwardTestTracker(
                log_dir=self.config.log_dir,
                test_name=self.config.test_name,
            )
        return self._tracker

    def _build_asset_class_map(self, tickers: list[str]) -> dict[str, int]:
        """Build ticker -> asset_class_id mapping (same as ExperimentRunner)."""
        from wavecast.core.universe import DEFAULT_UNIVERSE, LEGACY_UNIVERSE

        ticker_to_class: dict[str, int] = {}
        for asset in LEGACY_UNIVERSE.assets:
            class_id = ASSET_CLASS_ID_MAP.get(asset.asset_class.value, 0)
            ticker_to_class[asset.ticker] = class_id
        for asset in DEFAULT_UNIVERSE.assets:
            if asset.sector is not None:
                ticker_to_class[asset.ticker] = SECTOR_ID_MAP.get(
                    asset.sector.value, 0
                )

        return {t: ticker_to_class.get(t, 0) for t in tickers}
