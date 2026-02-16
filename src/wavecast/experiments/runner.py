"""Experiment runner: end-to-end walk-forward experiment execution."""

from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wavecast.core.config import SAXConfig
from wavecast.core.exceptions import DataNotFoundError, PipelineError
from wavecast.core.types import (
    MultiLevelTokenSequence,
    TimeSeries,
    TokenSequence,
)
from wavecast.data.cache import ParquetCache
from wavecast.evaluation.token_eval import evaluate_token_predictions
from wavecast.experiments.config import ExperimentConfig
from wavecast.experiments.metrics import (
    compute_baselines,
    compute_bootstrap_ci,
    level0_directional_accuracy,
)
from wavecast.experiments.result import ExperimentResult
from wavecast.experiments.splitter import (
    expanding_window_split,
    rolling_window_split,
    walk_forward_split,
)
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.sax.bow import extract_words
from wavecast.sax.sax import sax_transform
from wavecast.tokenizer.dataset import build_sequence_dataset
from wavecast.tokenizer.vocabulary import UNK_ID, SAXVocabulary
from wavecast.wavelets.dwt import decompose

logger = logging.getLogger(__name__)

# Sector.value returns strings — use this mapping for embedding IDs
SECTOR_ID_MAP: dict[str, int] = {
    "tech": 0,
    "finance": 1,
    "energy": 2,
    "healthcare": 3,
    "broad_etf": 4,
    "commodity_etf": 5,
}

# Legacy asset class mapping (for backward compat with old universes)
ASSET_CLASS_ID_MAP: dict[str, int] = {
    "equity": 0,
    "crypto": 1,
    "forex": 2,
    "commodity": 3,
}


class ExperimentRunner:
    """Runs walk-forward experiments with the full WaveCast pipeline.

    Pipeline per split:
    1. Split raw prices by date
    2. DWT decompose each split independently
    3. SAX transform each split's coefficients independently
    4. Build vocabulary from TRAIN words only
    5. Encode both splits with train-built vocabulary (test gets UNK for unseen)
    6. Build sequence datasets
    7. Train WaveletGPT on train, evaluate on test
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache = ParquetCache(cache_dir)

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Run an experiment, dispatching by split_mode.

        For "single": runs one walk-forward split.
        For "expanding"/"rolling": runs one experiment per split, aggregates metrics.

        Args:
            config: Experiment configuration.

        Returns:
            ExperimentResult with all metrics (mean +/- std for multi-split).

        Raises:
            DataNotFoundError: If cached price data is missing for a ticker.
            PipelineError: If the pipeline fails at any stage.
            ConfigError: If split_mode params are missing.
        """
        if config.split_mode == "single":
            return self._run_single(config)

        if config.split_mode in ("expanding", "rolling"):
            return self._run_multi_split(config)

        from wavecast.core.exceptions import ConfigError
        raise ConfigError(
            f"Unknown split_mode '{config.split_mode}'. "
            "Must be 'single', 'expanding', or 'rolling'."
        )

    def _run_multi_split(self, config: ExperimentConfig) -> ExperimentResult:
        """Run experiment across multiple expanding/rolling splits and aggregate."""
        t0 = time.monotonic()
        logger.info("Starting multi-split experiment: %s (mode=%s)", config.name, config.split_mode)

        price_series = self._load_prices(config)
        logger.info("Loaded %d tickers", len(price_series))

        if config.split_mode == "expanding":
            from wavecast.core.exceptions import ConfigError
            if config.initial_train_size is None or config.test_window_size is None or config.step_size is None:
                raise ConfigError(
                    "expanding split_mode requires initial_train_size, test_window_size, step_size"
                )
            splits = expanding_window_split(
                price_series,
                initial_train_end=config.initial_train_size,
                test_window_size=config.test_window_size,
                step_size=config.step_size,
            )
        else:  # rolling
            from wavecast.core.exceptions import ConfigError
            if config.train_window_size is None or config.test_window_size is None or config.step_size is None:
                raise ConfigError(
                    "rolling split_mode requires train_window_size, test_window_size, step_size"
                )
            splits = rolling_window_split(
                price_series,
                train_window_size=config.train_window_size,
                test_window_size=config.test_window_size,
                step_size=config.step_size,
            )

        logger.info("Generated %d splits", len(splits))

        split_token_accs: list[float] = []
        split_dir_accs: list[float] = []
        split_results: list[ExperimentResult] = []

        for i, (train_prices, test_prices) in enumerate(splits):
            logger.info("Split %d/%d", i + 1, len(splits))
            result = self._run_on_split(config, train_prices, test_prices)
            split_results.append(result)
            split_token_accs.append(result.token_accuracy)
            split_dir_accs.append(result.directional_accuracy)

        # Aggregate: use mean for primary metrics, compute std across splits
        mean_token_acc = float(np.mean(split_token_accs))
        std_token_acc = float(np.std(split_token_accs))
        mean_dir_acc = float(np.mean(split_dir_accs))
        std_dir_acc = float(np.std(split_dir_accs))
        mean_top3 = float(np.mean([r.top3_accuracy for r in split_results]))
        mean_mf = float(np.mean([r.baseline_most_frequent for r in split_results]))
        mean_pers = float(np.mean([r.baseline_persistence for r in split_results]))
        mean_mom = float(np.mean([r.baseline_momentum for r in split_results]))
        total_train = sum(r.n_train_samples for r in split_results)
        total_test = sum(r.n_test_samples for r in split_results)

        elapsed = time.monotonic() - t0
        logger.info(
            "Multi-split %s complete in %.1fs: token_acc=%.4f +/- %.4f, dir_acc=%.4f +/- %.4f",
            config.name, elapsed, mean_token_acc, std_token_acc, mean_dir_acc, std_dir_acc,
        )

        # Use last split's CI as representative (bootstrap on aggregated is complex)
        last = split_results[-1]
        return ExperimentResult(
            config=config,
            token_accuracy=mean_token_acc,
            token_accuracy_ci=last.token_accuracy_ci,
            top3_accuracy=mean_top3,
            directional_accuracy=mean_dir_acc,
            directional_accuracy_ci=last.directional_accuracy_ci,
            baseline_most_frequent=mean_mf,
            baseline_persistence=mean_pers,
            baseline_momentum=mean_mom,
            per_asset_accuracy=last.per_asset_accuracy,
            per_sector_accuracy={},
            per_level_accuracy=last.per_level_accuracy,
            vocab_size=last.vocab_size,
            unk_rate=float(np.mean([r.unk_rate for r in split_results])),
            n_train_samples=total_train,
            n_test_samples=total_test,
            training_time_seconds=elapsed,
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            n_splits=len(splits),
            token_accuracy_std=std_token_acc,
            directional_accuracy_std=std_dir_acc,
        )

    def _run_single(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single walk-forward split experiment."""
        t0 = time.monotonic()
        logger.info("Starting experiment: %s", config.name)

        price_series = self._load_prices(config)
        logger.info("Loaded %d tickers", len(price_series))

        train_prices, test_prices = walk_forward_split(
            price_series, config.train_end, config.test_start
        )
        logger.info(
            "Split: %d train tickers, %d test tickers",
            len(train_prices),
            len(test_prices),
        )

        result = self._run_on_split(config, train_prices, test_prices)

        elapsed = time.monotonic() - t0
        logger.info("Experiment %s complete in %.1fs", config.name, elapsed)

        # Update timing to cover full run including data loading
        return ExperimentResult(
            config=result.config,
            token_accuracy=result.token_accuracy,
            token_accuracy_ci=result.token_accuracy_ci,
            top3_accuracy=result.top3_accuracy,
            directional_accuracy=result.directional_accuracy,
            directional_accuracy_ci=result.directional_accuracy_ci,
            baseline_most_frequent=result.baseline_most_frequent,
            baseline_persistence=result.baseline_persistence,
            baseline_momentum=result.baseline_momentum,
            per_asset_accuracy=result.per_asset_accuracy,
            per_sector_accuracy=result.per_sector_accuracy,
            per_level_accuracy=result.per_level_accuracy,
            vocab_size=result.vocab_size,
            unk_rate=result.unk_rate,
            n_train_samples=result.n_train_samples,
            n_test_samples=result.n_test_samples,
            training_time_seconds=elapsed,
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
        )

    def _run_on_split(
        self,
        config: ExperimentConfig,
        train_prices: dict[str, TimeSeries],
        test_prices: dict[str, TimeSeries],
    ) -> ExperimentResult:
        """Run the full pipeline on a single pre-split train/test pair."""
        t0 = time.monotonic()

        common_tickers = sorted(
            set(train_prices.keys()) & set(test_prices.keys())
        )
        if not common_tickers:
            raise PipelineError(
                "No tickers have data in both train and test periods"
            )

        dwt_level = 5
        sax_config = SAXConfig(
            n_segments=config.n_segments,
            alphabet_size=config.alphabet_size,
            word_length=config.word_length,
            word_stride=config.word_stride,
        )

        train_words_all: list[list[str]] = []
        train_token_seqs: list[MultiLevelTokenSequence] = []
        test_token_seqs_raw: list[tuple[str, dict[int, list[str]]]] = []
        levels_to_use = config.dwt_levels or list(range(1, dwt_level + 1))

        for ticker in common_tickers:
            train_decomp = decompose(train_prices[ticker], level=dwt_level)
            test_decomp = decompose(test_prices[ticker], level=dwt_level)

            train_level_words: dict[int, list[str]] = {}
            test_level_words: dict[int, list[str]] = {}

            for lvl in levels_to_use:
                train_coeffs = train_decomp.detail_at_level(lvl)
                if len(train_coeffs) >= 2:
                    n_seg = min(sax_config.n_segments, len(train_coeffs))
                    train_sax = sax_transform(
                        train_coeffs, n_seg, sax_config.alphabet_size
                    )
                    tw = extract_words(
                        train_sax.symbols,
                        sax_config.word_length,
                        sax_config.word_stride,
                    )
                    train_level_words[lvl] = tw
                    train_words_all.append(tw)

                test_coeffs = test_decomp.detail_at_level(lvl)
                if len(test_coeffs) >= 2:
                    n_seg = min(sax_config.n_segments, len(test_coeffs))
                    test_sax = sax_transform(
                        test_coeffs, n_seg, sax_config.alphabet_size
                    )
                    test_level_words[lvl] = extract_words(
                        test_sax.symbols,
                        sax_config.word_length,
                        sax_config.word_stride,
                    )

            test_token_seqs_raw.append((ticker, test_level_words))
            train_token_seqs.append(
                _words_to_placeholder_mlt(ticker, config.interval, train_level_words)
            )

        vocabulary = SAXVocabulary.from_corpus(
            train_words_all,
            min_freq=config.min_word_freq,
            max_size=config.max_vocab_size,
        )
        vocab_size = vocabulary.size

        train_mlts = _encode_placeholder_mlts(train_token_seqs, vocabulary)
        test_mlts: list[MultiLevelTokenSequence] = []
        for ticker, level_words in test_token_seqs_raw:
            level_sequences: dict[int, TokenSequence] = {}
            for lvl, words in level_words.items():
                token_ids = vocabulary.encode_sequence(words)
                level_sequences[lvl] = TokenSequence(
                    token_ids=token_ids,
                    words=words,
                    ticker=ticker,
                    interval=config.interval,
                    wavelet_level=lvl,
                )
            test_mlts.append(
                MultiLevelTokenSequence(
                    ticker=ticker,
                    interval=config.interval,
                    level_sequences=level_sequences,
                )
            )

        asset_class_map = self._build_asset_class_map(common_tickers)

        train_dataset = build_sequence_dataset(
            train_mlts, vocabulary, config.context_length, asset_class_map
        )
        test_dataset = build_sequence_dataset(
            test_mlts, vocabulary, config.context_length, asset_class_map
        )

        n_train = len(train_dataset.samples)
        n_test = len(test_dataset.samples)

        if n_train == 0:
            raise PipelineError("No training samples generated")
        if n_test == 0:
            raise PipelineError("No test samples generated")

        X_train, y_train, _, _ = train_dataset.to_arrays()
        X_test, y_test, _, _ = test_dataset.to_arrays()

        train_levels = np.array([s.level for s in train_dataset.samples], dtype=np.int64)
        train_ac = np.array([s.asset_class_id for s in train_dataset.samples], dtype=np.int64)
        test_levels = np.array([s.level for s in test_dataset.samples], dtype=np.int64)
        test_ac = np.array([s.asset_class_id for s in test_dataset.samples], dtype=np.int64)

        X_train_full = np.column_stack([X_train, train_levels, train_ac])
        X_test_full = np.column_stack([X_test, test_levels, test_ac])

        model = WaveletGPT(
            vocab_size=vocab_size,
            context_length=config.context_length,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            patience=config.patience,
        )

        model.fit(X_train_full, y_train, X_val=X_test_full, y_val=y_test)

        predicted = model.predict(X_test_full)
        proba = model.predict_proba(X_test_full)

        eval_metrics = evaluate_token_predictions(
            predicted, y_test, vocab_size, proba
        )

        unk_count = int(np.sum(y_test == UNK_ID))
        unk_rate = unk_count / n_test if n_test > 0 else 0.0

        all_train_tokens = np.concatenate([
            np.array(s.context_tokens + [s.target_token], dtype=np.int64)
            for s in train_dataset.samples
        ])
        baselines = compute_baselines(all_train_tokens, X_test, y_test)

        def _token_acc(preds: NDArray, actuals: NDArray) -> float:
            return float(np.mean(preds == actuals))

        token_ci = compute_bootstrap_ci(_token_acc, predicted, y_test)

        def _dir_acc(preds: NDArray, actuals: NDArray) -> float:
            return level0_directional_accuracy(preds, actuals, vocabulary)

        dir_ci = compute_bootstrap_ci(_dir_acc, predicted, y_test)

        per_asset: dict[str, float] = {}
        sample_tickers: list[str] = []
        for mlt in test_mlts:
            for _lvl, seq in mlt.level_sequences.items():
                n_samples_for_level = max(0, len(seq.token_ids) - config.context_length)
                sample_tickers.extend([mlt.ticker] * n_samples_for_level)

        if len(sample_tickers) == n_test:
            ticker_arr = np.array(sample_tickers)
            for t in common_tickers:
                mask = ticker_arr == t
                if np.any(mask):
                    per_asset[t] = float(np.mean(predicted[mask] == y_test[mask]))

        per_level: dict[int, float] = {}
        for lvl in sorted(set(int(v) for v in test_levels)):
            mask = test_levels == lvl
            if np.any(mask):
                per_level[lvl] = float(np.mean(predicted[mask] == y_test[mask]))

        directional_accuracy = level0_directional_accuracy(
            predicted, y_test, vocabulary
        )

        elapsed = time.monotonic() - t0

        return ExperimentResult(
            config=config,
            token_accuracy=eval_metrics.token_accuracy,
            token_accuracy_ci=token_ci,
            top3_accuracy=eval_metrics.top3_accuracy,
            directional_accuracy=directional_accuracy,
            directional_accuracy_ci=dir_ci,
            baseline_most_frequent=baselines["most_frequent"],
            baseline_persistence=baselines["persistence"],
            baseline_momentum=baselines["momentum"],
            per_asset_accuracy=per_asset,
            per_sector_accuracy={},
            per_level_accuracy=per_level,
            vocab_size=vocab_size,
            unk_rate=unk_rate,
            n_train_samples=n_train,
            n_test_samples=n_test,
            training_time_seconds=elapsed,
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
        )

    def run_sweep(
        self, configs: list[ExperimentConfig]
    ) -> list[ExperimentResult]:
        """Run multiple experiments sequentially.

        Args:
            configs: List of experiment configurations.

        Returns:
            List of ExperimentResult, one per config.
        """
        results: list[ExperimentResult] = []
        for i, config in enumerate(configs):
            logger.info(
                "Sweep %d/%d: %s", i + 1, len(configs), config.name
            )
            result = self.run(config)
            results.append(result)
        return results

    def _load_prices(
        self, config: ExperimentConfig
    ) -> dict[str, TimeSeries]:
        """Load cached price data for all tickers in the config."""
        prices: dict[str, TimeSeries] = {}
        for ticker in config.tickers:
            ts = self.cache.get(ticker, config.interval)
            if ts is None:
                raise DataNotFoundError(
                    f"No cached data for {ticker} at interval {config.interval}. "
                    f"Run 'wavecast data fetch {ticker} --interval {config.interval}' first."
                )
            prices[ticker] = ts
        return prices

    def _build_asset_class_map(
        self, tickers: list[str]
    ) -> dict[str, int]:
        """Build ticker -> asset_class_id mapping.

        Prefers sector-based IDs from DEFAULT_UNIVERSE, falls back to
        legacy asset class IDs, then to 0 (equity/tech).
        """
        from wavecast.core.universe import DEFAULT_UNIVERSE, LEGACY_UNIVERSE

        ticker_to_class: dict[str, int] = {}
        # First pass: legacy universe (lower priority)
        for asset in LEGACY_UNIVERSE.assets:
            class_id = ASSET_CLASS_ID_MAP.get(asset.asset_class.value, 0)
            ticker_to_class[asset.ticker] = class_id
        # Second pass: default universe sector IDs (higher priority)
        for asset in DEFAULT_UNIVERSE.assets:
            if asset.sector is not None:
                ticker_to_class[asset.ticker] = SECTOR_ID_MAP.get(asset.sector.value, 0)

        return {t: ticker_to_class.get(t, 0) for t in tickers}


def _words_to_placeholder_mlt(
    ticker: str,
    interval: str,
    level_words: dict[int, list[str]],
) -> MultiLevelTokenSequence:
    """Create a MultiLevelTokenSequence storing words but no token_ids yet.

    The words field is populated; token_ids will be filled after vocabulary
    is built.
    """
    level_sequences: dict[int, TokenSequence] = {}
    for lvl, words in level_words.items():
        level_sequences[lvl] = TokenSequence(
            token_ids=[],  # placeholder — encoded later
            words=words,
            ticker=ticker,
            interval=interval,
            wavelet_level=lvl,
        )
    return MultiLevelTokenSequence(
        ticker=ticker,
        interval=interval,
        level_sequences=level_sequences,
    )


def _encode_placeholder_mlts(
    mlts: list[MultiLevelTokenSequence],
    vocabulary: SAXVocabulary,
) -> list[MultiLevelTokenSequence]:
    """Encode the words in placeholder MLTs using the given vocabulary."""
    encoded: list[MultiLevelTokenSequence] = []
    for mlt in mlts:
        level_sequences: dict[int, TokenSequence] = {}
        for lvl, seq in mlt.level_sequences.items():
            token_ids = vocabulary.encode_sequence(seq.words)
            level_sequences[lvl] = TokenSequence(
                token_ids=token_ids,
                words=seq.words,
                ticker=seq.ticker,
                interval=seq.interval,
                wavelet_level=seq.wavelet_level,
            )
        encoded.append(
            MultiLevelTokenSequence(
                ticker=mlt.ticker,
                interval=mlt.interval,
                level_sequences=level_sequences,
            )
        )
    return encoded
