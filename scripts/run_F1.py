"""F1: Multi-horizon prediction experiment.

Evaluates how prediction accuracy degrades across horizons 1, 2, 4, 8.

Uses Phase 3 optimal config:
  alphabet=7, levels=[1,2,5], context=16, vocab=100, min_freq=1, cross-sector
  Walk-forward: train 2021-2023, test 2024

For each horizon, reports: token accuracy, top-3, directional accuracy with bootstrap CIs.
Also tests expanding-window splits if available.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wavecast.core.config import SAXConfig
from wavecast.core.exceptions import PipelineError
from wavecast.core.types import MultiLevelTokenSequence, TimeSeries, TokenSequence
from wavecast.core.universe import PHASE3_UNIVERSE
from wavecast.data.cache import ParquetCache
from wavecast.evaluation.token_eval import evaluate_multi_horizon
from wavecast.experiments.metrics import (
    compute_baselines,
    compute_bootstrap_ci,
    level0_directional_accuracy,
)
from wavecast.experiments.splitter import expanding_window_split, walk_forward_split
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.sax.bow import extract_words
from wavecast.sax.sax import sax_transform
from wavecast.tokenizer.dataset import build_sequence_dataset
from wavecast.tokenizer.vocabulary import SAXVocabulary
from wavecast.wavelets.dwt import decompose

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)

TICKERS = PHASE3_UNIVERSE.tickers
CACHE_DIR = Path.home() / ".wavecast" / "cache"
OUTPUT_PATH = Path.home() / ".wavecast" / "experiments" / "F1_multi_horizon.json"

# Phase 3 optimal config
HORIZONS = [1, 2, 4, 8]
ALPHABET_SIZE = 7
DWT_LEVELS = [1, 2, 5]
CONTEXT_LENGTH = 16
MAX_VOCAB_SIZE = 100
MIN_WORD_FREQ = 1
N_SEGMENTS = 256
WORD_LENGTH = 4
WORD_STRIDE = 1

# Model architecture
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.1
EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
PATIENCE = 15

TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"

# Sector groupings for per-sector analysis
SECTORS = {
    "tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    "finance": ["JPM", "GS", "BAC"],
    "energy": ["XOM", "CVX", "COP"],
    "healthcare": ["JNJ", "UNH", "PFE"],
    "broad_etf": ["SPY", "QQQ"],
    "commodity_etf": ["GLD", "SLV", "USO", "UNG"],
}


def _words_to_placeholder_mlt(
    ticker: str, interval: str, level_words: dict[int, list[str]]
) -> MultiLevelTokenSequence:
    level_sequences: dict[int, TokenSequence] = {}
    for lvl, words in level_words.items():
        level_sequences[lvl] = TokenSequence(
            token_ids=[], words=words, ticker=ticker,
            interval=interval, wavelet_level=lvl,
        )
    return MultiLevelTokenSequence(
        ticker=ticker, interval=interval, level_sequences=level_sequences,
    )


def _encode_mlts(
    mlts: list[MultiLevelTokenSequence], vocabulary: SAXVocabulary
) -> list[MultiLevelTokenSequence]:
    encoded: list[MultiLevelTokenSequence] = []
    for mlt in mlts:
        level_sequences: dict[int, TokenSequence] = {}
        for lvl, seq in mlt.level_sequences.items():
            token_ids = vocabulary.encode_sequence(seq.words)
            level_sequences[lvl] = TokenSequence(
                token_ids=token_ids, words=seq.words, ticker=seq.ticker,
                interval=seq.interval, wavelet_level=seq.wavelet_level,
            )
        encoded.append(MultiLevelTokenSequence(
            ticker=mlt.ticker, interval=mlt.interval,
            level_sequences=level_sequences,
        ))
    return encoded


def _build_asset_class_map(tickers: list[str]) -> dict[str, int]:
    from wavecast.core.universe import DEFAULT_UNIVERSE
    from wavecast.experiments.runner import SECTOR_ID_MAP
    ticker_to_class: dict[str, int] = {}
    for asset in DEFAULT_UNIVERSE.assets:
        if asset.sector is not None:
            ticker_to_class[asset.ticker] = SECTOR_ID_MAP.get(asset.sector.value, 0)
    return {t: ticker_to_class.get(t, 0) for t in tickers}


def run_multi_horizon_experiment(
    train_prices: dict[str, TimeSeries],
    test_prices: dict[str, TimeSeries],
    split_label: str = "single",
) -> dict:
    """Run multi-horizon training and evaluation on a single split."""
    t0 = time.monotonic()

    common_tickers = sorted(set(train_prices.keys()) & set(test_prices.keys()))
    if not common_tickers:
        raise PipelineError("No tickers in both train and test")

    sax_config = SAXConfig(
        n_segments=N_SEGMENTS, alphabet_size=ALPHABET_SIZE,
        word_length=WORD_LENGTH, word_stride=WORD_STRIDE,
    )

    train_words_all: list[list[str]] = []
    train_token_seqs: list[MultiLevelTokenSequence] = []
    test_token_seqs_raw: list[tuple[str, dict[int, list[str]]]] = []

    for ticker in common_tickers:
        train_decomp = decompose(train_prices[ticker], level=5)
        test_decomp = decompose(test_prices[ticker], level=5)
        train_level_words: dict[int, list[str]] = {}
        test_level_words: dict[int, list[str]] = {}

        for lvl in DWT_LEVELS:
            train_coeffs = train_decomp.detail_at_level(lvl)
            if len(train_coeffs) >= 2:
                n_seg = min(sax_config.n_segments, len(train_coeffs))
                train_sax = sax_transform(train_coeffs, n_seg, sax_config.alphabet_size)
                tw = extract_words(train_sax.symbols, sax_config.word_length, sax_config.word_stride)
                train_level_words[lvl] = tw
                train_words_all.append(tw)

            test_coeffs = test_decomp.detail_at_level(lvl)
            if len(test_coeffs) >= 2:
                n_seg = min(sax_config.n_segments, len(test_coeffs))
                test_sax = sax_transform(test_coeffs, n_seg, sax_config.alphabet_size)
                test_level_words[lvl] = extract_words(
                    test_sax.symbols, sax_config.word_length, sax_config.word_stride
                )

        test_token_seqs_raw.append((ticker, test_level_words))
        train_token_seqs.append(
            _words_to_placeholder_mlt(ticker, "1h", train_level_words)
        )

    # Build vocabulary from train only
    vocabulary = SAXVocabulary.from_corpus(
        train_words_all, min_freq=MIN_WORD_FREQ, max_size=MAX_VOCAB_SIZE,
    )
    vocab_size = vocabulary.size
    logger.info("[%s] Vocabulary size: %d", split_label, vocab_size)

    # Encode
    train_mlts = _encode_mlts(train_token_seqs, vocabulary)
    test_mlts: list[MultiLevelTokenSequence] = []
    for ticker, level_words in test_token_seqs_raw:
        level_sequences: dict[int, TokenSequence] = {}
        for lvl, words in level_words.items():
            token_ids = vocabulary.encode_sequence(words)
            level_sequences[lvl] = TokenSequence(
                token_ids=token_ids, words=words, ticker=ticker,
                interval="1h", wavelet_level=lvl,
            )
        test_mlts.append(MultiLevelTokenSequence(
            ticker=ticker, interval="1h", level_sequences=level_sequences,
        ))

    asset_class_map = _build_asset_class_map(common_tickers)
    max_horizon = max(HORIZONS)

    # Build datasets with multi-horizon targets
    train_dataset = build_sequence_dataset(
        train_mlts, vocabulary, CONTEXT_LENGTH, asset_class_map, max_horizon=max_horizon,
    )
    test_dataset = build_sequence_dataset(
        test_mlts, vocabulary, CONTEXT_LENGTH, asset_class_map, max_horizon=max_horizon,
    )

    n_train = len(train_dataset.samples)
    n_test = len(test_dataset.samples)
    logger.info("[%s] Samples: %d train, %d test", split_label, n_train, n_test)

    if n_train == 0 or n_test == 0:
        raise PipelineError(f"No samples: {n_train} train, {n_test} test")

    # Get multi-horizon arrays
    X_train_ctx, y_train_2d, train_levels, train_ac = train_dataset.to_multi_horizon_arrays(HORIZONS)
    X_test_ctx, y_test_2d, test_levels, test_ac = test_dataset.to_multi_horizon_arrays(HORIZONS)

    X_train_full = np.column_stack([X_train_ctx, train_levels, train_ac])
    X_test_full = np.column_stack([X_test_ctx, test_levels, test_ac])

    # Train multi-horizon model
    model = WaveletGPT(
        vocab_size=vocab_size,
        context_length=CONTEXT_LENGTH,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
        prediction_horizons=HORIZONS,
    )

    logger.info("[%s] Training WaveletGPT with horizons=%s...", split_label, HORIZONS)
    train_metrics = model.fit(X_train_full, y_train_2d, X_val=X_test_full, y_val=y_test_2d)
    training_time = time.monotonic() - t0
    logger.info("[%s] Training complete in %.1fs", split_label, training_time)

    # Evaluate each horizon
    all_predictions = model.predict_all_horizons(X_test_full)
    all_probas = model.predict_proba_all_horizons(X_test_full)

    # Build actuals dict from y_test_2d columns
    actuals_dict: dict[int, np.ndarray] = {}
    for col_idx, h in enumerate(HORIZONS):
        actuals_dict[h] = y_test_2d[:, col_idx]

    multi_metrics = evaluate_multi_horizon(
        all_predictions, actuals_dict, vocab_size, all_probas,
    )

    # Per-horizon bootstrap CIs
    horizon_results: dict[int, dict] = {}
    for h in HORIZONS:
        preds_h = all_predictions[h]
        actual_h = actuals_dict[h]

        metrics = multi_metrics.per_horizon[h]

        def _token_acc(p: np.ndarray, a: np.ndarray) -> float:
            return float(np.mean(p == a))

        def _dir_acc(p: np.ndarray, a: np.ndarray) -> float:
            return level0_directional_accuracy(p, a, vocabulary)

        token_ci = compute_bootstrap_ci(_token_acc, preds_h, actual_h)
        dir_ci = compute_bootstrap_ci(_dir_acc, preds_h, actual_h)

        horizon_results[h] = {
            "horizon": h,
            "token_accuracy": metrics.token_accuracy,
            "token_accuracy_ci": token_ci,
            "top3_accuracy": metrics.top3_accuracy,
            "directional_accuracy": metrics.directional_accuracy,
            "directional_accuracy_ci": dir_ci,
        }

    # Baselines (horizon=1 only)
    all_train_tokens = np.concatenate([
        np.array(s.context_tokens + [s.target_token], dtype=np.int64)
        for s in train_dataset.samples
    ])
    _, y_test_h1, _, _ = test_dataset.to_arrays()
    X_test_ctx_1d, _, _, _ = test_dataset.to_arrays()
    baselines = compute_baselines(all_train_tokens, X_test_ctx_1d, y_test_h1)

    elapsed = time.monotonic() - t0

    return {
        "split_label": split_label,
        "horizons": HORIZONS,
        "per_horizon": horizon_results,
        "baselines": baselines,
        "vocab_size": vocab_size,
        "n_train_samples": n_train,
        "n_test_samples": n_test,
        "training_time_seconds": elapsed,
        "train_metrics": train_metrics,
    }


def main() -> None:
    cache = ParquetCache(CACHE_DIR)

    # Load prices
    price_series: dict[str, TimeSeries] = {}
    for ticker in TICKERS:
        ts = cache.get(ticker, "1h")
        if ts is None:
            logger.warning("No cached data for %s, skipping", ticker)
            continue
        price_series[ticker] = ts

    logger.info("Loaded %d tickers", len(price_series))
    if len(price_series) < 5:
        logger.error("Too few tickers loaded. Run fetch_phase3_universe.py first.")
        sys.exit(1)

    # === Single split: train 2021-2023, test 2024 ===
    logger.info("=" * 70)
    logger.info("F1: Multi-Horizon Prediction Experiment")
    logger.info("Horizons: %s", HORIZONS)
    logger.info("Phase 3 optimal: alphabet=%d, levels=%s, context=%d", ALPHABET_SIZE, DWT_LEVELS, CONTEXT_LENGTH)
    logger.info("=" * 70)

    train_prices, test_prices = walk_forward_split(price_series, TRAIN_END, TEST_START)
    single_result = run_multi_horizon_experiment(train_prices, test_prices, "single")

    # === Expanding window splits ===
    logger.info("\n" + "=" * 70)
    logger.info("Expanding window splits")
    logger.info("=" * 70)

    # Use about 60% of data for initial train, 20% test windows, 10% step
    n_samples = len(next(iter(price_series.values())).values)
    initial_train_size = int(n_samples * 0.6)
    test_window_size = int(n_samples * 0.15)
    step_size = int(n_samples * 0.1)

    expanding_results: list[dict] = []
    expanding_summary: dict[int, dict] = {}
    try:
        splits = expanding_window_split(
            price_series,
            initial_train_end=initial_train_size,
            test_window_size=test_window_size,
            step_size=step_size,
        )
        logger.info("Generated %d expanding window splits", len(splits))

        for i, (train_p, test_p) in enumerate(splits):
            logger.info("Expanding split %d/%d", i + 1, len(splits))
            try:
                split_result = run_multi_horizon_experiment(train_p, test_p, f"expanding_{i+1}")
                expanding_results.append(split_result)
            except Exception as e:
                logger.warning("Expanding split %d/%d failed: %s", i + 1, len(splits), e)

        if expanding_results:
            for h in HORIZONS:
                accs = [r["per_horizon"][h]["token_accuracy"] for r in expanding_results]
                dir_accs = [r["per_horizon"][h]["directional_accuracy"] for r in expanding_results]
                expanding_summary[h] = {
                    "horizon": h,
                    "mean_token_accuracy": float(np.mean(accs)),
                    "std_token_accuracy": float(np.std(accs)),
                    "mean_directional_accuracy": float(np.mean(dir_accs)),
                    "std_directional_accuracy": float(np.std(dir_accs)),
                    "n_splits": len(expanding_results),
                    "per_split_token_accuracy": accs,
                    "per_split_directional_accuracy": dir_accs,
                }
    except Exception as e:
        logger.warning("Expanding window setup failed: %s. Continuing with single split only.", e)

    # === Compile and save ===
    output = {
        "experiment": "F1_multi_horizon",
        "config": {
            "horizons": HORIZONS,
            "alphabet_size": ALPHABET_SIZE,
            "dwt_levels": DWT_LEVELS,
            "context_length": CONTEXT_LENGTH,
            "max_vocab_size": MAX_VOCAB_SIZE,
            "min_word_freq": MIN_WORD_FREQ,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "epochs": EPOCHS,
            "n_tickers": len(price_series),
            "train_end": TRAIN_END,
            "test_start": TEST_START,
        },
        "single_split": single_result,
        "expanding_window": {
            "n_splits": len(expanding_results),
            "per_horizon_summary": {str(k): v for k, v in expanding_summary.items()},
            "splits": expanding_results,
        } if expanding_results else None,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str))
    logger.info("Results saved to %s", OUTPUT_PATH)

    # === Print results ===
    print("\n" + "=" * 90)
    print("F1 RESULTS: Multi-Horizon Prediction")
    print("=" * 90)

    print("\n--- Single Split (train 2021-2023, test 2024) ---")
    print(f"{'Horizon':>8} {'Token Acc':>10} {'95% CI':>20} {'Top-3':>8} {'Dir Acc':>10} {'95% CI':>20}")
    print("-" * 80)
    for h in HORIZONS:
        hr = single_result["per_horizon"][h]
        tc = hr["token_accuracy_ci"]
        dc = hr["directional_accuracy_ci"]
        print(
            f"{h:>8} {hr['token_accuracy']:>10.4f} [{tc[0]:.4f}, {tc[1]:.4f}] "
            f"{hr['top3_accuracy']:>8.4f} {hr['directional_accuracy']:>10.4f} "
            f"[{dc[0]:.4f}, {dc[1]:.4f}]"
        )

    bl = single_result["baselines"]
    print(f"\nBaselines (horizon=1): persistence={bl['persistence']:.4f}, "
          f"most_frequent={bl['most_frequent']:.4f}, momentum={bl['momentum']:.4f}")

    # Accuracy degradation analysis
    h1_acc = single_result["per_horizon"][1]["token_accuracy"]
    print("\n--- Accuracy Degradation ---")
    for h in HORIZONS:
        acc = single_result["per_horizon"][h]["token_accuracy"]
        rel_drop = (h1_acc - acc) / h1_acc * 100 if h1_acc > 0 else 0
        print(f"  Horizon {h}: {acc:.4f} ({rel_drop:+.1f}% from h=1)")

    if expanding_summary:
        print("\n--- Expanding Window Summary ---")
        print(f"{'Horizon':>8} {'Mean TokAcc':>12} {'Std':>8} {'Mean DirAcc':>12} {'Std':>8} {'N splits':>10}")
        print("-" * 65)
        for h in HORIZONS:
            es = expanding_summary[h]
            print(
                f"{h:>8} {es['mean_token_accuracy']:>12.4f} {es['std_token_accuracy']:>8.4f} "
                f"{es['mean_directional_accuracy']:>12.4f} {es['std_directional_accuracy']:>8.4f} "
                f"{es['n_splits']:>10}"
            )

    print(f"\nTotal training time: {single_result['training_time_seconds']:.1f}s")
    print(f"Vocab size: {single_result['vocab_size']}")
    print(f"Samples: {single_result['n_train_samples']} train, {single_result['n_test_samples']} test")


if __name__ == "__main__":
    main()
