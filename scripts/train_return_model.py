#!/usr/bin/env python3
"""Train a return-quantile WaveletGPT on all 20 tickers.

Usage:
    python scripts/train_return_model.py

Trains on 2021-2024, tests on 2025. Saves model + vocab + boundaries to
~/.wavecast/models/return_quantile_v1/. Prints mini-audit at end.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from wavecast.core.types import TimeSeries
from wavecast.core.universe import DEFAULT_UNIVERSE
from wavecast.data.cache import ParquetCache
from wavecast.evaluation.return_eval import evaluate_return_predictions
from wavecast.experiments.metrics import compute_return_baselines
from wavecast.experiments.runner import (
    SECTOR_ID_MAP,
    _encode_placeholder_mlts,
    _words_to_placeholder_mlt,
)
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.sax.bow import extract_words
from wavecast.sax.sax import sax_transform
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
    compute_sample_returns,
)
from wavecast.tokenizer.dataset import build_return_target_dataset
from wavecast.tokenizer.vocabulary import SAXVocabulary
from wavecast.wavelets.dwt import decompose

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Config
TICKERS = [a.ticker for a in DEFAULT_UNIVERSE.assets]
INTERVAL = "1h"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
DWT_LEVELS = [1, 2, 5]
CONTEXT_LENGTH = 16
N_CLASSES = 5
PERCENTILES = [10.0, 30.0, 70.0, 90.0]

# Model hyperparameters (Phase 4 Optuna-optimized)
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 6
DROPOUT = 0.2
EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
PATIENCE = 15

# SAX parameters (Phase 4 Optuna-optimized)
N_SEGMENTS = 512
ALPHABET_SIZE = 7
WORD_LENGTH = 4
WORD_STRIDE = 1
MIN_WORD_FREQ = 1
MAX_VOCAB_SIZE = 100

OUTPUT_DIR = Path.home() / ".wavecast" / "models" / "return_quantile_v1"


def main() -> None:
    cache = ParquetCache(Path.home() / ".wavecast" / "cache")

    # Load and split prices
    logger.info("Loading prices for %d tickers...", len(TICKERS))
    train_prices: dict[str, TimeSeries] = {}
    test_prices: dict[str, TimeSeries] = {}

    for ticker in TICKERS:
        ts = cache.get(ticker, INTERVAL)
        if ts is None:
            logger.warning("No data for %s, skipping", ticker)
            continue
        # Split by date
        train_mask = ts.timestamps <= np.datetime64(TRAIN_END)
        test_mask = ts.timestamps >= np.datetime64(TEST_START)
        if train_mask.sum() > 100 and test_mask.sum() > 50:
            train_prices[ticker] = TimeSeries(
                values=ts.values[train_mask],
                timestamps=ts.timestamps[train_mask],
                ticker=ticker,
                interval=INTERVAL,
            )
            test_prices[ticker] = TimeSeries(
                values=ts.values[test_mask],
                timestamps=ts.timestamps[test_mask],
                ticker=ticker,
                interval=INTERVAL,
            )

    common = sorted(set(train_prices) & set(test_prices))
    logger.info("Using %d tickers", len(common))

    # DWT → SAX → words
    train_words_all: list[list[str]] = []
    train_token_seqs = []
    test_token_seqs_raw = []
    train_coeffs_map: dict[tuple[str, int], int] = {}
    train_symbols_map: dict[tuple[str, int], int] = {}
    test_coeffs_map: dict[tuple[str, int], int] = {}
    test_symbols_map: dict[tuple[str, int], int] = {}

    for ticker in common:
        train_decomp = decompose(train_prices[ticker], level=5)
        test_decomp = decompose(test_prices[ticker], level=5)
        train_level_words: dict[int, list[str]] = {}
        test_level_words: dict[int, list[str]] = {}

        for lvl in DWT_LEVELS:
            tc = train_decomp.detail_at_level(lvl)
            if len(tc) >= 2:
                ns = min(N_SEGMENTS, len(tc))
                tsax = sax_transform(tc, ns, ALPHABET_SIZE)
                tw = extract_words(tsax.symbols, WORD_LENGTH, WORD_STRIDE)
                train_level_words[lvl] = tw
                train_words_all.append(tw)
                train_coeffs_map[(ticker, lvl)] = len(tc)
                train_symbols_map[(ticker, lvl)] = len(tsax.symbols)

            ec = test_decomp.detail_at_level(lvl)
            if len(ec) >= 2:
                ns = min(N_SEGMENTS, len(ec))
                esax = sax_transform(ec, ns, ALPHABET_SIZE)
                ew = extract_words(esax.symbols, WORD_LENGTH, WORD_STRIDE)
                test_level_words[lvl] = ew
                test_coeffs_map[(ticker, lvl)] = len(ec)
                test_symbols_map[(ticker, lvl)] = len(esax.symbols)

        test_token_seqs_raw.append((ticker, test_level_words))
        train_token_seqs.append(
            _words_to_placeholder_mlt(ticker, INTERVAL, train_level_words)
        )

    # Build vocabulary
    vocabulary = SAXVocabulary.from_corpus(
        train_words_all, min_freq=MIN_WORD_FREQ, max_size=MAX_VOCAB_SIZE
    )
    vocab_size = vocabulary.size
    logger.info("Vocabulary size: %d", vocab_size)

    # Encode
    from wavecast.core.types import MultiLevelTokenSequence, TokenSequence

    train_mlts = _encode_placeholder_mlts(train_token_seqs, vocabulary)
    test_mlts = []
    for ticker, lw in test_token_seqs_raw:
        ls = {}
        for lvl, words in lw.items():
            ids = vocabulary.encode_sequence(words)
            ls[lvl] = TokenSequence(
                token_ids=ids, words=words, ticker=ticker,
                interval=INTERVAL, wavelet_level=lvl,
            )
        test_mlts.append(MultiLevelTokenSequence(
            ticker=ticker, interval=INTERVAL, level_sequences=ls,
        ))

    # Asset class map
    asset_class_map = {}
    for a in DEFAULT_UNIVERSE.assets:
        if a.sector is not None:
            asset_class_map[a.ticker] = SECTOR_ID_MAP.get(a.sector.value, 0)

    # Build datasets with metadata
    train_ds = build_return_target_dataset(
        train_mlts, vocabulary, CONTEXT_LENGTH, asset_class_map,
        n_coeffs_map=train_coeffs_map, n_symbols_map=train_symbols_map,
    )
    test_ds = build_return_target_dataset(
        test_mlts, vocabulary, CONTEXT_LENGTH, asset_class_map,
        n_coeffs_map=test_coeffs_map, n_symbols_map=test_symbols_map,
    )

    logger.info("Train samples: %d, Test samples: %d", len(train_ds.samples), len(test_ds.samples))

    # Compute returns
    def compute_ds_returns(ds, mlts, prices_dict, cm, sm):
        tp = np.array([s.token_position for s in ds.samples], dtype=np.int64)
        lvls = np.array([s.level for s in ds.samples], dtype=np.int64)
        nc = np.array([s.n_coeffs for s in ds.samples], dtype=np.int64)
        ns = np.array([s.n_symbols for s in ds.samples], dtype=np.int64)

        tickers_list = []
        for mlt in mlts:
            for _l, seq in mlt.level_sequences.items():
                n_s = max(0, len(seq.token_ids) - CONTEXT_LENGTH)
                tickers_list.extend([mlt.ticker] * n_s)

        rets = np.full(len(ds.samples), np.nan)
        valid = np.zeros(len(ds.samples), dtype=np.bool_)

        if len(tickers_list) == len(ds.samples):
            tarr = np.array(tickers_list)
            for t in set(tickers_list):
                m = tarr == t
                if t not in prices_dict:
                    continue
                r, v = compute_sample_returns(tp[m], lvls[m], prices_dict[t].values, nc[m], ns[m])
                rets[m] = r
                valid[m] = v

        return rets, valid, lvls

    train_rets, train_valid, train_lvls = compute_ds_returns(
        train_ds, train_mlts, train_prices, train_coeffs_map, train_symbols_map
    )
    test_rets, test_valid, test_lvls = compute_ds_returns(
        test_ds, test_mlts, test_prices, test_coeffs_map, test_symbols_map
    )

    logger.info("Train valid: %d/%d, Test valid: %d/%d",
                train_valid.sum(), len(train_valid), test_valid.sum(), len(test_valid))

    # Compute boundaries from training
    boundaries = compute_quantile_boundaries(
        train_rets, train_lvls, train_valid, PERCENTILES, per_level=True
    )
    y_train_labels = assign_quantile_labels(train_rets, train_lvls, boundaries)
    y_test_labels = assign_quantile_labels(test_rets, test_lvls, boundaries)

    # Class weights
    valid_labels = y_train_labels[train_valid]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()
    logger.info("Class distribution (train): %s", dict(enumerate(counts.astype(int).tolist())))
    logger.info("Class weights: %s", [f"{w:.3f}" for w in class_weights])

    # Prepare arrays
    X_train, _, _, _ = train_ds.to_arrays()
    X_test, _, _, _ = test_ds.to_arrays()
    train_lvl_arr = np.array([s.level for s in train_ds.samples], dtype=np.int64)
    train_ac_arr = np.array([s.asset_class_id for s in train_ds.samples], dtype=np.int64)
    test_lvl_arr = np.array([s.level for s in test_ds.samples], dtype=np.int64)
    test_ac_arr = np.array([s.asset_class_id for s in test_ds.samples], dtype=np.int64)
    X_train_full = np.column_stack([X_train, train_lvl_arr, train_ac_arr])
    X_test_full = np.column_stack([X_test, test_lvl_arr, test_ac_arr])

    y_train_target = y_train_labels.astype(np.float64)
    y_test_target = y_test_labels.astype(np.float64)

    # Train
    logger.info("Training return-quantile model...")
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
        task="return_quantile",
        n_output_classes=N_CLASSES,
        class_weights=class_weights,
    )
    metrics = model.fit(X_train_full, y_train_target, X_val=X_test_full, y_val=y_test_target)
    logger.info("Training metrics: %s", metrics)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_DIR)
    vocabulary.save(OUTPUT_DIR / "vocabulary.json")
    with open(OUTPUT_DIR / "boundaries.json", "w") as f:
        json.dump({str(k): v.tolist() for k, v in boundaries.items()}, f, indent=2)
    logger.info("Saved to %s", OUTPUT_DIR)

    # Mini-audit
    predicted = model.predict(X_test_full)
    pred_labels = predicted.astype(np.int64)

    rt_metrics = evaluate_return_predictions(
        pred_labels, test_rets, y_test_labels, n_classes=N_CLASSES
    )
    baselines = compute_return_baselines(
        y_train_labels[train_valid], y_test_labels[test_valid], N_CLASSES
    )

    print("\n" + "=" * 60)
    print("RETURN-QUANTILE MODEL — MINI AUDIT")
    print("=" * 60)
    print(f"Quantile accuracy:     {rt_metrics.quantile_accuracy:.4f}")
    print(f"Directional accuracy:  {rt_metrics.directional_accuracy:.4f}")
    print(f"Strong signal acc:     {rt_metrics.strong_signal_accuracy:.4f}")
    print(f"Economic value:        {rt_metrics.economic_value:.6f}")
    print("\nBaselines:")
    print(f"  Always flat:         {baselines['always_flat']:.4f}")
    print(f"  Always up:           {baselines['always_up']:.4f}")
    print(f"  Random (1/{N_CLASSES}):         {baselines['random']:.4f}")
    print(f"  Level majority:      {baselines['level_majority']:.4f}")
    print("\nMean return per predicted class:")
    for c, r in sorted(rt_metrics.mean_return_per_class.items()):
        label = ["strong_down", "down", "flat", "up", "strong_up"][c]
        print(f"  {label} (class {c}): {r:+.6f}")
    print("=" * 60)

    # Key verdict
    if rt_metrics.directional_accuracy > 0.55:
        print("\n*** STRONG SIGNAL: Economic directional accuracy > 55% ***")
    elif rt_metrics.directional_accuracy > 0.50:
        print("\n*** MARGINAL SIGNAL: Economic directional accuracy > 50% ***")
    else:
        print("\n*** NO SIGNAL: Economic directional accuracy <= 50% ***")
        print("*** SAX features may be uninformative for returns ***")


if __name__ == "__main__":
    main()
