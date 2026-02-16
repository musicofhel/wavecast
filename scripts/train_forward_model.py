#!/usr/bin/env python3
"""Train a WaveletGPT model and save artifacts for forward testing.

Uses the Phase 3 optimal configuration on all 20 DEFAULT_UNIVERSE tickers.
Trains on 2021-2024 hourly data, validates on 2025.
Saves model + vocabulary to ~/.wavecast/models/forward_ready/.

Usage:
    cd ~/wavecast && source .venv/bin/activate
    python scripts/train_forward_model.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wavecast.core.config import SAXConfig
from wavecast.core.types import MultiLevelTokenSequence, TokenSequence
from wavecast.core.universe import DEFAULT_UNIVERSE
from wavecast.data.cache import ParquetCache
from wavecast.experiments.runner import (
    ASSET_CLASS_ID_MAP,
    SECTOR_ID_MAP,
    ExperimentRunner,
)
from wavecast.experiments.splitter import walk_forward_split
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.sax.bow import extract_words
from wavecast.sax.sax import sax_transform
from wavecast.tokenizer.dataset import build_sequence_dataset
from wavecast.tokenizer.vocabulary import SAXVocabulary
from wavecast.wavelets.dwt import decompose

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Phase 3 optimal configuration
ALPHABET_SIZE = 7
N_SEGMENTS = 256
WORD_LENGTH = 4
WORD_STRIDE = 1
DWT_LEVELS = [1, 2, 5]
CONTEXT_LENGTH = 16
MIN_WORD_FREQ = 1
MAX_VOCAB_SIZE = 100

# Model architecture
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.1
EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
PATIENCE = 15

# Paths
CACHE_DIR = Path.home() / ".wavecast" / "cache"
OUTPUT_DIR = Path.home() / ".wavecast" / "models" / "forward_ready"

# Split dates
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"


def build_asset_class_map(tickers: list[str]) -> dict[str, int]:
    """Build ticker -> asset_class_id mapping (matches ExperimentRunner)."""
    from wavecast.core.universe import DEFAULT_UNIVERSE, LEGACY_UNIVERSE

    ticker_to_class: dict[str, int] = {}
    for asset in LEGACY_UNIVERSE.assets:
        class_id = ASSET_CLASS_ID_MAP.get(asset.asset_class.value, 0)
        ticker_to_class[asset.ticker] = class_id
    for asset in DEFAULT_UNIVERSE.assets:
        if asset.sector is not None:
            ticker_to_class[asset.ticker] = SECTOR_ID_MAP.get(asset.sector.value, 0)

    return {t: ticker_to_class.get(t, 0) for t in tickers}


def main() -> None:
    t0 = time.monotonic()

    # 1. Load cached data
    cache = ParquetCache(CACHE_DIR)
    tickers = DEFAULT_UNIVERSE.tickers
    logger.info("Loading data for %d tickers: %s", len(tickers), tickers)

    prices = {}
    for ticker in tickers:
        ts = cache.get(ticker, "1h")
        if ts is None:
            logger.warning("No cached data for %s, skipping", ticker)
            continue
        prices[ticker] = ts
        logger.info("  %s: %d bars", ticker, len(ts.values))

    logger.info("Loaded %d/%d tickers", len(prices), len(tickers))

    # 2. Walk-forward split
    train_prices, test_prices = walk_forward_split(prices, TRAIN_END, TEST_START)
    common_tickers = sorted(set(train_prices.keys()) & set(test_prices.keys()))
    logger.info("Split: %d common tickers, train_end=%s, test_start=%s",
                len(common_tickers), TRAIN_END, TEST_START)

    # 3. DWT → SAX → words (exact same pipeline as ExperimentRunner)
    sax_config = SAXConfig(
        n_segments=N_SEGMENTS,
        alphabet_size=ALPHABET_SIZE,
        word_length=WORD_LENGTH,
        word_stride=WORD_STRIDE,
    )
    dwt_level = 5

    train_words_all: list[list[str]] = []
    train_mlts: list[MultiLevelTokenSequence] = []
    test_raw: list[tuple[str, dict[int, list[str]]]] = []

    for ticker in common_tickers:
        train_decomp = decompose(train_prices[ticker], level=dwt_level)
        test_decomp = decompose(test_prices[ticker], level=dwt_level)

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

        test_raw.append((ticker, test_level_words))

        # Build placeholder MLT for train (words only, encode later)
        level_sequences: dict[int, TokenSequence] = {}
        for lvl, words in train_level_words.items():
            level_sequences[lvl] = TokenSequence(
                token_ids=[],
                words=words,
                ticker=ticker,
                interval="1h",
                wavelet_level=lvl,
            )
        train_mlts.append(MultiLevelTokenSequence(
            ticker=ticker, interval="1h", level_sequences=level_sequences,
        ))

    # 4. Build vocabulary from TRAIN words only
    vocabulary = SAXVocabulary.from_corpus(
        train_words_all, min_freq=MIN_WORD_FREQ, max_size=MAX_VOCAB_SIZE,
    )
    vocab_size = vocabulary.size
    logger.info("Vocabulary: %d tokens (incl. PAD+UNK)", vocab_size)

    # 5. Encode train sequences
    encoded_train: list[MultiLevelTokenSequence] = []
    for mlt in train_mlts:
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
        encoded_train.append(MultiLevelTokenSequence(
            ticker=mlt.ticker, interval=mlt.interval, level_sequences=level_sequences,
        ))

    # Encode test sequences
    test_mlts: list[MultiLevelTokenSequence] = []
    for ticker, level_words in test_raw:
        level_sequences: dict[int, TokenSequence] = {}
        for lvl, words in level_words.items():
            token_ids = vocabulary.encode_sequence(words)
            level_sequences[lvl] = TokenSequence(
                token_ids=token_ids,
                words=words,
                ticker=ticker,
                interval="1h",
                wavelet_level=lvl,
            )
        test_mlts.append(MultiLevelTokenSequence(
            ticker=ticker, interval="1h", level_sequences=level_sequences,
        ))

    # 6. Build sequence datasets
    asset_class_map = build_asset_class_map(common_tickers)
    train_dataset = build_sequence_dataset(encoded_train, vocabulary, CONTEXT_LENGTH, asset_class_map)
    test_dataset = build_sequence_dataset(test_mlts, vocabulary, CONTEXT_LENGTH, asset_class_map)

    X_train, y_train, _, _ = train_dataset.to_arrays()
    X_test, y_test, _, _ = test_dataset.to_arrays()

    train_levels = np.array([s.level for s in train_dataset.samples], dtype=np.int64)
    train_ac = np.array([s.asset_class_id for s in train_dataset.samples], dtype=np.int64)
    test_levels = np.array([s.level for s in test_dataset.samples], dtype=np.int64)
    test_ac = np.array([s.asset_class_id for s in test_dataset.samples], dtype=np.int64)

    X_train_full = np.column_stack([X_train, train_levels, train_ac])
    X_test_full = np.column_stack([X_test, test_levels, test_ac])

    logger.info("Dataset: %d train, %d test samples", len(X_train_full), len(X_test_full))

    # 7. Train WaveletGPT
    logger.info("Training WaveletGPT (vocab=%d, ctx=%d, embed=%d, layers=%d, epochs=%d)...",
                vocab_size, CONTEXT_LENGTH, EMBED_DIM, NUM_LAYERS, EPOCHS)

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
    )

    model.fit(X_train_full, y_train, X_val=X_test_full, y_val=y_test)

    # 8. Quick evaluation
    predicted = model.predict(X_test_full)
    token_acc = float(np.mean(predicted == y_test))
    logger.info("Test token accuracy: %.4f (%.1f%%)", token_acc, token_acc * 100)

    # 9. Save artifacts
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_DIR)
    vocabulary.save(OUTPUT_DIR / "vocabulary.json")

    elapsed = time.monotonic() - t0
    logger.info("Done in %.1fs. Saved to %s", elapsed, OUTPUT_DIR)
    logger.info("  model.pt: %s", OUTPUT_DIR / "model.pt")
    logger.info("  config.json: %s", OUTPUT_DIR / "config.json")
    logger.info("  vocabulary.json: %s", OUTPUT_DIR / "vocabulary.json")


if __name__ == "__main__":
    main()
