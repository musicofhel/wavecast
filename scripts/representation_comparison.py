#!/usr/bin/env python3
"""Phase 9: Input Representation Comparison.

Trains and evaluates 8 candidate input representations on the same
train/test split. All candidates use return-quantile targets (5 classes).

Candidates:
- Baseline: SAX on detail coefficients (current pipeline)
- A:  SAX on approximation coefficients
- B1: SAX on detail coefficient deltas
- B2: SAX on approximation coefficient deltas
- C1: Raw detail coefficients (continuous input)
- C2: Raw approximation coefficients (continuous input)
- C3: Raw detail coefficient deltas (continuous input)
- C4: Raw [approx + detail] (continuous input)

Usage:
    python scripts/representation_comparison.py [--quick]

    --quick: Use smaller model (embed=64, layers=3, epochs=20) for fast screening
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wavecast.core.types import (
    MultiLevelTokenSequence,
    TimeSeries,
    TokenSequence,
)
from wavecast.core.universe import DEFAULT_UNIVERSE
from wavecast.data.auxiliary_features import (
    N_AUX_FEATURES,
    compute_detail_auxiliary_features,
)
from wavecast.data.cache import ParquetCache
from wavecast.data.continuous_dataset import build_continuous_dataset
from wavecast.experiments.runner import SECTOR_ID_MAP
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.sax.bow import extract_words
from wavecast.sax.sax import sax_transform
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
)
from wavecast.tokenizer.dataset import build_return_target_dataset
from wavecast.tokenizer.vocabulary import SAXVocabulary
from wavecast.wavelets.dwt import decompose

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Config ---
TICKERS = [a.ticker for a in DEFAULT_UNIVERSE.assets]
INTERVAL = "1h"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
DETAIL_LEVELS = [1, 2, 5]
APPROX_LEVEL = 5  # DWT level for approximation
CONTEXT_LENGTH = 16
N_CLASSES = 5
PERCENTILES = [10.0, 30.0, 70.0, 90.0]
N_SEGMENTS = 512
ALPHABET_SIZE = 7
WORD_LENGTH = 4
WORD_STRIDE = 1
MAX_VOCAB_SIZE = 100
MIN_RETURN_THRESHOLD = 0.001  # 0.1% filter for economic directional accuracy
COST_BPS = 7.0  # 5bps spread + 2bps slippage round-trip
N_RANDOM_TRIALS = 100


@dataclass
class CandidateResult:
    """Evaluation results for a single candidate."""

    name: str
    n_train: int
    n_test: int
    n_valid: int
    quantile_accuracy: float
    econ_dir_accuracy: float  # filtered to |return| > threshold
    sharpe_raw: float
    sharpe_with_costs: float
    sharpe_percentile: float
    transition_accuracy: float
    train_time: float
    train_loss: float


# --- Data Loading ---


def load_and_split_prices(
    cache: ParquetCache,
) -> tuple[dict[str, TimeSeries], dict[str, TimeSeries], list[str]]:
    """Load prices and split into train/test periods."""
    train_prices: dict[str, TimeSeries] = {}
    test_prices: dict[str, TimeSeries] = {}

    for ticker in TICKERS:
        ts = cache.get(ticker, INTERVAL)
        if ts is None:
            continue
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
    return train_prices, test_prices, common


def build_asset_class_map() -> dict[str, int]:
    """Build ticker -> sector ID mapping."""
    ac_map = {}
    for a in DEFAULT_UNIVERSE.assets:
        if a.sector is not None:
            ac_map[a.ticker] = SECTOR_ID_MAP.get(a.sector.value, 0)
    return ac_map


# --- Return Computation ---


def compute_coeff_returns(n_coeffs: int, level: int, prices: NDArray) -> NDArray:
    """Compute forward returns at each coefficient position.

    Coefficient at index i covers bars [i * 2^level, (i+1) * 2^level).
    Return[i] = (price[(i+1)*span] - price[i*span]) / price[i*span].
    """
    span = 2 ** level
    returns = np.full(n_coeffs, np.nan)
    for i in range(n_coeffs):
        bar_start = i * span
        bar_end = bar_start + span
        if bar_end < len(prices) and prices[bar_start] > 0:
            returns[i] = (prices[bar_end] - prices[bar_start]) / prices[bar_start]
    return returns


# --- Evaluation ---


def evaluate_candidate(
    name: str,
    pred_labels: NDArray,
    proba: NDArray,
    actual_returns: NDArray,
    actual_labels: NDArray,
    valid_mask: NDArray,
    train_time: float,
    train_loss: float,
    n_train: int,
) -> CandidateResult:
    """Full evaluation battery for a candidate."""
    n_test = len(pred_labels)
    n_valid = int(valid_mask.sum())
    mid = N_CLASSES // 2

    # Predicted direction: UP(+1) for classes > mid, DOWN(-1) for < mid, FLAT(0)
    pred_dir = np.zeros(len(pred_labels), dtype=np.float64)
    pred_dir[pred_labels > mid] = 1.0
    pred_dir[pred_labels < mid] = -1.0

    actual_dir = np.sign(actual_returns)

    # --- Economic directional accuracy (filtered) ---
    filt = valid_mask & (np.abs(actual_returns) > MIN_RETURN_THRESHOLD)
    if filt.sum() > 0:
        # Only count non-flat predictions against non-flat returns
        has_pred = pred_dir[filt] != 0
        if has_pred.sum() > 0:
            econ_dir_acc = float(
                np.mean(pred_dir[filt][has_pred] == actual_dir[filt][has_pred])
            )
        else:
            econ_dir_acc = 0.5
    else:
        econ_dir_acc = 0.5

    # --- Quantile accuracy ---
    if n_valid > 0:
        q_acc = float(np.mean(pred_labels[valid_mask] == actual_labels[valid_mask]))
    else:
        q_acc = 0.0

    # --- Sharpe (raw and with costs) ---
    if n_valid > 0:
        pnl_raw = pred_dir[valid_mask] * actual_returns[valid_mask]
        pnl_raw = pnl_raw[~np.isnan(pnl_raw)]
        # Direction changes incur double cost
        dir_changes = np.abs(np.diff(pred_dir[valid_mask]))
        cost_per_step = np.zeros(n_valid)
        cost_per_step[1:] = dir_changes * (COST_BPS / 10000)
        # Also cost for initial position
        cost_per_step[0] = abs(pred_dir[valid_mask][0]) * (COST_BPS / 10000) if n_valid > 0 else 0
        pnl_net = pred_dir[valid_mask] * actual_returns[valid_mask] - cost_per_step
        pnl_net = pnl_net[~np.isnan(pnl_net)]

        sharpe_raw = (
            float(np.mean(pnl_raw) / np.std(pnl_raw) * np.sqrt(252 * 7))
            if len(pnl_raw) > 1 and np.std(pnl_raw) > 0
            else 0.0
        )
        sharpe_costs = (
            float(np.mean(pnl_net) / np.std(pnl_net) * np.sqrt(252 * 7))
            if len(pnl_net) > 1 and np.std(pnl_net) > 0
            else 0.0
        )
    else:
        sharpe_raw = 0.0
        sharpe_costs = 0.0

    # --- Sharpe vs random ---
    rng = np.random.default_rng(42)
    random_sharpes = []
    if n_valid > 0:
        valid_rets = actual_returns[valid_mask]
        valid_rets_clean = valid_rets[~np.isnan(valid_rets)]
        for _ in range(N_RANDOM_TRIALS):
            rand_dir = rng.choice([-1.0, 0.0, 1.0], size=len(valid_rets_clean))
            rand_pnl = rand_dir * valid_rets_clean
            if len(rand_pnl) > 1 and np.std(rand_pnl) > 0:
                random_sharpes.append(
                    float(np.mean(rand_pnl) / np.std(rand_pnl) * np.sqrt(252 * 7))
                )
    sharpe_pctile = (
        float(np.mean(np.array(random_sharpes) < sharpe_costs) * 100)
        if random_sharpes
        else 50.0
    )

    # --- Transition accuracy ---
    # Accuracy specifically on points where actual direction changed
    if n_valid > 10:
        valid_idx = np.where(valid_mask)[0]
        actual_dir_valid = actual_dir[valid_idx]
        pred_dir_valid = pred_dir[valid_idx]
        transitions = np.where(np.diff(np.sign(actual_dir_valid)) != 0)[0] + 1
        if len(transitions) > 0:
            trans_acc = float(
                np.mean(pred_dir_valid[transitions] == actual_dir_valid[transitions])
            )
        else:
            trans_acc = 0.5
    else:
        trans_acc = 0.5

    return CandidateResult(
        name=name,
        n_train=n_train,
        n_test=n_test,
        n_valid=n_valid,
        quantile_accuracy=q_acc,
        econ_dir_accuracy=econ_dir_acc,
        sharpe_raw=sharpe_raw,
        sharpe_with_costs=sharpe_costs,
        sharpe_percentile=sharpe_pctile,
        transition_accuracy=trans_acc,
        train_time=train_time,
        train_loss=train_loss,
    )


# --- SAX Pipeline (for Baseline, A, B1, B2) ---


def run_sax_candidate(
    name: str,
    tickers: list[str],
    train_prices: dict[str, TimeSeries],
    test_prices: dict[str, TimeSeries],
    ac_map: dict[str, int],
    coeff_fn,
    levels: list[int],
    model_kwargs: dict,
) -> CandidateResult:
    """Run a SAX-based candidate end-to-end."""
    logger.info("Running %s ...", name)

    train_words_all: list[list[str]] = []
    train_seqs_raw: list[tuple[str, dict[int, list[str]]]] = []
    test_seqs_raw: list[tuple[str, dict[int, list[str]]]] = []
    train_coeffs_map: dict[tuple[str, int], int] = {}
    train_symbols_map: dict[tuple[str, int], int] = {}
    test_coeffs_map: dict[tuple[str, int], int] = {}
    test_symbols_map: dict[tuple[str, int], int] = {}

    # Per-ticker returns (for evaluation)
    train_returns_map: dict[tuple[str, int], NDArray] = {}
    test_returns_map: dict[tuple[str, int], NDArray] = {}

    for ticker in tickers:
        train_decomp = decompose(train_prices[ticker], level=5)
        test_decomp = decompose(test_prices[ticker], level=5)
        train_lw: dict[int, list[str]] = {}
        test_lw: dict[int, list[str]] = {}

        for lvl in levels:
            tc = coeff_fn(train_decomp, lvl)
            if len(tc) < 2:
                continue
            ns = min(N_SEGMENTS, len(tc))
            tsax = sax_transform(tc, ns, ALPHABET_SIZE)
            tw = extract_words(tsax.symbols, WORD_LENGTH, WORD_STRIDE)
            train_lw[lvl] = tw
            train_words_all.append(tw)
            train_coeffs_map[(ticker, lvl)] = len(tc)
            train_symbols_map[(ticker, lvl)] = len(tsax.symbols)
            train_returns_map[(ticker, lvl)] = compute_coeff_returns(
                len(tc), lvl, train_prices[ticker].values
            )

            ec = coeff_fn(test_decomp, lvl)
            if len(ec) < 2:
                continue
            ns = min(N_SEGMENTS, len(ec))
            esax = sax_transform(ec, ns, ALPHABET_SIZE)
            ew = extract_words(esax.symbols, WORD_LENGTH, WORD_STRIDE)
            test_lw[lvl] = ew
            test_coeffs_map[(ticker, lvl)] = len(ec)
            test_symbols_map[(ticker, lvl)] = len(esax.symbols)
            test_returns_map[(ticker, lvl)] = compute_coeff_returns(
                len(ec), lvl, test_prices[ticker].values
            )

        train_seqs_raw.append((ticker, train_lw))
        test_seqs_raw.append((ticker, test_lw))

    if not train_words_all:
        logger.warning("%s: no training data", name)
        return _empty_result(name)

    # Vocabulary
    vocab = SAXVocabulary.from_corpus(
        train_words_all, min_freq=1, max_size=MAX_VOCAB_SIZE
    )

    # Encode into MLTs
    def encode_seqs(seqs_raw):
        mlts = []
        for ticker, lw in seqs_raw:
            ls = {}
            for lvl, words in lw.items():
                ids = vocab.encode_sequence(words)
                ls[lvl] = TokenSequence(
                    token_ids=ids, words=words, ticker=ticker,
                    interval=INTERVAL, wavelet_level=lvl,
                )
            mlts.append(MultiLevelTokenSequence(
                ticker=ticker, interval=INTERVAL, level_sequences=ls,
            ))
        return mlts

    train_mlts = encode_seqs(train_seqs_raw)
    test_mlts = encode_seqs(test_seqs_raw)

    # Build datasets
    train_ds = build_return_target_dataset(
        train_mlts, vocab, CONTEXT_LENGTH, ac_map,
        n_coeffs_map=train_coeffs_map, n_symbols_map=train_symbols_map,
    )
    test_ds = build_return_target_dataset(
        test_mlts, vocab, CONTEXT_LENGTH, ac_map,
        n_coeffs_map=test_coeffs_map, n_symbols_map=test_symbols_map,
    )

    if len(train_ds.samples) == 0 or len(test_ds.samples) == 0:
        logger.warning("%s: empty dataset", name)
        return _empty_result(name)

    # Compute returns for each sample
    train_rets, train_valid, train_lvls = _compute_ds_returns_sax(
        train_ds, train_mlts, train_prices, train_coeffs_map, train_symbols_map
    )
    test_rets, test_valid, test_lvls = _compute_ds_returns_sax(
        test_ds, test_mlts, test_prices, test_coeffs_map, test_symbols_map
    )

    # Quantile boundaries from training
    boundaries = compute_quantile_boundaries(
        train_rets, train_lvls, train_valid, PERCENTILES, per_level=True
    )
    y_train = assign_quantile_labels(train_rets, train_lvls, boundaries)
    y_test = assign_quantile_labels(test_rets, test_lvls, boundaries)

    # Class weights
    valid_train_labels = y_train[train_valid]
    counts = np.bincount(valid_train_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    # Build X arrays
    X_train, _, _, _ = train_ds.to_arrays()
    X_test, _, _, _ = test_ds.to_arrays()
    train_lvl_arr = np.array([s.level for s in train_ds.samples], dtype=np.int64)
    train_ac_arr = np.array([s.asset_class_id for s in train_ds.samples], dtype=np.int64)
    test_lvl_arr = np.array([s.level for s in test_ds.samples], dtype=np.int64)
    test_ac_arr = np.array([s.asset_class_id for s in test_ds.samples], dtype=np.int64)
    X_train_full = np.column_stack([X_train, train_lvl_arr, train_ac_arr])
    X_test_full = np.column_stack([X_test, test_lvl_arr, test_ac_arr])

    # Train
    model = WaveletGPT(
        vocab_size=vocab.size,
        context_length=CONTEXT_LENGTH,
        task="return_quantile",
        n_output_classes=N_CLASSES,
        class_weights=class_weights,
        **model_kwargs,
    )
    t0 = time.time()
    metrics = model.fit(
        X_train_full, y_train.astype(np.float64),
        X_val=X_test_full, y_val=y_test.astype(np.float64),
    )
    train_time = time.time() - t0

    # Predict
    predicted = model.predict(X_test_full).astype(np.int64)
    proba = model.predict_proba(X_test_full)

    return evaluate_candidate(
        name, predicted, proba, test_rets, y_test, test_valid,
        train_time, metrics["train_loss"], len(train_ds.samples),
    )


def _compute_ds_returns_sax(ds, mlts, prices_dict, coeffs_map, symbols_map):
    """Compute per-sample returns for a SAX dataset."""
    from wavecast.targets.returns import compute_sample_returns

    tp = np.array([s.token_position for s in ds.samples], dtype=np.int64)
    lvls = np.array([s.level for s in ds.samples], dtype=np.int64)
    nc = np.array([s.n_coeffs for s in ds.samples], dtype=np.int64)
    ns = np.array([s.n_symbols for s in ds.samples], dtype=np.int64)

    tickers_list: list[str] = []
    for mlt in mlts:
        for _, seq in mlt.level_sequences.items():
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
            r, v = compute_sample_returns(
                tp[m], lvls[m], prices_dict[t].values, nc[m], ns[m]
            )
            rets[m] = r
            valid[m] = v

    return rets, valid, lvls


# --- Continuous Pipeline (for C1, C2, C3, C4) ---


def run_continuous_candidate(
    name: str,
    tickers: list[str],
    train_prices: dict[str, TimeSeries],
    test_prices: dict[str, TimeSeries],
    ac_map: dict[str, int],
    coeff_fn,
    levels: list[int],
    model_kwargs: dict,
    aux_fn=None,
    n_aux_features: int = 0,
) -> CandidateResult:
    """Run a continuous-input candidate end-to-end.

    Args:
        aux_fn: Optional function(decomposition, level) -> NDArray of shape (n, n_features).
            Must return features aligned with coeff_fn output (same length).
        n_aux_features: Number of auxiliary feature channels (0 = no aux).
    """
    logger.info("Running %s ...", name)

    train_series: dict[tuple[str, int], NDArray] = {}
    test_series: dict[tuple[str, int], NDArray] = {}
    train_aux_series: dict[tuple[str, int], NDArray] = {}
    test_aux_series: dict[tuple[str, int], NDArray] = {}

    for ticker in tickers:
        train_decomp = decompose(train_prices[ticker], level=5)
        test_decomp = decompose(test_prices[ticker], level=5)
        for lvl in levels:
            tc = coeff_fn(train_decomp, lvl)
            if len(tc) > CONTEXT_LENGTH:
                train_series[(ticker, lvl)] = tc
                if aux_fn is not None:
                    train_aux_series[(ticker, lvl)] = aux_fn(train_decomp, lvl)
            ec = coeff_fn(test_decomp, lvl)
            if len(ec) > CONTEXT_LENGTH:
                test_series[(ticker, lvl)] = ec
                if aux_fn is not None:
                    test_aux_series[(ticker, lvl)] = aux_fn(test_decomp, lvl)

    if not train_series:
        logger.warning("%s: no training data", name)
        return _empty_result(name)

    # Build continuous datasets
    train_ds = build_continuous_dataset(
        train_series, CONTEXT_LENGTH, ac_map, normalize=True
    )
    test_ds = build_continuous_dataset(
        test_series, CONTEXT_LENGTH, ac_map, normalize=True
    )

    if len(train_ds.windows) == 0 or len(test_ds.windows) == 0:
        logger.warning("%s: empty dataset", name)
        return _empty_result(name)

    # Compute returns for each window
    def compute_window_returns(ds, prices_dict, series_dict):
        rets = np.full(len(ds.windows), np.nan)
        valid = np.zeros(len(ds.windows), dtype=np.bool_)
        for i, w in enumerate(ds.windows):
            key = (w.ticker, w.level)
            if key not in series_dict or w.ticker not in prices_dict:
                continue
            coeffs_len = len(series_dict[key])
            target_pos = w.token_position  # position just past window
            if target_pos >= coeffs_len:
                continue
            span = 2 ** w.level
            bar_start = target_pos * span
            bar_end = bar_start + span
            pv = prices_dict[w.ticker].values
            if bar_end < len(pv) and pv[bar_start] > 0:
                rets[i] = (pv[bar_end] - pv[bar_start]) / pv[bar_start]
                valid[i] = True
        return rets, valid

    train_rets, train_valid = compute_window_returns(
        train_ds, train_prices, train_series
    )
    test_rets, test_valid = compute_window_returns(
        test_ds, test_prices, test_series
    )
    train_lvls = np.array([w.level for w in train_ds.windows], dtype=np.int64)
    test_lvls = np.array([w.level for w in test_ds.windows], dtype=np.int64)

    # Quantile boundaries
    boundaries = compute_quantile_boundaries(
        train_rets, train_lvls, train_valid, PERCENTILES, per_level=True
    )
    y_train = assign_quantile_labels(train_rets, train_lvls, boundaries)
    y_test = assign_quantile_labels(test_rets, test_lvls, boundaries)

    # Class weights
    valid_train_labels = y_train[train_valid]
    counts = np.bincount(valid_train_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    # Build X arrays
    ctx_arr, lvl_arr, ac_arr = train_ds.to_arrays()
    ctx_arr_t, lvl_arr_t, ac_arr_t = test_ds.to_arrays()

    if n_aux_features > 0 and train_aux_series:
        train_aux_win = _build_aux_windows(
            train_aux_series, CONTEXT_LENGTH, n_aux_features
        )
        test_aux_win = _build_aux_windows(
            test_aux_series, CONTEXT_LENGTH, n_aux_features
        )
        train_aux_flat = train_aux_win.reshape(len(train_aux_win), -1)
        test_aux_flat = test_aux_win.reshape(len(test_aux_win), -1)
        X_train_full = np.column_stack([ctx_arr, train_aux_flat, lvl_arr, ac_arr])
        X_test_full = np.column_stack([ctx_arr_t, test_aux_flat, lvl_arr_t, ac_arr_t])
    else:
        X_train_full = np.column_stack([ctx_arr, lvl_arr, ac_arr])
        X_test_full = np.column_stack([ctx_arr_t, lvl_arr_t, ac_arr_t])

    # Train (continuous input mode)
    model = WaveletGPT(
        vocab_size=1,  # unused for continuous
        context_length=CONTEXT_LENGTH,
        task="return_quantile",
        n_output_classes=N_CLASSES,
        class_weights=class_weights,
        input_mode="continuous",
        n_aux_features=n_aux_features,
        **model_kwargs,
    )
    t0 = time.time()
    metrics = model.fit(
        X_train_full, y_train.astype(np.float64),
        X_val=X_test_full, y_val=y_test.astype(np.float64),
    )
    train_time = time.time() - t0

    # Predict
    predicted = model.predict(X_test_full).astype(np.int64)
    proba = model.predict_proba(X_test_full)

    return evaluate_candidate(
        name, predicted, proba, test_rets, y_test, test_valid,
        train_time, metrics["train_loss"], len(train_ds.windows),
    )


def _build_aux_windows(
    aux_series: dict[tuple[str, int], NDArray],
    context_length: int,
    n_aux_features: int,
) -> NDArray:
    """Build sliding windows from auxiliary feature series.

    Must use same (ticker, level) keys and same-length series as the
    coefficient series used in build_continuous_dataset, to ensure alignment.
    Iterates in sorted order matching build_continuous_dataset's sort.
    """
    windows: list[NDArray] = []
    for (_ticker, _level), aux in sorted(aux_series.items()):
        if len(aux) <= context_length:
            continue
        for i in range(len(aux) - context_length):
            windows.append(aux[i : i + context_length])
    if not windows:
        return np.empty((0, context_length, n_aux_features), dtype=np.float64)
    return np.array(windows, dtype=np.float64)


def _empty_result(name: str) -> CandidateResult:
    return CandidateResult(
        name=name, n_train=0, n_test=0, n_valid=0,
        quantile_accuracy=0.0, econ_dir_accuracy=0.5,
        sharpe_raw=0.0, sharpe_with_costs=0.0, sharpe_percentile=50.0,
        transition_accuracy=0.5, train_time=0.0, train_loss=0.0,
    )


# --- Coefficient Extractors ---


def detail_coeffs(decomp, lvl):
    return decomp.detail_at_level(lvl)


def approx_coeffs(decomp, _lvl):
    return decomp.approximation


def detail_delta_coeffs(decomp, lvl):
    c = decomp.detail_at_level(lvl)
    return np.diff(c) if len(c) > 1 else c


def approx_delta_coeffs(decomp, _lvl):
    c = decomp.approximation
    return np.diff(c) if len(c) > 1 else c


def approx_and_detail_coeffs(decomp, lvl):
    """Concatenate approximation and detail for a given level.
    For level 5: approx + detail_5. For others: just detail."""
    if lvl == APPROX_LEVEL:
        return np.concatenate([decomp.approximation, decomp.detail_at_level(lvl)])
    return decomp.detail_at_level(lvl)


# --- Main ---


def main() -> None:
    quick = "--quick" in sys.argv
    only = None
    for arg in sys.argv[1:]:
        if arg.startswith("--only="):
            only = arg.split("=", 1)[1]

    if quick:
        model_kwargs = {
            "embed_dim": 64, "num_heads": 4, "num_layers": 3,
            "dropout": 0.1, "epochs": 20, "batch_size": 64,
            "learning_rate": 0.0005, "patience": 10,
        }
        logger.info("QUICK MODE: small model, 20 epochs")
    else:
        model_kwargs = {
            "embed_dim": 128, "num_heads": 4, "num_layers": 6,
            "dropout": 0.2, "epochs": 80, "batch_size": 64,
            "learning_rate": 0.0005, "patience": 15,
        }

    # Load data
    cache = ParquetCache(Path.home() / ".wavecast" / "cache")
    train_prices, test_prices, tickers = load_and_split_prices(cache)
    logger.info("Loaded %d tickers", len(tickers))

    ac_map = build_asset_class_map()
    results: list[CandidateResult] = []

    def should_run(name: str) -> bool:
        return only is None or only.lower() in name.lower()

    # --- Baseline: SAX on detail coefficients ---
    if should_run("Baseline"):
        results.append(run_sax_candidate(
            "Baseline", tickers, train_prices, test_prices, ac_map,
            detail_coeffs, DETAIL_LEVELS, model_kwargs,
        ))

    # --- A: SAX on approximation coefficients ---
    if should_run("A:"):
        results.append(run_sax_candidate(
            "A: Approx SAX", tickers, train_prices, test_prices, ac_map,
            approx_coeffs, [APPROX_LEVEL], model_kwargs,
        ))

    # --- B1: SAX on detail coefficient deltas ---
    if should_run("B1"):
        results.append(run_sax_candidate(
            "B1: Det Delta", tickers, train_prices, test_prices, ac_map,
            detail_delta_coeffs, DETAIL_LEVELS, model_kwargs,
        ))

    # --- B2: SAX on approximation coefficient deltas ---
    if should_run("B2"):
        results.append(run_sax_candidate(
            "B2: Apx Delta", tickers, train_prices, test_prices, ac_map,
            approx_delta_coeffs, [APPROX_LEVEL], model_kwargs,
        ))

    # --- C1: Raw detail coefficients ---
    if should_run("C1"):
        results.append(run_continuous_candidate(
            "C1: Raw Det", tickers, train_prices, test_prices, ac_map,
            detail_coeffs, DETAIL_LEVELS, model_kwargs,
        ))

    # --- C2: Raw approximation coefficients ---
    if should_run("C2"):
        results.append(run_continuous_candidate(
            "C2: Raw Apx", tickers, train_prices, test_prices, ac_map,
            approx_coeffs, [APPROX_LEVEL], model_kwargs,
        ))

    # --- C3: Raw detail coefficient deltas ---
    if should_run("C3"):
        results.append(run_continuous_candidate(
            "C3: Raw Delta", tickers, train_prices, test_prices, ac_map,
            detail_delta_coeffs, DETAIL_LEVELS, model_kwargs,
        ))

    # --- C4: Raw [approx + detail] ---
    if should_run("C4"):
        results.append(run_continuous_candidate(
            "C4: Raw A+D", tickers, train_prices, test_prices, ac_map,
            approx_and_detail_coeffs, DETAIL_LEVELS, model_kwargs,
        ))

    # --- D1: Raw detail deltas + auxiliary features ---
    if should_run("D1"):
        def detail_delta_aux_fn(decomp, lvl):
            detail = decomp.detail_at_level(lvl)
            approx = decomp.approximation
            return compute_detail_auxiliary_features(detail, approx)

        results.append(run_continuous_candidate(
            "D1: Delta+Aux", tickers, train_prices, test_prices, ac_map,
            detail_delta_coeffs, DETAIL_LEVELS, model_kwargs,
            aux_fn=detail_delta_aux_fn, n_aux_features=N_AUX_FEATURES,
        ))

    # --- Print Comparison Table ---
    print("\n" + "=" * 110)
    print("PHASE 9: INPUT REPRESENTATION COMPARISON")
    print("=" * 110)
    print(
        f"{'Candidate':<16} {'Econ Dir':>8} {'Q.Acc':>6} "
        f"{'Sharpe':>7} {'S+Cost':>7} {'vs Rnd':>7} "
        f"{'Trans':>6} {'N_train':>7} {'N_test':>7} {'Time':>6}"
    )
    print("-" * 110)

    for r in results:
        verdict = ""
        if r.econ_dir_accuracy > 0.55:
            verdict = " PASS"
        elif r.econ_dir_accuracy > 0.52:
            verdict = " MARG"
        elif r.econ_dir_accuracy < 0.48:
            verdict = " FAIL"

        print(
            f"{r.name:<16} {r.econ_dir_accuracy:>7.1%} {r.quantile_accuracy:>6.1%} "
            f"{r.sharpe_raw:>+7.3f} {r.sharpe_with_costs:>+7.3f} {r.sharpe_percentile:>6.1f}% "
            f"{r.transition_accuracy:>5.1%} {r.n_train:>7d} {r.n_test:>7d} {r.train_time:>5.1f}s"
            f"{verdict}"
        )

    print("-" * 110)
    print("\nPass/Fail criteria:")
    print("  PASS: Econ Dir > 55%, Sharpe(+cost) > 0.5, vs Random > 95th, Transition > 50%")
    print("  MARG: Econ Dir 52-55%")
    print("  FAIL: Econ Dir < 48%")

    # --- Decision ---
    best = max(results, key=lambda r: r.econ_dir_accuracy)
    print(f"\nBest candidate: {best.name} (Econ Dir = {best.econ_dir_accuracy:.1%})")

    if best.econ_dir_accuracy > 0.55:
        print("VERDICT: Signal found — proceed with this representation")
    elif best.econ_dir_accuracy > 0.52:
        print("VERDICT: Marginal signal — combine candidates or add features")
    else:
        print("VERDICT: No signal — wavelet coefficients don't encode tradeable direction")
        print("NEXT: Exploit what the model IS good at (volatility, regime detection)")
        print("  or try entirely different features (order flow, cross-asset momentum)")

    print("=" * 110)

    # Save results
    results_dir = Path.home() / ".wavecast" / "audit"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_json = [
        {
            "name": r.name,
            "n_train": r.n_train,
            "n_test": r.n_test,
            "n_valid": r.n_valid,
            "quantile_accuracy": r.quantile_accuracy,
            "econ_dir_accuracy": r.econ_dir_accuracy,
            "sharpe_raw": r.sharpe_raw,
            "sharpe_with_costs": r.sharpe_with_costs,
            "sharpe_percentile": r.sharpe_percentile,
            "transition_accuracy": r.transition_accuracy,
            "train_time": r.train_time,
            "train_loss": r.train_loss,
        }
        for r in results
    ]
    with open(results_dir / "representation_comparison.json", "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info("Results saved to %s", results_dir / "representation_comparison.json")


if __name__ == "__main__":
    main()
