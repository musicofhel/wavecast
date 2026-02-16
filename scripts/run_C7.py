"""C7: Ensemble comparison — P1 vs P2 vs combined.

Research Q7: Does combining Pipeline 1 + Pipeline 2 beat either alone?

Method (4 configs, or 3 if P1 fails):
1. P1 alone: WaveletLSTM + XGBoost ensemble -> directional accuracy
2. P2 alone: WaveletGPT (optimal config, level 0) -> directional accuracy
3. P2 multi-level: all optimal levels -> token probs -> meta-classifier -> directional accuracy
4. Combined: P1 feature vector + P2 token probabilities -> XGBoost meta-learner -> directional accuracy

Also: Pearson correlation between P1 and P2 error vectors.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import pearsonr

# Ensure project source is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from wavecast.core.config import SAXConfig, WaveCastConfig, WaveletConfig
from wavecast.core.types import TimeSeries, WaveletDecomposition, AssetClass
from wavecast.data.preprocessing import label_returns, log_returns
from wavecast.evaluation.metrics import directional_accuracy, rmse, mae
from wavecast.features.pipeline import FeaturePipeline
from wavecast.models.ensemble import EnsembleModel
from wavecast.models.gradient_boost import GradientBoostModel
from wavecast.models.wavelet_lstm import WaveletLSTM
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.sax.sax import sax_transform
from wavecast.sax.bow import extract_words
from wavecast.tokenizer.vocabulary import SAXVocabulary
from wavecast.tokenizer.tokenizer import WaveletSAXTokenizer
from wavecast.tokenizer.dataset import build_sequence_dataset
from wavecast.wavelets.dwt import decompose
from wavecast.core.universe import PHASE3_UNIVERSE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
CACHE_DIR = Path.home() / ".wavecast" / "cache"
EXPERIMENTS_DIR = Path.home() / ".wavecast" / "experiments"
C7_OUTPUT = EXPERIMENTS_DIR / "C7_ensemble.json"
C5_CONTEXT = EXPERIMENTS_DIR / "C5_context.json"

# Walk-forward split dates (same as all C experiments)
TRAIN_START = "2021-02-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2024-12-31"

# Assets for C7 comparison (P1 evaluated on these 3)
C7_ASSETS = ["AAPL", "SPY", "GLD"]

# Full Phase 3 universe for P2 training (20 assets, matching other C experiments)
P2_TRAIN_UNIVERSE = PHASE3_UNIVERSE

# Asset class mapping — built from Phase 3 universe
ASSET_CLASS_MAP = {
    asset.ticker: asset.asset_class for asset in PHASE3_UNIVERSE.assets
}
ASSET_CLASS_IDS = {cls: i for i, cls in enumerate(AssetClass)}


# ── Data Loading ──────────────────────────────────────────────────────────

def load_ts(ticker: str, interval: str = "1d") -> TimeSeries:
    """Load close prices from OHLCV parquet cache.

    Args:
        ticker: Asset ticker symbol.
        interval: Data interval ('1d' for daily, '1h' for hourly).
    """
    ohlcv_path = CACHE_DIR / f"{ticker}_{interval}_ohlcv.parquet"
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"No cached data for {ticker} {interval}: {ohlcv_path}")

    df = pd.read_parquet(ohlcv_path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Remove timezone info for consistent datetime comparison
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    values = df["close"].to_numpy(dtype=np.float64)
    timestamps = df["timestamp"].to_numpy(dtype="datetime64[ns]")

    return TimeSeries(
        values=values,
        timestamps=timestamps,
        ticker=ticker,
        interval=interval,
        column="close",
    )


# Backward compat alias
def load_daily_ts(ticker: str) -> TimeSeries:
    return load_ts(ticker, interval="1d")


def split_train_test(ts: TimeSeries) -> tuple[TimeSeries, TimeSeries]:
    """Split TimeSeries into train (2021-02-01 to 2023-12-31) and test (2024-01-01 to 2024-12-31)."""
    train_mask = (
        (ts.timestamps >= np.datetime64(TRAIN_START))
        & (ts.timestamps <= np.datetime64(TRAIN_END))
    )
    test_mask = (
        (ts.timestamps >= np.datetime64(TEST_START))
        & (ts.timestamps <= np.datetime64(TEST_END))
    )

    train_ts = TimeSeries(
        values=ts.values[train_mask],
        timestamps=ts.timestamps[train_mask],
        ticker=ts.ticker,
        interval=ts.interval,
        column=ts.column,
    )
    test_ts = TimeSeries(
        values=ts.values[test_mask],
        timestamps=ts.timestamps[test_mask],
        ticker=ts.ticker,
        interval=ts.interval,
        column=ts.column,
    )

    return train_ts, test_ts


# ── Pipeline 1: Feature-Based Forecasting ─────────────────────────────────

def run_p1_for_asset(
    ticker: str,
    train_ts: TimeSeries,
    test_ts: TimeSeries,
    cfg: WaveCastConfig,
) -> dict:
    """Run Pipeline 1 (WaveletLSTM + XGBoost ensemble) on a single asset.

    Returns dict with predictions, actual values, directional accuracy, etc.
    """
    logger.info(f"[P1] Running Pipeline 1 for {ticker}...")

    # Combine train+test for full decomposition (features need context)
    full_values = np.concatenate([train_ts.values, test_ts.values])
    full_timestamps = np.concatenate([train_ts.timestamps, test_ts.timestamps])
    full_ts = TimeSeries(
        values=full_values,
        timestamps=full_timestamps,
        ticker=ticker,
        interval="1d",
        column="close",
    )

    # Decompose full series
    decomp = decompose(full_ts, wavelet=cfg.wavelet.wavelet, level=cfg.wavelet.level)

    # Build feature matrix using rolling windows
    fp = FeaturePipeline()
    window = 50
    horizon = 1

    X, y = fp.build_feature_matrix(
        ts=full_ts,
        decomp=decomp,
        window=window,
        horizon=horizon,
        market_window=20,
        sax_config=cfg.sax,
    )

    if len(X) < 50:
        raise ValueError(f"Too few samples for {ticker}: {len(X)}")

    # Split: features built from full series, split by index corresponding to dates
    n_train = len(train_ts.values) - window - horizon + 1
    n_train = max(0, min(n_train, len(X)))

    if n_train < 30:
        raise ValueError(f"Too few training samples for {ticker}: {n_train}")

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    if len(X_test) < 10:
        raise ValueError(f"Too few test samples for {ticker}: {len(X_test)}")

    # Handle NaN/inf in features
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Validation split from training data (80/20)
    val_split = int(len(X_train) * 0.8)
    X_train_fit = X_train[:val_split]
    y_train_fit = y_train[:val_split]
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]

    logger.info(f"[P1] {ticker}: train={len(X_train_fit)}, val={len(X_val)}, test={len(X_test)}, features={X.shape[1]}")

    # Adjust LSTM branch sizes to match feature vector
    n_features = X.shape[1]
    # Default branches: [50, 25, 13, 7, 4] = 99
    # If features < 99, scale down branches proportionally
    default_branches = [50, 25, 13, 7, 4]
    total_default = sum(default_branches)
    if n_features < total_default:
        # Scale branches to fit within available features
        scale = n_features / total_default
        branches = [max(1, int(b * scale)) for b in default_branches]
        # Ensure total doesn't exceed n_features
        while sum(branches) > n_features:
            branches[-1] = max(1, branches[-1] - 1)
    else:
        branches = default_branches

    # Train ensemble
    xgb_model = GradientBoostModel(n_estimators=200, max_depth=6, learning_rate=0.1)
    lstm_model = WaveletLSTM(
        branch_input_sizes=branches,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        epochs=50,  # Reduced for speed
        batch_size=32,
        learning_rate=0.001,
        patience=10,
    )

    # Fit XGBoost
    xgb_model.fit(X_train_fit, y_train_fit, X_val, y_val)
    xgb_pred = xgb_model.predict(X_test)

    # Fit LSTM
    lstm_model.fit(X_train_fit, y_train_fit, X_val, y_val)
    lstm_pred = lstm_model.predict(X_test)

    # Ensemble: average of XGBoost and LSTM
    ensemble_pred = 0.5 * xgb_pred + 0.5 * lstm_pred

    # Metrics
    da_xgb = directional_accuracy(y_test, xgb_pred)
    da_lstm = directional_accuracy(y_test, lstm_pred)
    da_ensemble = directional_accuracy(y_test, ensemble_pred)

    return {
        "ticker": ticker,
        "n_train": len(X_train_fit),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_features": n_features,
        "branch_sizes": branches,
        "xgb_da": da_xgb,
        "lstm_da": da_lstm,
        "ensemble_da": da_ensemble,
        "xgb_rmse": rmse(y_test, xgb_pred),
        "lstm_rmse": rmse(y_test, lstm_pred),
        "ensemble_rmse": rmse(y_test, ensemble_pred),
        "y_test": y_test,
        "ensemble_pred": ensemble_pred,
        "p1_errors": y_test - ensemble_pred,  # Error vector for correlation
    }


# ── Pipeline 2: Token Prediction ──────────────────────────────────────────

def load_universe_train_data(
    sax_config: SAXConfig,
    wavelet: str,
    decomp_level: int,
    interval: str = "1h",
) -> tuple[list[list[str]], dict[str, WaveletDecomposition], dict[str, int]]:
    """Load and decompose all Phase 3 universe assets for P2 training.

    Args:
        interval: Data interval to load ('1h' for hourly, '1d' for daily).
                  C experiments use '1h' for more data points per asset.
    """
    all_word_sequences: list[list[str]] = []
    train_decomps: dict[str, WaveletDecomposition] = {}
    asset_class_ids_map: dict[str, int] = {}

    min_bars = 500 if interval == "1h" else 100

    for asset in P2_TRAIN_UNIVERSE.assets:
        ticker = asset.ticker
        ohlcv_path = CACHE_DIR / f"{ticker}_{interval}_ohlcv.parquet"
        if not ohlcv_path.exists():
            logger.warning(f"[P2] No cached {interval} data for {ticker}, skipping")
            continue

        df = pd.read_parquet(ohlcv_path)
        df = df.sort_values("timestamp").reset_index(drop=True)
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        mask = (df["timestamp"] >= pd.Timestamp(TRAIN_START)) & (df["timestamp"] <= pd.Timestamp(TRAIN_END))
        df_train = df[mask]
        if len(df_train) < min_bars:
            logger.warning(f"[P2] {ticker}: only {len(df_train)} training {interval} bars, skipping")
            continue

        values = df_train["close"].to_numpy(dtype=np.float64)
        timestamps = df_train["timestamp"].to_numpy(dtype="datetime64[ns]")
        ts = TimeSeries(values=values, timestamps=timestamps, ticker=ticker, interval=interval, column="close")

        d = decompose(ts, wavelet=wavelet, level=decomp_level)
        d = WaveletDecomposition(
            coefficients=d.coefficients, wavelet=d.wavelet,
            level=d.level, original_length=d.original_length, ticker=ticker,
        )
        train_decomps[ticker] = d
        asset_class_ids_map[ticker] = ASSET_CLASS_IDS.get(asset.asset_class, 0)

        for lvl in range(1, d.level + 1):
            coeffs = d.detail_at_level(lvl)
            if len(coeffs) < 2:
                continue
            n_seg = min(sax_config.n_segments, len(coeffs))
            sax_rep = sax_transform(coeffs, n_seg, sax_config.alphabet_size)
            words = extract_words(sax_rep.symbols, sax_config.word_length, sax_config.word_stride)
            all_word_sequences.append(words)

    logger.info(f"[P2] Loaded {len(train_decomps)} assets for training")
    return all_word_sequences, train_decomps, asset_class_ids_map


def run_p2_for_assets(
    eval_assets: list[str],
    test_ts_map: dict[str, TimeSeries],
    optimal_config: dict,
    p2_test_ts_map: dict[str, TimeSeries] | None = None,
) -> dict:
    """Run Pipeline 2 (WaveletGPT) trained on full Phase 3 universe, evaluated on target assets.

    Uses optimal config from C1-C5 experiments.
    p2_test_ts_map: If provided, use these TimeSeries for P2 evaluation (e.g., hourly data).
                    Falls back to test_ts_map if not provided.
    Returns dict with per-asset directional accuracy and error vectors.
    """
    logger.info("[P2] Running Pipeline 2 with optimal config (full universe training)...")

    sax_config = SAXConfig(
        n_segments=optimal_config.get("n_segments", 20),
        alphabet_size=optimal_config.get("alphabet_size", 8),
        word_length=optimal_config.get("word_length", 4),
        word_stride=1,
    )
    context_length = optimal_config.get("context_length", 32)
    vocab_size = optimal_config.get("vocab_size", 500)
    embed_dim = optimal_config.get("embed_dim", 64)
    num_heads = optimal_config.get("num_heads", 4)
    num_layers = optimal_config.get("num_layers", 3)
    dropout = optimal_config.get("dropout", 0.1)
    wavelet = optimal_config.get("wavelet", "db4")
    decomp_level = optimal_config.get("decomp_level", 5)

    interval = optimal_config.get("interval", "1h")

    # Step 1: Load full universe training data (hourly for C experiments)
    all_word_sequences, train_decomps, asset_class_ids_map = load_universe_train_data(
        sax_config, wavelet, decomp_level, interval=interval
    )

    vocab = SAXVocabulary.from_corpus(all_word_sequences, min_freq=2, max_size=vocab_size)
    logger.info(f"[P2] Vocabulary size: {vocab.size} (from {len(train_decomps)} assets, interval={interval})")

    # Step 2: Tokenize all training decompositions
    tokenizer = WaveletSAXTokenizer(vocab, sax_config)
    train_token_sequences = []
    for ticker in train_decomps:
        token_seq = tokenizer.tokenize(train_decomps[ticker])
        train_token_sequences.append(token_seq)

    # Step 3: Build training dataset from full universe
    dataset = build_sequence_dataset(
        train_token_sequences, vocab,
        context_length=context_length, asset_class_map=asset_class_ids_map,
    )
    contexts, targets, levels, asset_classes = dataset.to_arrays()

    if len(targets) == 0:
        logger.error("[P2] No training samples generated")
        return {"failed": True, "reason": "No training samples"}

    X_all = np.column_stack([contexts, levels.reshape(-1, 1), asset_classes.reshape(-1, 1)])
    y_all = targets

    # 80/20 train/val split
    n_train = int(len(y_all) * 0.8)
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_val = X_all[n_train:]
    y_val = y_all[n_train:]

    logger.info(f"[P2] Training samples: {len(y_train)}, validation: {len(y_val)}")

    # Step 4: Train WaveletGPT
    model = WaveletGPT(
        vocab_size=vocab.size, context_length=context_length,
        embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
        dropout=dropout, epochs=80, batch_size=64, learning_rate=0.0003, patience=15,
    )
    train_metrics = model.fit(X_train, y_train, X_val, y_val)
    logger.info(f"[P2] Training complete: {train_metrics}")

    # Step 5: Evaluate on C7 target assets
    # Use hourly test data for P2 if provided, otherwise fall back to daily
    eval_ts_map = p2_test_ts_map if p2_test_ts_map is not None else test_ts_map

    results = {
        "train_metrics": train_metrics, "vocab_size": vocab.size,
        "n_train_assets": len(train_decomps), "n_train_samples": len(y_train),
        "per_asset": {}, "interval": interval,
    }

    midpoint = vocab.size // 2
    quarter = vocab.size // 4

    def to_direction(token_ids):
        dirs = np.zeros_like(token_ids, dtype=np.float64)
        dirs[token_ids >= midpoint + quarter] = 1.0
        dirs[token_ids <= midpoint - quarter] = -1.0
        return dirs

    for ticker in eval_assets:
        test_ts = eval_ts_map.get(ticker, test_ts_map.get(ticker))
        if test_ts is None or test_ts.length < 50:
            logger.warning(f"[P2] Skipping {ticker}: not enough test data")
            continue

        test_decomp = decompose(test_ts, wavelet=wavelet, level=decomp_level)
        test_decomp = WaveletDecomposition(
            coefficients=test_decomp.coefficients, wavelet=test_decomp.wavelet,
            level=test_decomp.level, original_length=test_decomp.original_length, ticker=ticker,
        )

        test_token_seq = tokenizer.tokenize(test_decomp)
        test_dataset = build_sequence_dataset(
            [test_token_seq], vocab,
            context_length=context_length, asset_class_map=asset_class_ids_map,
        )
        t_ctx, t_tgt, t_lvl, t_ac = test_dataset.to_arrays()

        if len(t_tgt) == 0:
            logger.warning(f"[P2] No test samples for {ticker}")
            continue

        X_test = np.column_stack([t_ctx, t_lvl.reshape(-1, 1), t_ac.reshape(-1, 1)])
        y_test = t_tgt

        predicted = model.predict(X_test)
        proba = model.predict_proba(X_test)

        pred_dirs = to_direction(predicted)
        actual_dirs = to_direction(y_test)
        has_dir = actual_dirs != 0.0
        da = float(np.mean(pred_dirs[has_dir] == actual_dirs[has_dir])) if np.any(has_dir) else 0.5
        token_acc = float(np.mean(predicted == y_test))
        p2_errors = y_test.astype(np.float64) - predicted.astype(np.float64)

        results["per_asset"][ticker] = {
            "directional_accuracy": da,
            "token_accuracy": token_acc,
            "n_test_samples": len(y_test),
            "predicted": predicted,
            "actual": y_test,
            "proba": proba,
            "p2_errors": p2_errors,
            "pred_dirs": pred_dirs,
            "actual_dirs": actual_dirs,
        }

    return results


def run_p2_multilevel(
    eval_assets: list[str],
    test_ts_map: dict[str, TimeSeries],
    optimal_config: dict,
    p2_test_ts_map: dict[str, TimeSeries] | None = None,
) -> dict:
    """Run P2 multi-level: all optimal levels -> token probs -> meta-classifier -> directional accuracy.

    Trains on full Phase 3 universe, evaluates on target assets.
    p2_test_ts_map: If provided, use these (e.g., hourly) for evaluation.
    """
    logger.info("[P2-ML] Running multi-level P2 (full universe training)...")

    optimal_levels = optimal_config.get("optimal_levels", [1, 2, 3])
    sax_config = SAXConfig(
        n_segments=optimal_config.get("n_segments", 20),
        alphabet_size=optimal_config.get("alphabet_size", 8),
        word_length=optimal_config.get("word_length", 4),
        word_stride=1,
    )
    context_length = optimal_config.get("context_length", 32)
    vocab_size_max = optimal_config.get("vocab_size", 500)
    embed_dim = optimal_config.get("embed_dim", 64)
    num_heads = optimal_config.get("num_heads", 4)
    num_layers_gpt = optimal_config.get("num_layers", 3)
    wavelet = optimal_config.get("wavelet", "db4")
    decomp_level = optimal_config.get("decomp_level", 5)
    interval = optimal_config.get("interval", "1h")

    # Load full universe training data (hourly for C experiments)
    all_word_sequences, train_decomps, asset_class_ids_map = load_universe_train_data(
        sax_config, wavelet, decomp_level, interval=interval
    )

    vocab = SAXVocabulary.from_corpus(all_word_sequences, min_freq=2, max_size=vocab_size_max)

    # Train one WaveletGPT model on all levels using full universe
    tokenizer = WaveletSAXTokenizer(vocab, sax_config)
    train_token_sequences = []
    for ticker in train_decomps:
        token_seq = tokenizer.tokenize(train_decomps[ticker])
        train_token_sequences.append(token_seq)

    dataset = build_sequence_dataset(
        train_token_sequences, vocab,
        context_length=context_length, asset_class_map=asset_class_ids_map,
    )
    contexts, targets, levels_arr, asset_classes_arr = dataset.to_arrays()
    if len(targets) == 0:
        return {"failed": True, "reason": "No training samples for multi-level"}

    X_all = np.column_stack([contexts, levels_arr.reshape(-1, 1), asset_classes_arr.reshape(-1, 1)])
    y_all = targets

    n_train = int(len(y_all) * 0.8)
    model = WaveletGPT(
        vocab_size=vocab.size, context_length=context_length,
        embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers_gpt,
        dropout=0.1, epochs=80, batch_size=64, learning_rate=0.0003, patience=15,
    )
    model.fit(X_all[:n_train], y_all[:n_train], X_all[n_train:], y_all[n_train:])

    # For each test asset, get probabilities from each optimal level, then meta-classify
    eval_ts_map = p2_test_ts_map if p2_test_ts_map is not None else test_ts_map
    results = {"per_asset": {}}

    for ticker in eval_assets:
        test_ts = eval_ts_map.get(ticker, test_ts_map.get(ticker))
        if test_ts is None or test_ts.length < 50:
            continue

        test_decomp = decompose(test_ts, wavelet=wavelet, level=decomp_level)
        test_decomp = WaveletDecomposition(
            coefficients=test_decomp.coefficients, wavelet=test_decomp.wavelet,
            level=test_decomp.level, original_length=test_decomp.original_length, ticker=ticker,
        )
        test_token_seq = tokenizer.tokenize(test_decomp)

        # Collect probabilities per level
        level_probas = {}
        level_actuals = {}
        min_samples = float("inf")

        for lvl in optimal_levels:
            if lvl not in test_token_seq.level_sequences:
                continue

            test_dataset_lvl = build_sequence_dataset(
                [test_token_seq], vocab,
                context_length=context_length, asset_class_map=asset_class_ids_map,
            )
            t_ctx, t_tgt, t_lvl, t_ac = test_dataset_lvl.to_arrays()

            # Filter to just this level
            level_mask = t_lvl == lvl
            if not np.any(level_mask):
                continue

            t_ctx_lvl = t_ctx[level_mask]
            t_tgt_lvl = t_tgt[level_mask]
            t_lvl_lvl = t_lvl[level_mask]
            t_ac_lvl = t_ac[level_mask]

            X_test_lvl = np.column_stack([t_ctx_lvl, t_lvl_lvl.reshape(-1, 1), t_ac_lvl.reshape(-1, 1)])
            proba = model.predict_proba(X_test_lvl)

            level_probas[lvl] = proba
            level_actuals[lvl] = t_tgt_lvl

        # Filter out levels with too few samples (need at least 10)
        valid_levels = {lvl: p for lvl, p in level_probas.items() if len(p) >= 10}
        if len(valid_levels) < 2:
            # Need at least 2 levels for multi-level to be meaningful
            logger.warning(f"[P2-ML] {ticker}: only {len(valid_levels)} valid levels, skipping")
            continue

        min_samples = min(len(p) for p in valid_levels.values())

        # Truncate all valid levels to same length and stack probabilities
        stacked_features = []
        actual_tokens = None
        for lvl in sorted(valid_levels.keys()):
            n = int(min_samples)
            stacked_features.append(valid_levels[lvl][:n])
            if actual_tokens is None:
                actual_tokens = level_actuals[lvl][:n]

        # Meta-features: concatenate probabilities from all levels
        meta_X = np.concatenate(stacked_features, axis=1)

        # Meta-target: directional labels from actual tokens
        midpoint = vocab.size // 2
        quarter = vocab.size // 4
        actual_dirs = np.zeros(len(actual_tokens), dtype=np.float64)
        actual_dirs[actual_tokens >= midpoint + quarter] = 1.0
        actual_dirs[actual_tokens <= midpoint - quarter] = -1.0

        # Train XGBoost meta-classifier using first 60% as train, rest as test
        meta_split = int(len(meta_X) * 0.6)
        if meta_split < 5:
            continue

        meta_clf = GradientBoostModel(n_estimators=100, max_depth=4, learning_rate=0.1)
        meta_clf.fit(meta_X[:meta_split], actual_dirs[:meta_split])
        meta_pred = meta_clf.predict(meta_X[meta_split:])

        # Directional accuracy on meta-test
        meta_pred_dirs = np.sign(meta_pred)
        meta_actual_dirs = actual_dirs[meta_split:]
        has_dir = meta_actual_dirs != 0.0
        if np.any(has_dir):
            da = float(np.mean(meta_pred_dirs[has_dir] == meta_actual_dirs[has_dir]))
        else:
            da = 0.5

        results["per_asset"][ticker] = {
            "directional_accuracy": da,
            "levels_used": sorted(level_probas.keys()),
            "n_meta_test": len(meta_actual_dirs),
            "n_levels": len(level_probas),
        }

    return results


# ── Experiment 4: Combined P1 + P2 ────────────────────────────────────────

def run_combined(
    assets: list[str],
    p1_results: dict[str, dict],
    p2_results: dict,
    train_ts_map: dict[str, TimeSeries],
    test_ts_map: dict[str, TimeSeries],
    optimal_config: dict,
) -> dict:
    """Combined: P1 feature vectors + P2 token probabilities -> XGBoost meta-learner."""
    logger.info("[Combined] Running combined P1+P2 ensemble...")

    cfg = WaveCastConfig()
    cfg.sax = SAXConfig(
        n_segments=optimal_config.get("n_segments", 20),
        alphabet_size=optimal_config.get("alphabet_size", 8),
        word_length=optimal_config.get("word_length", 4),
        word_stride=1,
    )
    wavelet = optimal_config.get("wavelet", "db4")
    decomp_level = optimal_config.get("decomp_level", 5)
    context_length = optimal_config.get("context_length", 32)
    vocab_size_max = optimal_config.get("vocab_size", 500)

    results = {"per_asset": {}}

    for ticker in assets:
        if ticker not in p1_results or ticker not in p2_results.get("per_asset", {}):
            logger.warning(f"[Combined] Skipping {ticker}: missing P1 or P2 results")
            continue

        p1 = p1_results[ticker]
        p2 = p2_results["per_asset"][ticker]

        # The challenge: P1 and P2 have different test set sizes because they
        # operate on different representations. We align by taking the minimum
        # and using the last N samples from each.
        n_p1 = p1["n_test"]
        n_p2 = p2["n_test_samples"]
        n_common = min(n_p1, n_p2)

        if n_common < 20:
            logger.warning(f"[Combined] {ticker}: too few common samples ({n_common})")
            continue

        # P1 predictions (log returns) -> take last n_common
        p1_pred = p1["ensemble_pred"][-n_common:]
        p1_actual = p1["y_test"][-n_common:]

        # P2 token probabilities -> take last n_common
        p2_proba = p2["proba"][-n_common:]  # (n_common, vocab_size)

        # Top-K probability features to keep dimensionality manageable
        top_k = min(20, p2_proba.shape[1])
        # For each sample, take top-K probabilities as features
        sorted_proba = np.sort(p2_proba, axis=1)[:, -top_k:]  # top K probs

        # Combined features: P1 prediction + P2 top-K probabilities
        combined_X = np.column_stack([p1_pred.reshape(-1, 1), sorted_proba])

        # Target: actual direction from P1 (log returns)
        y_dir = np.sign(p1_actual)

        # Train/test split on combined features
        split = int(len(combined_X) * 0.6)
        if split < 10:
            continue

        meta_clf = GradientBoostModel(n_estimators=100, max_depth=4, learning_rate=0.1)
        meta_clf.fit(combined_X[:split], y_dir[:split])
        meta_pred = meta_clf.predict(combined_X[split:])

        # Directional accuracy
        meta_pred_dir = np.sign(meta_pred)
        y_dir_test = y_dir[split:]
        has_dir = y_dir_test != 0.0
        if np.any(has_dir):
            da = float(np.mean(meta_pred_dir[has_dir] == y_dir_test[has_dir]))
        else:
            da = 0.5

        results["per_asset"][ticker] = {
            "directional_accuracy": da,
            "n_combined_test": len(y_dir_test),
            "n_features": combined_X.shape[1],
        }

    return results


# ── Error Correlation ──────────────────────────────────────────────────────

def compute_error_correlation(
    p1_results: dict[str, dict],
    p2_results: dict,
) -> dict:
    """Compute Pearson correlation between P1 and P2 error vectors."""
    correlations = {}

    for ticker, p1 in p1_results.items():
        if ticker not in p2_results.get("per_asset", {}):
            continue

        p1_errors = p1["p1_errors"]
        p2_data = p2_results["per_asset"][ticker]
        p2_errors = p2_data["p2_errors"]

        # Align lengths (take minimum from the end)
        n = min(len(p1_errors), len(p2_errors))
        if n < 10:
            continue

        p1_e = p1_errors[-n:].astype(np.float64)
        p2_e = p2_errors[-n:].astype(np.float64)

        # Remove any NaN/inf
        mask = np.isfinite(p1_e) & np.isfinite(p2_e)
        if mask.sum() < 10:
            continue

        corr, pval = pearsonr(p1_e[mask], p2_e[mask])
        correlations[ticker] = {
            "pearson_r": float(corr),
            "p_value": float(pval),
            "n_samples": int(mask.sum()),
        }

    return correlations


# ── Main ──────────────────────────────────────────────────────────────────

def load_optimal_config() -> dict:
    """Load optimal config from C1-C5 experiment results."""
    config = {}

    # Try to load C5 context (has aggregated optimal config)
    if C5_CONTEXT.exists():
        with open(C5_CONTEXT) as f:
            c5 = json.load(f)
        config.update(c5.get("optimal_config", {}))
        logger.info(f"Loaded C5 context: {list(config.keys())}")
    else:
        logger.warning("C5 context not found, using defaults")

    # Load individual experiment results if available
    for name in ["C1_granularity.json", "C2_levels.json", "C3_cross_sector.json",
                  "C4_vocab.json", "C5_context.json"]:
        path = EXPERIMENTS_DIR / name
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            # Extract any config values that were determined as optimal
            if "best_config" in data:
                config.update(data["best_config"])
            if "optimal" in data:
                config.update(data["optimal"])

    # Defaults matching C1 experiment baseline parameters.
    # C experiments use HOURLY data with n_segments=256, context_length=16.
    defaults = {
        "n_segments": 256,
        "alphabet_size": 7,
        "word_length": 4,
        "context_length": 16,
        "vocab_size": 300,
        "embed_dim": 64,
        "num_heads": 4,
        "num_layers": 3,
        "dropout": 0.1,
        "wavelet": "db4",
        "decomp_level": 5,
        "optimal_levels": [1, 2, 3],
        "interval": "1h",
    }
    for k, v in defaults.items():
        if k not in config:
            config[k] = v

    # For hourly data, n_segments=256 is the baseline (plenty of bars).
    # For daily data, ensure at least 64 to produce enough tokens.
    interval = config.get("interval", "1h")
    min_segments = 64 if interval == "1d" else 32
    if config.get("n_segments", 256) < min_segments:
        logger.info(f"Raising n_segments from {config['n_segments']} to {min_segments} ({interval} data minimum)")
        config["n_segments"] = min_segments

    return config


def main() -> None:
    start_time = time.monotonic()
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load optimal config
    optimal_config = load_optimal_config()
    logger.info(f"Optimal config: {json.dumps(optimal_config, indent=2, default=str)}")

    # Load and split data
    # P1 uses daily data (features from daily price series)
    # P2 uses hourly data (matching C experiment methodology)
    p2_interval = optimal_config.get("interval", "1h")
    logger.info("Loading data for %s (P1: daily, P2: %s)...", C7_ASSETS, p2_interval)

    train_ts_map: dict[str, TimeSeries] = {}
    test_ts_map: dict[str, TimeSeries] = {}
    p2_test_ts_map: dict[str, TimeSeries] = {}

    for ticker in C7_ASSETS:
        # Daily for P1
        ts_daily = load_ts(ticker, interval="1d")
        train_ts, test_ts = split_train_test(ts_daily)
        train_ts_map[ticker] = train_ts
        test_ts_map[ticker] = test_ts
        logger.info(f"  {ticker} daily: train={train_ts.length}, test={test_ts.length}")

        # Hourly for P2
        ts_hourly = load_ts(ticker, interval=p2_interval)
        _, test_ts_h = split_train_test(ts_hourly)
        p2_test_ts_map[ticker] = test_ts_h
        logger.info(f"  {ticker} {p2_interval}: test={test_ts_h.length}")

    # ── Experiment 1: P1 Alone ──
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 1: Pipeline 1 (WaveletLSTM + XGBoost)")
    logger.info("=" * 60)

    p1_failed = False
    p1_results: dict[str, dict] = {}
    p1_errors: list[str] = []
    cfg = WaveCastConfig()

    for ticker in C7_ASSETS:
        try:
            result = run_p1_for_asset(ticker, train_ts_map[ticker], test_ts_map[ticker], cfg)
            p1_results[ticker] = result
            logger.info(f"  {ticker} P1 DA: XGB={result['xgb_da']:.3f}, LSTM={result['lstm_da']:.3f}, Ensemble={result['ensemble_da']:.3f}")
        except Exception as e:
            logger.error(f"  P1 FAILED for {ticker}: {e}")
            p1_errors.append(f"{ticker}: {str(e)}")
            traceback.print_exc()

    if not p1_results:
        p1_failed = True
        logger.error("P1 completely failed — degrading to P2-only results")

    # ── Experiment 2: P2 Alone ──
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 2: Pipeline 2 (WaveletGPT)")
    logger.info("=" * 60)

    p2_results = run_p2_for_assets(C7_ASSETS, test_ts_map, optimal_config, p2_test_ts_map=p2_test_ts_map)

    if not p2_results.get("failed"):
        for ticker, data in p2_results.get("per_asset", {}).items():
            logger.info(f"  {ticker} P2 DA: {data['directional_accuracy']:.3f}, Token Acc: {data['token_accuracy']:.3f}")

    # ── Experiment 3: P2 Multi-Level ──
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 3: Pipeline 2 Multi-Level Meta-Classifier")
    logger.info("=" * 60)

    p2_ml_results = run_p2_multilevel(C7_ASSETS, test_ts_map, optimal_config, p2_test_ts_map=p2_test_ts_map)
    for ticker, data in p2_ml_results.get("per_asset", {}).items():
        logger.info(f"  {ticker} P2-ML DA: {data['directional_accuracy']:.3f} (levels={data['levels_used']})")

    # ── Experiment 4: Combined P1 + P2 ──
    if not p1_failed and not p2_results.get("failed"):
        logger.info("\n" + "=" * 60)
        logger.info("Experiment 4: Combined P1 + P2 Meta-Learner")
        logger.info("=" * 60)

        combined_results = run_combined(
            C7_ASSETS, p1_results, p2_results,
            train_ts_map, test_ts_map, optimal_config,
        )
        for ticker, data in combined_results.get("per_asset", {}).items():
            logger.info(f"  {ticker} Combined DA: {data['directional_accuracy']:.3f}")
    else:
        combined_results = {"skipped": True, "reason": "P1 or P2 failed"}

    # ── Error Correlation ──
    error_correlations = {}
    if not p1_failed and not p2_results.get("failed"):
        logger.info("\n" + "=" * 60)
        logger.info("Error Correlation Analysis")
        logger.info("=" * 60)

        error_correlations = compute_error_correlation(p1_results, p2_results)
        for ticker, corr in error_correlations.items():
            logger.info(f"  {ticker}: Pearson r={corr['pearson_r']:.3f} (p={corr['p_value']:.4f}, n={corr['n_samples']})")

    # ── Summary Table ──
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: Side-by-Side Directional Accuracy")
    logger.info("=" * 80)
    logger.info(f"{'Ticker':<8} {'P1':>8} {'P2':>8} {'P2-ML':>8} {'Combined':>10} {'P1-P2 Corr':>12}")
    logger.info("-" * 80)

    summary_rows = []
    for ticker in C7_ASSETS:
        p1_da = p1_results[ticker]["ensemble_da"] if ticker in p1_results else None
        p2_da = p2_results.get("per_asset", {}).get(ticker, {}).get("directional_accuracy")
        p2ml_da = p2_ml_results.get("per_asset", {}).get(ticker, {}).get("directional_accuracy")
        comb_da = combined_results.get("per_asset", {}).get(ticker, {}).get("directional_accuracy") if isinstance(combined_results, dict) else None
        corr_r = error_correlations.get(ticker, {}).get("pearson_r")

        row = {
            "ticker": ticker,
            "p1_da": p1_da,
            "p2_da": p2_da,
            "p2_ml_da": p2ml_da,
            "combined_da": comb_da,
            "error_correlation": corr_r,
        }
        summary_rows.append(row)

        def fmt(v):
            return f"{v:.3f}" if v is not None else "N/A"

        logger.info(f"{ticker:<8} {fmt(p1_da):>8} {fmt(p2_da):>8} {fmt(p2ml_da):>8} {fmt(comb_da):>10} {fmt(corr_r):>12}")

    # ── Decision Logic ──
    recommendation = "insufficient_data"
    if not p1_failed and not p2_results.get("failed"):
        # Check if combined > max(P1, P2) by > 2%
        improvements = []
        for row in summary_rows:
            if row["combined_da"] is not None and row["p1_da"] is not None and row["p2_da"] is not None:
                best_single = max(row["p1_da"], row["p2_da"])
                improvement = row["combined_da"] - best_single
                improvements.append(improvement)

        avg_corr = np.mean([c["pearson_r"] for c in error_correlations.values()]) if error_correlations else 1.0

        if improvements:
            avg_improvement = np.mean(improvements)
            if avg_improvement > 0.02 and avg_corr < 0.3:
                recommendation = "permanent_ensemble"
            elif avg_improvement > 0.01:
                recommendation = "conditional_ensemble"
            elif avg_improvement > -0.01:
                recommendation = "marginal_benefit"
            else:
                recommendation = "no_benefit"
    elif p1_failed:
        recommendation = "p2_only"

    logger.info(f"\nRecommendation: {recommendation}")
    if error_correlations:
        avg_corr = np.mean([c["pearson_r"] for c in error_correlations.values()])
        logger.info(f"Avg error correlation: {avg_corr:.3f} (< 0.3 supports ensemble)")

    # ── Save Results ──
    duration = time.monotonic() - start_time

    output = {
        "experiment": "C7_ensemble_comparison",
        "research_question": "Does combining Pipeline 1 + Pipeline 2 beat either alone?",
        "assets": C7_ASSETS,
        "walk_forward_split": {
            "train": f"{TRAIN_START} to {TRAIN_END}",
            "test": f"{TEST_START} to {TEST_END}",
        },
        "optimal_config_used": optimal_config,
        "p1_interval": "1d",
        "p2_interval": p2_interval,
        "p1_failed": p1_failed,
        "p1_errors": p1_errors,
        "results": {
            "p1_alone": {
                ticker: {
                    "xgb_da": r["xgb_da"],
                    "lstm_da": r["lstm_da"],
                    "ensemble_da": r["ensemble_da"],
                    "xgb_rmse": r["xgb_rmse"],
                    "lstm_rmse": r["lstm_rmse"],
                    "ensemble_rmse": r["ensemble_rmse"],
                    "n_train": r["n_train"],
                    "n_test": r["n_test"],
                    "n_features": r["n_features"],
                }
                for ticker, r in p1_results.items()
            } if not p1_failed else "FAILED",
            "p2_alone": {
                ticker: {
                    "directional_accuracy": d["directional_accuracy"],
                    "token_accuracy": d["token_accuracy"],
                    "n_test_samples": d["n_test_samples"],
                }
                for ticker, d in p2_results.get("per_asset", {}).items()
            },
            "p2_multilevel": {
                ticker: d
                for ticker, d in p2_ml_results.get("per_asset", {}).items()
            },
            "combined": {
                ticker: d
                for ticker, d in combined_results.get("per_asset", {}).items()
            } if isinstance(combined_results, dict) and "per_asset" in combined_results else combined_results,
        },
        "error_correlations": error_correlations,
        "summary": summary_rows,
        "recommendation": recommendation,
        "duration_seconds": duration,
    }

    with open(C7_OUTPUT, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\nResults saved to {C7_OUTPUT}")
    logger.info(f"Total duration: {duration:.1f}s")


if __name__ == "__main__":
    main()
