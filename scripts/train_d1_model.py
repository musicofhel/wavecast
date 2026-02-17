#!/usr/bin/env python3
"""Train production D1 (Delta+Aux) model on 2021-2025 data.

Uses quick-mode architecture (embed=64, 3 layers, 20 epochs) — Phase 9
showed signal is in the representation, not model capacity. Quick mode
marginally outperformed full-scale (60.5% vs 60.0% econ dir accuracy).

Saves to ~/.wavecast/models/d1_augmented_v1/:
  - model.pt + config.json (WaveletGPT.save())
  - quantile_boundaries.json (per-level return quantile boundaries)
  - training_config.json (all hyperparams for reproducibility)
  - training_metrics.json (final loss, epoch, time)

Usage:
    python scripts/train_d1_model.py [--output-dir PATH]
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wavecast.core.universe import DEFAULT_UNIVERSE
from wavecast.data.auxiliary_features import (
    N_AUX_FEATURES,
    compute_detail_auxiliary_features,
)
from wavecast.data.cache import ParquetCache
from wavecast.data.continuous_dataset import build_continuous_dataset
from wavecast.experiments.runner import SECTOR_ID_MAP
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
)
from wavecast.wavelets.dwt import decompose

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TICKERS = [a.ticker for a in DEFAULT_UNIVERSE.assets]
INTERVAL = "1h"
# Train on ALL available data (2021-2025) — forward test evaluates on 2026 only
TRAIN_END = "2025-12-31"
DETAIL_LEVELS = [1, 2, 5]
CONTEXT_LENGTH = 16
N_CLASSES = 5
PERCENTILES = [10.0, 30.0, 70.0, 90.0]

# Quick-mode architecture — signal is in representation, not model capacity
MODEL_KWARGS = {
    "embed_dim": 64,
    "num_heads": 4,
    "num_layers": 3,
    "dropout": 0.1,
    "epochs": 20,
    "batch_size": 64,
    "learning_rate": 0.0005,
    "patience": 10,
}

DEFAULT_OUTPUT = Path.home() / ".wavecast" / "models" / "d1_augmented_v1"


def detail_delta_coeffs(decomp, lvl):
    c = decomp.detail_at_level(lvl)
    return np.diff(c) if len(c) > 1 else c


def detail_delta_aux_fn(decomp, lvl):
    detail = decomp.detail_at_level(lvl)
    approx = decomp.approximation
    return compute_detail_auxiliary_features(detail, approx)


def build_aux_windows(aux_series, context_length, n_aux_features):
    windows = []
    for (_ticker, _level), aux in sorted(aux_series.items()):
        if len(aux) <= context_length:
            continue
        for i in range(len(aux) - context_length):
            windows.append(aux[i:i + context_length])
    if not windows:
        return np.empty((0, context_length, n_aux_features), dtype=np.float64)
    return np.array(windows, dtype=np.float64)


def build_asset_class_map():
    ac_map = {}
    for a in DEFAULT_UNIVERSE.assets:
        if a.sector is not None:
            ac_map[a.ticker] = SECTOR_ID_MAP.get(a.sector.value, 0)
    return ac_map


def main() -> None:
    output_dir = DEFAULT_OUTPUT
    for arg in sys.argv[1:]:
        if arg.startswith("--output-dir="):
            output_dir = Path(arg.split("=", 1)[1])

    cache = ParquetCache(Path.home() / ".wavecast" / "cache")
    ac_map = build_asset_class_map()

    # Load ALL available data up to TRAIN_END
    train_series: dict[tuple[str, int], NDArray] = {}
    train_aux_series: dict[tuple[str, int], NDArray] = {}

    loaded = 0
    for ticker in TICKERS:
        ts = cache.get(ticker, INTERVAL)
        if ts is None:
            logger.warning("No data for %s", ticker)
            continue
        mask = ts.timestamps <= np.datetime64(TRAIN_END)
        if mask.sum() < 200:
            logger.warning("Insufficient data for %s (%d bars)", ticker, mask.sum())
            continue

        from wavecast.core.types import TimeSeries

        train_ts = TimeSeries(
            values=ts.values[mask], timestamps=ts.timestamps[mask],
            ticker=ticker, interval=INTERVAL,
        )
        decomp = decompose(train_ts, level=5)
        for lvl in DETAIL_LEVELS:
            tc = detail_delta_coeffs(decomp, lvl)
            if len(tc) > CONTEXT_LENGTH:
                train_series[(ticker, lvl)] = tc
                train_aux_series[(ticker, lvl)] = detail_delta_aux_fn(decomp, lvl)
        loaded += 1

    logger.info("Loaded %d tickers", loaded)

    # Build dataset
    train_ds = build_continuous_dataset(train_series, CONTEXT_LENGTH, ac_map, normalize=True)
    logger.info("Training dataset: %d windows", len(train_ds.windows))

    # Compute returns
    def compute_window_returns(ds, prices_cache, series_dict):
        rets = np.full(len(ds.windows), np.nan)
        valid = np.zeros(len(ds.windows), dtype=np.bool_)
        for i, w in enumerate(ds.windows):
            key = (w.ticker, w.level)
            if key not in series_dict:
                continue
            ts = prices_cache.get(w.ticker, INTERVAL)
            if ts is None:
                continue
            prices = ts.values[ts.timestamps <= np.datetime64(TRAIN_END)]
            coeffs_len = len(series_dict[key])
            target_pos = w.token_position
            if target_pos >= coeffs_len:
                continue
            span = 2 ** w.level
            bar_start = target_pos * span
            bar_end = bar_start + span
            if bar_end < len(prices) and prices[bar_start] > 0:
                rets[i] = (prices[bar_end] - prices[bar_start]) / prices[bar_start]
                valid[i] = True
        return rets, valid

    train_rets, train_valid = compute_window_returns(train_ds, cache, train_series)
    train_lvls = np.array([w.level for w in train_ds.windows], dtype=np.int64)

    # Quantile boundaries
    boundaries = compute_quantile_boundaries(
        train_rets, train_lvls, train_valid, PERCENTILES, per_level=True
    )
    y_train = assign_quantile_labels(train_rets, train_lvls, boundaries)

    # Class weights
    valid_train_labels = y_train[train_valid]
    counts = np.bincount(valid_train_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    # Build X array
    ctx_arr, lvl_arr, ac_arr = train_ds.to_arrays()
    train_aux_win = build_aux_windows(train_aux_series, CONTEXT_LENGTH, N_AUX_FEATURES)
    train_aux_flat = train_aux_win.reshape(len(train_aux_win), -1)
    X_train = np.column_stack([ctx_arr, train_aux_flat, lvl_arr, ac_arr])

    # Train
    model = WaveletGPT(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=N_CLASSES,
        class_weights=class_weights, input_mode="continuous",
        n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
    )
    logger.info("Training D1 model (quick architecture: 64d, 3 layers, 20 epochs)...")
    t0 = time.time()
    metrics = model.fit(X_train, y_train.astype(np.float64))
    train_time = time.time() - t0
    logger.info("Training done in %.1fs (loss=%.4f)", train_time, metrics["train_loss"])

    # Save model
    model.save(output_dir)
    logger.info("Model saved to %s", output_dir)

    # Save quantile boundaries alongside model
    # Convert numpy arrays to lists for JSON serialization
    boundaries_json = {}
    for level, bounds in boundaries.items():
        boundaries_json[str(level)] = bounds.tolist()
    with open(output_dir / "quantile_boundaries.json", "w") as f:
        json.dump(boundaries_json, f, indent=2)
    logger.info("Quantile boundaries saved")

    # Save training config
    training_config = {
        "model_kwargs": MODEL_KWARGS,
        "tickers": TICKERS,
        "interval": INTERVAL,
        "train_end": TRAIN_END,
        "detail_levels": DETAIL_LEVELS,
        "context_length": CONTEXT_LENGTH,
        "n_classes": N_CLASSES,
        "percentiles": PERCENTILES,
        "n_aux_features": N_AUX_FEATURES,
        "class_weights": class_weights,
        "n_windows": len(train_ds.windows),
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    # Save training metrics
    training_metrics = {
        "train_loss": metrics["train_loss"],
        "train_time_seconds": train_time,
        "n_windows": len(train_ds.windows),
        "n_valid_returns": int(train_valid.sum()),
    }
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=2)

    logger.info("All artifacts saved to %s", output_dir)
    print(f"\nProduction D1 model trained and saved to {output_dir}")
    print(f"  Windows: {len(train_ds.windows)}")
    print(f"  Loss: {metrics['train_loss']:.4f}")
    print(f"  Time: {train_time:.1f}s")


if __name__ == "__main__":
    main()
