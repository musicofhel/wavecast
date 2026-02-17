#!/usr/bin/env python3
"""Train level-1-only D1 model on 2021-2025 data.

Identical to train_d1_model.py except DETAIL_LEVELS = [1].
Levels 2 and 5 are pure noise (51% / 52% econ dir on 2025 test, confirmed
on 2026 OOS). Removing them eliminates ~35% gradient noise from training.

Saves to ~/.wavecast/models/d1_level1_v1/

Usage:
    python scripts/train_d1_level1.py [--output-dir=PATH]
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wavecast.core.types import TimeSeries
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
TRAIN_END = "2025-12-31"
DETAIL_LEVELS = [1]  # <-- ONLY level 1
CONTEXT_LENGTH = 16
N_CLASSES = 5
PERCENTILES = [10.0, 30.0, 70.0, 90.0]

MODEL_KWARGS = {
    "embed_dim": 64, "num_heads": 4, "num_layers": 3,
    "dropout": 0.1, "epochs": 20, "batch_size": 64,
    "learning_rate": 0.0005, "patience": 15,
}

DEFAULT_OUTPUT = Path.home() / ".wavecast" / "models" / "d1_level1_v1"


def main() -> None:
    output_dir = DEFAULT_OUTPUT
    for arg in sys.argv[1:]:
        if arg.startswith("--output-dir="):
            output_dir = Path(arg.split("=", 1)[1])

    cache = ParquetCache(Path.home() / ".wavecast" / "cache")
    ac_map = {
        a.ticker: SECTOR_ID_MAP.get(a.sector.value, 0)
        for a in DEFAULT_UNIVERSE.assets if a.sector
    }

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

        train_ts = TimeSeries(
            values=ts.values[mask], timestamps=ts.timestamps[mask],
            ticker=ticker, interval=INTERVAL,
        )
        decomp = decompose(train_ts, level=5)
        for lvl in DETAIL_LEVELS:
            detail = decomp.detail_at_level(lvl)
            deltas = np.diff(detail) if len(detail) > 1 else detail
            if len(deltas) > CONTEXT_LENGTH:
                train_series[(ticker, lvl)] = deltas
                train_aux_series[(ticker, lvl)] = compute_detail_auxiliary_features(
                    detail, decomp.approximation
                )
        loaded += 1

    logger.info("Loaded %d tickers, levels=%s", loaded, DETAIL_LEVELS)

    train_ds = build_continuous_dataset(train_series, CONTEXT_LENGTH, ac_map, normalize=True)
    logger.info("Training dataset: %d windows (level-1 only)", len(train_ds.windows))

    # Compute returns
    rets = np.full(len(train_ds.windows), np.nan)
    valid = np.zeros(len(train_ds.windows), dtype=np.bool_)
    for i, w in enumerate(train_ds.windows):
        key = (w.ticker, w.level)
        if key not in train_series:
            continue
        ts = cache.get(w.ticker, INTERVAL)
        if ts is None:
            continue
        prices = ts.values[ts.timestamps <= np.datetime64(TRAIN_END)]
        target_pos = w.token_position
        if target_pos >= len(train_series[key]):
            continue
        span = 2 ** w.level
        bar_start = target_pos * span
        bar_end = bar_start + span
        if bar_end < len(prices) and prices[bar_start] > 0:
            rets[i] = (prices[bar_end] - prices[bar_start]) / prices[bar_start]
            valid[i] = True

    lvls = np.array([w.level for w in train_ds.windows], dtype=np.int64)
    boundaries = compute_quantile_boundaries(rets, lvls, valid, PERCENTILES, per_level=True)
    y_train = assign_quantile_labels(rets, lvls, boundaries)

    valid_labels = y_train[valid]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    # Build X
    ctx_arr, lvl_arr, ac_arr = train_ds.to_arrays()
    aux_windows = []
    for (_t, _l), aux in sorted(train_aux_series.items()):
        if len(aux) <= CONTEXT_LENGTH:
            continue
        for j in range(len(aux) - CONTEXT_LENGTH):
            aux_windows.append(aux[j:j + CONTEXT_LENGTH])
    aux_win = np.array(aux_windows, dtype=np.float64) if aux_windows else np.empty(
        (0, CONTEXT_LENGTH, N_AUX_FEATURES), dtype=np.float64
    )
    aux_flat = aux_win.reshape(len(aux_win), -1)
    X_train = np.column_stack([ctx_arr, aux_flat, lvl_arr, ac_arr])

    # Train
    model = WaveletGPT(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=N_CLASSES,
        class_weights=class_weights, input_mode="continuous",
        n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
    )
    logger.info("Training D1 level-1-only model...")
    t0 = time.time()
    metrics = model.fit(X_train, y_train.astype(np.float64))
    train_time = time.time() - t0
    logger.info("Training done in %.1fs (loss=%.4f)", train_time, metrics["train_loss"])

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir)

    boundaries_json = {str(k): v.tolist() for k, v in boundaries.items()}
    with open(output_dir / "quantile_boundaries.json", "w") as f:
        json.dump(boundaries_json, f, indent=2)

    with open(output_dir / "training_config.json", "w") as f:
        json.dump({
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
            "level_1_only": True,
        }, f, indent=2)

    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump({
            "train_loss": metrics["train_loss"],
            "train_time_seconds": train_time,
            "n_windows": len(train_ds.windows),
            "n_valid_returns": int(valid.sum()),
        }, f, indent=2)

    print(f"\nLevel-1-only D1 model saved to {output_dir}")
    print(f"  Windows: {len(train_ds.windows)}")
    print(f"  Loss: {metrics['train_loss']:.4f}")
    print(f"  Time: {train_time:.1f}s")


if __name__ == "__main__":
    main()
