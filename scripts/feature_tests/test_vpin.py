#!/usr/bin/env python3
"""Experiment 13: VPIN (Volume-Synchronized PIN) Feature.

Source: Easley, Lopez de Prado, O'Hara 2012
Hypothesis: All 3 failed feature tests used price-derived features. VPIN uses
volume data -- orthogonal to D1. BVC rule classifies volume as buy/sell via
(close - low) / (high - low). VPIN = rolling |V_buy - V_sell| / V_total.

1 feature: vpin_score

Usage:
    python -m scripts.feature_tests.test_vpin
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scripts.feature_tests.harness import run_feature_test

N_FEATURES = 1


def compute_vpin(
    ohlcv_df: pd.DataFrame,
    detail_coeffs: NDArray,
    approx_coeffs: NDArray,
    level: int,
) -> NDArray:
    """VPIN feature at coefficient-delta resolution.

    Returns (n_deltas, 1) array.
    """
    n_deltas = len(detail_coeffs) - 1
    if n_deltas <= 0:
        return np.empty((0, N_FEATURES), dtype=np.float64)

    h = ohlcv_df["high"].values.astype(np.float64)
    lo = ohlcv_df["low"].values.astype(np.float64)
    c = ohlcv_df["close"].values.astype(np.float64)
    v = ohlcv_df["volume"].values.astype(np.float64)

    # BVC rule: buy fraction = (close - low) / (high - low)
    bar_range = h - lo
    bar_range = np.where(bar_range < 1e-10, 1e-10, bar_range)
    buy_frac = (c - lo) / bar_range
    buy_frac = np.clip(buy_frac, 0.0, 1.0)

    v_buy = buy_frac * v
    v_sell = (1.0 - buy_frac) * v

    # Rolling VPIN: |V_buy - V_sell| / V_total over window of 50 bars
    window = min(50, len(v) // 2) if len(v) > 10 else max(len(v) // 2, 1)
    vpin = np.zeros(len(v), dtype=np.float64)
    for i in range(len(v)):
        start = max(0, i - window + 1)
        total = np.sum(v[start : i + 1])
        if total > 0:
            vpin[i] = np.abs(np.sum(v_buy[start : i + 1]) - np.sum(v_sell[start : i + 1])) / total

    # Resample to coefficient-delta resolution
    indices = np.linspace(0, len(vpin) - 1, n_deltas)
    resampled = np.interp(indices, np.arange(len(vpin)), vpin)

    return resampled.reshape(-1, N_FEATURES)


if __name__ == "__main__":
    run_feature_test("vpin", compute_vpin, n_new_features=N_FEATURES)
