#!/usr/bin/env python3
"""Feature Test 1: Bar Structure Ratios.

Hypothesis: OHLC bar microstructure encodes rejection/acceptance patterns
the model can't see from close-only coefficient deltas.

3 new features per timestep:
  - body_ratio: |close-open| / range. Near 0 = rejection, near 1 = conviction
  - upper_wick: rejection of higher prices
  - lower_wick: rejection of lower prices

Usage:
    python -m scripts.feature_tests.test_bar_structure
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scripts.feature_tests.harness import run_feature_test  # noqa: I001

N_FEATURES = 3


def compute_bar_structure(
    ohlcv_df: pd.DataFrame,
    detail_coeffs: NDArray,
    approx_coeffs: NDArray,
    level: int,
) -> NDArray:
    """Bar structure ratios at coefficient-delta resolution.

    Returns (n_deltas, 3) where n_deltas = len(detail_coeffs) - 1.
    """
    n_deltas = len(detail_coeffs) - 1
    if n_deltas <= 0:
        return np.empty((0, N_FEATURES), dtype=np.float64)

    o = ohlcv_df["open"].values.astype(np.float64)
    h = ohlcv_df["high"].values.astype(np.float64)
    lo = ohlcv_df["low"].values.astype(np.float64)
    c = ohlcv_df["close"].values.astype(np.float64)

    bar_range = h - lo
    bar_range = np.where(bar_range < 1e-10, 1e-10, bar_range)

    body_ratio = np.abs(c - o) / bar_range
    upper_wick = (h - np.maximum(o, c)) / bar_range
    lower_wick = (np.minimum(o, c) - lo) / bar_range

    bar_features = np.column_stack([body_ratio, upper_wick, lower_wick])

    # Resample from bar resolution to coefficient-delta resolution
    # Level 1: ~2 bars per coefficient, so n_bars ≈ 2 * n_coeffs
    indices = np.linspace(0, len(bar_features) - 1, n_deltas)
    resampled = np.zeros((n_deltas, N_FEATURES), dtype=np.float64)
    for j in range(N_FEATURES):
        resampled[:, j] = np.interp(
            indices, np.arange(len(bar_features)), bar_features[:, j]
        )

    return resampled


if __name__ == "__main__":
    run_feature_test("bar_structure", compute_bar_structure, n_new_features=N_FEATURES)
