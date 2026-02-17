#!/usr/bin/env python3
"""Feature Test 2: Range Dynamics.

Hypothesis: Range expansion after contraction signals structural breaks.
Effort-vs-result ratio (Wyckoff) detects distribution/accumulation.

2 new features per timestep:
  - range_expansion: short-term range / long-term range
    > 1 = volatility expanding (potential structural break)
    < 1 = compression (energy building)
  - effort_result: close-to-close move / bar range
    Low ratio = big range but small move = churning/distribution
    High ratio = small range but big move = breakout with conviction

Usage:
    python -m scripts.feature_tests.test_range_dynamics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scripts.feature_tests.harness import run_feature_test  # noqa: I001

N_FEATURES = 2


def compute_range_dynamics(
    ohlcv_df: pd.DataFrame,
    detail_coeffs: NDArray,
    approx_coeffs: NDArray,
    level: int,
) -> NDArray:
    """Range dynamics features at coefficient-delta resolution.

    Returns (n_deltas, 2) where n_deltas = len(detail_coeffs) - 1.
    """
    n_deltas = len(detail_coeffs) - 1
    if n_deltas <= 0:
        return np.empty((0, N_FEATURES), dtype=np.float64)

    h = ohlcv_df["high"].values.astype(np.float64)
    lo = ohlcv_df["low"].values.astype(np.float64)
    c = ohlcv_df["close"].values.astype(np.float64)

    bar_range = h - lo

    # Range expansion: short-term range / long-term range
    short_avg = pd.Series(bar_range).rolling(5, min_periods=1).mean().values
    long_avg = pd.Series(bar_range).rolling(40, min_periods=1).mean().values
    range_expansion = short_avg / (long_avg + 1e-10)

    # Effort vs result: close-to-close move / bar range
    close_move = np.abs(np.diff(c, prepend=c[0]))
    effort_result = close_move / (bar_range + 1e-10)

    bar_features = np.column_stack([range_expansion, effort_result])

    # Resample to coefficient-delta resolution
    indices = np.linspace(0, len(bar_features) - 1, n_deltas)
    resampled = np.zeros((n_deltas, N_FEATURES), dtype=np.float64)
    for j in range(N_FEATURES):
        resampled[:, j] = np.interp(
            indices, np.arange(len(bar_features)), bar_features[:, j]
        )

    return resampled


if __name__ == "__main__":
    run_feature_test("range_dynamics", compute_range_dynamics, n_new_features=N_FEATURES)
