#!/usr/bin/env python3
"""Experiment 14: Amihud Illiquidity Ratio Feature.

Source: Amihud 2002; New Moment Estimators (Springer)
Hypothesis: Amihud = |return| / dollar_volume. Measures price impact per unit
of trading. High illiquidity -> thin market -> more volatile/predictable
regime transitions.

1 feature: amihud_illiq (rolling 20-bar z-scored)

Usage:
    python -m scripts.feature_tests.test_amihud
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scripts.feature_tests.harness import run_feature_test

N_FEATURES = 1


def compute_amihud(
    ohlcv_df: pd.DataFrame,
    detail_coeffs: NDArray,
    approx_coeffs: NDArray,
    level: int,
) -> NDArray:
    """Amihud illiquidity ratio at coefficient-delta resolution.

    Returns (n_deltas, 1).
    """
    n_deltas = len(detail_coeffs) - 1
    if n_deltas <= 0:
        return np.empty((0, N_FEATURES), dtype=np.float64)

    c = ohlcv_df["close"].values.astype(np.float64)
    v = ohlcv_df["volume"].values.astype(np.float64)

    # Amihud = |close[t] - close[t-1]| / (volume[t] * close[t])
    dollar_vol = v * c
    dollar_vol = np.where(dollar_vol < 1e-10, 1e-10, dollar_vol)
    ret = np.abs(np.diff(c))
    amihud_raw = ret / dollar_vol[1:]

    # Rolling 20-bar average
    window = 20
    rolling_amihud = np.zeros(len(amihud_raw), dtype=np.float64)
    for i in range(len(amihud_raw)):
        start = max(0, i - window + 1)
        rolling_amihud[i] = np.mean(amihud_raw[start : i + 1])

    # Z-score
    std = np.std(rolling_amihud)
    if std > 1e-10:
        amihud_z = (rolling_amihud - np.mean(rolling_amihud)) / std
    else:
        amihud_z = np.zeros_like(rolling_amihud)

    # Resample to coefficient-delta resolution
    indices = np.linspace(0, len(amihud_z) - 1, n_deltas)
    resampled = np.interp(indices, np.arange(len(amihud_z)), amihud_z)

    return resampled.reshape(-1, N_FEATURES)


if __name__ == "__main__":
    run_feature_test("amihud", compute_amihud, n_new_features=N_FEATURES)
