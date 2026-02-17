#!/usr/bin/env python3
"""Feature Test 3: Wavelet Jump Filters (ψMR / ψTR).

Hypothesis: Causal convolution kernels derived from Aubrun et al. 2024
("Riding Wavelets") capture mean-reversion vs trend structure that
coefficient deltas alone cannot encode.

2 new features per timestep:
  - |mr_score|: mean-reversion strength (unsigned). High = recent past
    shows reversal pattern (older returns opposite sign to recent returns).
  - tr_score: trend score (signed). Positive = past returns consistently
    directional. Negative = choppy/inconsistent.

Both kernels are CAUSAL (backward-looking only). No future information.

Reference: arXiv:2404.16467, Section II.3-II.4, Figures 5, 7, 8.

Usage:
    python -m scripts.feature_tests.test_jump_filters
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scripts.feature_tests.harness import run_feature_test  # noqa: I001

N_FEATURES = 2
KERNEL_WIDTH = 8  # 8 bars = 1 trading day at hourly resolution


def _build_causal_psi_mr(width: int = KERNEL_WIDTH) -> NDArray:
    """Causal mean-reversion kernel: compares older vs recent past returns.

    Odd-symmetric around the midpoint of the lookback window.
    Older past weighted positively, recent past weighted negatively.
    High |score| = reversal pattern in the lookback window.
    """
    k = np.arange(width, dtype=np.float64)
    midpoint = (width - 1) / 2.0
    envelope = np.cos(np.pi * (k - midpoint) / width)
    sign = np.sign(k - midpoint)
    kernel = envelope * sign
    norm = np.linalg.norm(kernel)
    if norm > 0:
        kernel /= norm
    return kernel


def _build_causal_psi_tr(width: int = KERNEL_WIDTH) -> NDArray:
    """Causal trend kernel: are past returns consistently directional?

    Even-symmetric around the midpoint of the lookback window.
    All past bars weighted positively, peaked at center.
    High positive score = consistent upward trend in lookback.
    High negative score = consistent downward trend.
    """
    k = np.arange(width, dtype=np.float64)
    midpoint = (width - 1) / 2.0
    kernel = np.cos(np.pi * (k - midpoint) / width)
    norm = np.linalg.norm(kernel)
    if norm > 0:
        kernel /= norm
    return kernel


def compute_jump_filter_features(
    ohlcv_df: pd.DataFrame,
    detail_coeffs: NDArray,
    approx_coeffs: NDArray,
    level: int,
) -> NDArray:
    """Causal ψMR and ψTR convolution scores on log returns.

    Returns (n_deltas, 2) where n_deltas = len(detail_coeffs) - 1.
    Column 0: |mr_score| (unsigned mean-reversion strength)
    Column 1: tr_score (signed trend score)
    """
    n_deltas = len(detail_coeffs) - 1
    if n_deltas <= 0:
        return np.empty((0, N_FEATURES), dtype=np.float64)

    close = ohlcv_df["close"].values.astype(np.float64)
    # Log returns (safe against zeros)
    log_close = np.log(np.maximum(close, 1e-10))
    log_returns = np.diff(log_close)

    if len(log_returns) < KERNEL_WIDTH:
        return np.zeros((n_deltas, N_FEATURES), dtype=np.float64)

    psi_mr = _build_causal_psi_mr(KERNEL_WIDTH)
    psi_tr = _build_causal_psi_tr(KERNEL_WIDTH)

    # Causal convolution: at each t, apply kernel to returns[t-width+1:t+1]
    n_ret = len(log_returns)
    mr_scores = np.zeros(n_ret, dtype=np.float64)
    tr_scores = np.zeros(n_ret, dtype=np.float64)

    for t in range(KERNEL_WIDTH - 1, n_ret):
        window = log_returns[t - KERNEL_WIDTH + 1 : t + 1]
        mr_scores[t] = np.dot(window, psi_mr)
        tr_scores[t] = np.dot(window, psi_tr)

    bar_features = np.column_stack([np.abs(mr_scores), tr_scores])

    # Resample from bar resolution to coefficient-delta resolution
    indices = np.linspace(0, len(bar_features) - 1, n_deltas)
    resampled = np.zeros((n_deltas, N_FEATURES), dtype=np.float64)
    for j in range(N_FEATURES):
        resampled[:, j] = np.interp(
            indices, np.arange(len(bar_features)), bar_features[:, j]
        )

    return resampled


if __name__ == "__main__":
    run_feature_test(
        "jump_filters", compute_jump_filter_features, n_new_features=N_FEATURES
    )
