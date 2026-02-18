#!/usr/bin/env python3
"""Feature Test 15: Wavelet Scattering Features.

Hypothesis: D1 is a linear wavelet decomposition. Wavelet scattering adds
nonlinear multi-resolution features via second-order cross-band interactions.
Detects jumps, volatility asymmetry, and regime changes invisible to D1.

3 new features per timestep (moments of scattering coefficients):
  - scatter_mean: mean of scattering output across paths
  - scatter_var: variance of scattering output across paths
  - scatter_skew: skewness of scattering output across paths

Usage:
    python -m scripts.feature_tests.test_wavelet_scatter
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import skew
from scripts.feature_tests.harness import run_feature_test  # noqa: I001

N_FEATURES = 3
SCATTER_J = 4  # Number of octaves for scattering transform


def compute_wavelet_scattering(
    ohlcv_df: pd.DataFrame,
    detail_coeffs: NDArray,
    approx_coeffs: NDArray,
    level: int,
) -> NDArray:
    """Wavelet scattering features at coefficient-delta resolution.

    Returns (n_deltas, 3) where n_deltas = len(detail_coeffs) - 1.
    """
    n_deltas = len(detail_coeffs) - 1
    if n_deltas <= 0:
        return np.empty((0, N_FEATURES), dtype=np.float64)

    close = ohlcv_df["close"].values.astype(np.float64)

    # Compute log returns for scattering
    log_returns = np.diff(np.log(np.maximum(close, 1e-10)))
    if len(log_returns) < 32:
        return np.zeros((n_deltas, N_FEATURES), dtype=np.float64)

    # Scattering transform on rolling windows of log returns
    # Use window size that's a power of 2 for FFT efficiency
    win_size = 64
    if len(log_returns) < win_size:
        win_size = max(32, 2 ** int(np.floor(np.log2(len(log_returns)))))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from kymatio.scattering1d.frontend.numpy_frontend import (
            ScatteringNumPy1D,
        )

        scatter = ScatteringNumPy1D(J=min(SCATTER_J, int(np.log2(win_size)) - 1), shape=(win_size,))

    # Compute scattering on overlapping windows
    step = max(1, win_size // 4)
    n_windows = max(1, (len(log_returns) - win_size) // step + 1)

    scatter_means = np.zeros(n_windows)
    scatter_vars = np.zeros(n_windows)
    scatter_skews = np.zeros(n_windows)

    for i in range(n_windows):
        start = i * step
        window = log_returns[start : start + win_size].astype(np.float32)
        window = window.reshape(1, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeffs = scatter(window)  # shape: (1, n_paths, n_time)

        # Flatten across paths and time
        flat = coeffs.flatten()
        scatter_means[i] = np.mean(flat)
        scatter_vars[i] = np.var(flat)
        scatter_skews[i] = float(skew(flat)) if len(flat) > 2 else 0.0

    # Z-score normalize
    for arr in [scatter_means, scatter_vars, scatter_skews]:
        std = np.std(arr)
        if std > 1e-10:
            arr[:] = (arr - np.mean(arr)) / std

    # Resample to coefficient-delta resolution
    features = np.column_stack([scatter_means, scatter_vars, scatter_skews])
    indices = np.linspace(0, len(features) - 1, n_deltas)
    resampled = np.zeros((n_deltas, N_FEATURES), dtype=np.float64)
    for j in range(N_FEATURES):
        resampled[:, j] = np.interp(
            indices, np.arange(len(features)), features[:, j]
        )

    return resampled


if __name__ == "__main__":
    run_feature_test(
        "wavelet_scattering", compute_wavelet_scattering, n_new_features=N_FEATURES
    )
