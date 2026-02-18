#!/usr/bin/env python3
"""Feature Test 16: Topological Data Analysis (TDA) Features.

Hypothesis: Persistent homology captures structural properties of the time
series (loops, holes, connected components) invisible to standard features.
Persistence diagrams encode multi-scale topological information.

3 new features per timestep:
  - persistent_entropy: Shannon entropy of persistence diagram lifetimes
  - amplitude: L2 norm of persistence diagram
  - n_points: number of topological features (connected components + loops)

Uses giotto-tda for Vietoris-Rips persistence on time-delay embeddings.

Usage:
    python -m scripts.feature_tests.test_tda
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scripts.feature_tests.harness import run_feature_test  # noqa: I001

N_FEATURES = 3
EMBEDDING_DIM = 3
EMBEDDING_DELAY = 1
WINDOW_SIZE = 32


def _persistent_entropy(diagram: NDArray) -> float:
    """Compute persistent entropy from a persistence diagram."""
    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0
    total = np.sum(lifetimes)
    if total < 1e-12:
        return 0.0
    probs = lifetimes / total
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _amplitude(diagram: NDArray) -> float:
    """L2 norm of persistence diagram lifetimes."""
    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0
    return float(np.sqrt(np.sum(lifetimes**2)))


def compute_tda_features(
    ohlcv_df: pd.DataFrame,
    detail_coeffs: NDArray,
    approx_coeffs: NDArray,
    level: int,
) -> NDArray:
    """TDA features at coefficient-delta resolution.

    Returns (n_deltas, 3) where n_deltas = len(detail_coeffs) - 1.
    """
    from gtda.homology import VietorisRipsPersistence

    n_deltas = len(detail_coeffs) - 1
    if n_deltas <= 0:
        return np.empty((0, N_FEATURES), dtype=np.float64)

    close = ohlcv_df["close"].values.astype(np.float64)
    log_returns = np.diff(np.log(np.maximum(close, 1e-10)))

    if len(log_returns) < WINDOW_SIZE + EMBEDDING_DIM:
        return np.zeros((n_deltas, N_FEATURES), dtype=np.float64)

    vr = VietorisRipsPersistence(
        homology_dimensions=[0, 1], max_edge_length=np.inf, n_jobs=1
    )

    step = max(1, WINDOW_SIZE // 4)
    n_windows = max(1, (len(log_returns) - WINDOW_SIZE) // step + 1)

    pe_vals = np.zeros(n_windows)
    amp_vals = np.zeros(n_windows)
    npt_vals = np.zeros(n_windows)

    for i in range(n_windows):
        start = i * step
        window = log_returns[start : start + WINDOW_SIZE]

        # Time-delay embedding
        n_embed = len(window) - (EMBEDDING_DIM - 1) * EMBEDDING_DELAY
        if n_embed < 4:
            continue
        embedded = np.zeros((n_embed, EMBEDDING_DIM))
        for d in range(EMBEDDING_DIM):
            embedded[:, d] = window[d * EMBEDDING_DELAY : d * EMBEDDING_DELAY + n_embed]

        # Compute persistence diagram
        diagrams = vr.fit_transform(embedded.reshape(1, *embedded.shape))
        diag = diagrams[0]  # shape: (n_features, 3) — birth, death, dimension

        # Filter to finite features
        finite_mask = np.isfinite(diag[:, 1])
        finite_diag = diag[finite_mask]

        pe_vals[i] = _persistent_entropy(finite_diag[:, :2])
        amp_vals[i] = _amplitude(finite_diag[:, :2])
        npt_vals[i] = float(len(finite_diag))

    # Z-score normalize
    for arr in [pe_vals, amp_vals, npt_vals]:
        std = np.std(arr)
        if std > 1e-10:
            arr[:] = (arr - np.mean(arr)) / std

    # Resample to coefficient-delta resolution
    features = np.column_stack([pe_vals, amp_vals, npt_vals])
    indices = np.linspace(0, len(features) - 1, n_deltas)
    resampled = np.zeros((n_deltas, N_FEATURES), dtype=np.float64)
    for j in range(N_FEATURES):
        resampled[:, j] = np.interp(
            indices, np.arange(len(features)), features[:, j]
        )

    return resampled


if __name__ == "__main__":
    run_feature_test("tda_features", compute_tda_features, n_new_features=N_FEATURES)
