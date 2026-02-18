#!/usr/bin/env python3
"""Feature Test 17: Ordinal Network Features.

Hypothesis: Build graphs from temporal succession of ordinal patterns
(permutation patterns) in D1 coefficients. Network statistics (entropy,
complexity, determinism) characterize regime type without distributional
assumptions.

3 new features per timestep:
  - ordinal_entropy: Shannon entropy of ordinal pattern frequencies
  - ordinal_complexity: statistical complexity (disequilibrium * entropy)
  - determinism: predictability of pattern transitions

Usage:
    python -m scripts.feature_tests.test_ordinal_net
"""

from __future__ import annotations

from itertools import permutations
from math import factorial

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scripts.feature_tests.harness import run_feature_test  # noqa: I001

N_FEATURES = 3
PATTERN_ORDER = 3
WINDOW_SIZE = 32


def _ordinal_pattern(segment: NDArray) -> tuple[int, ...]:
    """Convert a segment to its ordinal pattern (permutation rank)."""
    return tuple(int(x) for x in np.argsort(segment))


def _pattern_distribution(patterns: list[tuple[int, ...]], order: int) -> NDArray:
    """Compute probability distribution over ordinal patterns."""
    n_patterns = factorial(order)
    all_perms = list(permutations(range(order)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}

    counts = np.zeros(n_patterns, dtype=np.float64)
    for pat in patterns:
        idx = perm_to_idx.get(pat)
        if idx is not None:
            counts[idx] += 1

    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def _shannon_entropy(probs: NDArray) -> float:
    """Shannon entropy of a probability distribution."""
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def _statistical_complexity(probs: NDArray) -> float:
    """Jensen-Shannon complexity: disequilibrium * entropy."""
    n = len(probs)
    if n <= 1:
        return 0.0
    uniform = np.ones(n) / n
    # Jensen-Shannon divergence
    m = 0.5 * (probs + uniform)
    kl_pm = np.sum(probs[probs > 0] * np.log2(probs[probs > 0] / m[probs > 0]))
    kl_um = np.sum(uniform * np.log2(uniform / m))
    jsd = 0.5 * (kl_pm + kl_um)
    # Normalize by max possible JSD
    max_entropy = np.log2(n)
    entropy = _shannon_entropy(probs)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return float(jsd * norm_entropy)


def _transition_determinism(patterns: list[tuple[int, ...]], order: int) -> float:
    """Determinism: how predictable are pattern transitions."""
    if len(patterns) < 2:
        return 0.0
    n_patterns = factorial(order)
    all_perms = list(permutations(range(order)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}

    # Build transition matrix
    trans = np.zeros((n_patterns, n_patterns), dtype=np.float64)
    for i in range(len(patterns) - 1):
        src = perm_to_idx.get(patterns[i])
        dst = perm_to_idx.get(patterns[i + 1])
        if src is not None and dst is not None:
            trans[src, dst] += 1

    # Normalize rows
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1.0)
    trans_probs = trans / row_sums

    # Determinism = average max transition probability
    active_rows = trans.sum(axis=1) > 0
    if not active_rows.any():
        return 0.0
    max_probs = trans_probs.max(axis=1)[active_rows]
    return float(np.mean(max_probs))


def compute_ordinal_network(
    ohlcv_df: pd.DataFrame,
    detail_coeffs: NDArray,
    approx_coeffs: NDArray,
    level: int,
) -> NDArray:
    """Ordinal network features at coefficient-delta resolution.

    Returns (n_deltas, 3) where n_deltas = len(detail_coeffs) - 1.
    """
    n_deltas = len(detail_coeffs) - 1
    if n_deltas <= 0:
        return np.empty((0, N_FEATURES), dtype=np.float64)

    # Use D1 coefficient deltas directly
    deltas = np.diff(detail_coeffs)

    if len(deltas) < WINDOW_SIZE + PATTERN_ORDER:
        return np.zeros((n_deltas, N_FEATURES), dtype=np.float64)

    step = max(1, WINDOW_SIZE // 4)
    n_windows = max(1, (len(deltas) - WINDOW_SIZE) // step + 1)

    entropy_vals = np.zeros(n_windows)
    complexity_vals = np.zeros(n_windows)
    determ_vals = np.zeros(n_windows)

    for i in range(n_windows):
        start = i * step
        window = deltas[start : start + WINDOW_SIZE]

        # Extract ordinal patterns
        patterns = []
        for j in range(len(window) - PATTERN_ORDER + 1):
            pat = _ordinal_pattern(window[j : j + PATTERN_ORDER])
            patterns.append(pat)

        if not patterns:
            continue

        dist = _pattern_distribution(patterns, PATTERN_ORDER)
        entropy_vals[i] = _shannon_entropy(dist)
        complexity_vals[i] = _statistical_complexity(dist)
        determ_vals[i] = _transition_determinism(patterns, PATTERN_ORDER)

    # Z-score normalize
    for arr in [entropy_vals, complexity_vals, determ_vals]:
        std = np.std(arr)
        if std > 1e-10:
            arr[:] = (arr - np.mean(arr)) / std

    # Resample to coefficient-delta resolution
    features = np.column_stack([entropy_vals, complexity_vals, determ_vals])
    indices = np.linspace(0, len(features) - 1, n_deltas)
    resampled = np.zeros((n_deltas, N_FEATURES), dtype=np.float64)
    for j in range(N_FEATURES):
        resampled[:, j] = np.interp(
            indices, np.arange(len(features)), features[:, j]
        )

    return resampled


if __name__ == "__main__":
    run_feature_test(
        "ordinal_network", compute_ordinal_network, n_new_features=N_FEATURES
    )
