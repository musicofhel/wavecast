"""Auxiliary feature computation for augmented input representations.

Computes per-position features that encode information SAX destroys:
coefficient sign, magnitude, volatility context, and approximation trend direction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

N_AUX_FEATURES = 4


def compute_detail_auxiliary_features(
    detail_coeffs: NDArray,
    approx_coeffs: NDArray | None = None,
    rolling_window: int = 16,
) -> NDArray:
    """Compute per-position auxiliary features aligned with np.diff(detail_coeffs).

    Returns (n_coeffs - 1, 4) array. Features are aligned with the output of
    np.diff(detail_coeffs), i.e., feature[i] corresponds to delta[i].

    Feature channels:
        0: coeff_sign — sign of original coefficient at position i+1
        1: abs_coeff_zscore — z-scored |coefficient| at position i+1
        2: volatility_ratio — |delta[i]| / EMA(|delta|), capped at 5.0
        3: approx_direction — sign of approximation coefficient delta at mapped position

    Args:
        detail_coeffs: Raw detail coefficients (before differencing).
        approx_coeffs: Approximation coefficients for trend context.
        rolling_window: Window for EMA in volatility ratio computation.

    Returns:
        (n-1, 4) array of auxiliary features, where n = len(detail_coeffs).
    """
    n = len(detail_coeffs)
    if n < 2:
        return np.empty((0, N_AUX_FEATURES), dtype=np.float64)

    deltas = np.diff(detail_coeffs)
    n_deltas = len(deltas)
    features = np.zeros((n_deltas, N_AUX_FEATURES), dtype=np.float64)

    # Feature 0: sign of original coefficient (at position i+1 since deltas shift by 1)
    features[:, 0] = np.sign(detail_coeffs[1:])

    # Feature 1: z-scored absolute coefficient magnitude
    abs_coeffs = np.abs(detail_coeffs[1:])
    std = np.std(abs_coeffs)
    if std > 1e-10:
        features[:, 1] = (abs_coeffs - np.mean(abs_coeffs)) / std

    # Feature 2: volatility ratio — |delta| / EMA(|delta|)
    abs_deltas = np.abs(deltas)
    alpha = 2.0 / (rolling_window + 1)
    ema = np.zeros(n_deltas, dtype=np.float64)
    ema[0] = abs_deltas[0]
    for i in range(1, n_deltas):
        ema[i] = alpha * abs_deltas[i] + (1 - alpha) * ema[i - 1]
    mask = ema > 1e-10
    features[mask, 2] = np.minimum(abs_deltas[mask] / ema[mask], 5.0)

    # Feature 3: approximation direction
    if approx_coeffs is not None and len(approx_coeffs) > 1:
        approx_deltas = np.diff(approx_coeffs)
        if len(approx_deltas) > 0 and n_deltas > 0:
            ratio = len(approx_deltas) / n_deltas
            indices = np.minimum(
                (np.arange(n_deltas) * ratio).astype(np.int64),
                len(approx_deltas) - 1,
            )
            features[:, 3] = np.sign(approx_deltas[indices])

    return features
