"""Return target computation for SAX-context → return-quantile prediction.

Computes actual returns from price data using the temporal mapping validated
in scripts/model_audit.py (lines 156-173):

    token_pos = sample_idx + context_length
    coeff_idx = token_pos * n_coeffs // n_symbols
    bar_start = coeff_idx * 2^level
    bar_end   = bar_start + 2^level
    return    = (price[bar_end] - price[bar_start]) / price[bar_start]

Quantile boundaries are computed per-level (level 5 returns span ~0.5-5%,
level 1 spans ~0.01-0.5%) to avoid misclassification from identical thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ReturnTargetResult:
    """Result of return target computation.

    Attributes:
        returns: Per-sample actual returns (NaN where not computable).
        quantile_labels: 0..n_classes-1 class labels (0 for invalid samples).
        boundaries: Per-level quantile boundary values.
        valid_mask: True where return was computable.
    """

    returns: NDArray[np.float64]
    quantile_labels: NDArray[np.int64]
    boundaries: dict[int, NDArray[np.float64]]
    valid_mask: NDArray[np.bool_]


def compute_sample_returns(
    token_positions: NDArray[np.int64],
    levels: NDArray[np.int64],
    price_values: NDArray[np.float64],
    n_coeffs: NDArray[np.int64],
    n_symbols: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Compute per-sample returns using the temporal mapping.

    Args:
        token_positions: Token position in the symbol sequence per sample.
        levels: DWT level per sample (1-5).
        price_values: Raw price array for the ticker.
        n_coeffs: Number of DWT coefficients per sample's level.
        n_symbols: Number of SAX symbols per sample's level.

    Returns:
        (returns, valid_mask) where returns[i] is NaN if not computable.
    """
    n = len(token_positions)
    returns = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=np.bool_)

    n_prices = len(price_values)

    for i in range(n):
        tp = int(token_positions[i])
        lvl = int(levels[i])
        nc = int(n_coeffs[i])
        ns = int(n_symbols[i])

        if ns <= 0:
            continue

        coeff_idx = tp * nc // ns
        bar_start = coeff_idx * (2**lvl)
        bar_end = bar_start + (2**lvl)

        if (
            bar_end < n_prices
            and bar_start < n_prices
            and price_values[bar_start] != 0.0
        ):
            returns[i] = (
                price_values[bar_end] - price_values[bar_start]
            ) / price_values[bar_start]
            valid[i] = True

    return returns, valid


def compute_quantile_boundaries(
    returns: NDArray[np.float64],
    levels: NDArray[np.int64],
    valid_mask: NDArray[np.bool_],
    percentiles: list[float] | None = None,
    per_level: bool = True,
) -> dict[int, NDArray[np.float64]]:
    """Compute quantile boundaries for return classification.

    Default percentiles [10, 30, 70, 90] produce 5 classes with approximate
    distribution 10/20/40/20/10%.

    Args:
        returns: Per-sample returns.
        levels: Per-sample DWT levels.
        valid_mask: True where return is valid.
        percentiles: Boundary percentiles (default: [10, 30, 70, 90]).
        per_level: If True, compute separate boundaries per DWT level.

    Returns:
        Dict mapping level -> boundary array. Level -1 is used for global
        (non-per-level) boundaries.
    """
    if percentiles is None:
        percentiles = [10.0, 30.0, 70.0, 90.0]

    boundaries: dict[int, NDArray[np.float64]] = {}

    if per_level:
        unique_levels = sorted(set(int(v) for v in levels[valid_mask]))
        for lvl in unique_levels:
            mask = valid_mask & (levels == lvl)
            level_returns = returns[mask]
            if len(level_returns) < 5:
                # Too few samples; use global boundaries as fallback
                boundaries[lvl] = np.percentile(
                    returns[valid_mask], percentiles
                ).astype(np.float64)
            else:
                boundaries[lvl] = np.percentile(
                    level_returns, percentiles
                ).astype(np.float64)
    else:
        valid_returns = returns[valid_mask]
        global_bounds = np.percentile(valid_returns, percentiles).astype(
            np.float64
        )
        # Store under level -1 for global
        boundaries[-1] = global_bounds

    return boundaries


def assign_quantile_labels(
    returns: NDArray[np.float64],
    levels: NDArray[np.int64],
    boundaries: dict[int, NDArray[np.float64]],
) -> NDArray[np.int64]:
    """Assign quantile class labels to returns.

    Classes: 0=strong_down, 1=down, 2=flat, 3=up, 4=strong_up
    (for 5-class with 4 boundaries).

    Args:
        returns: Per-sample returns.
        levels: Per-sample DWT levels.
        boundaries: Per-level (or global with key -1) boundary arrays.

    Returns:
        Array of class labels 0..n_classes-1.
    """
    n = len(returns)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        ret = returns[i]
        if np.isnan(ret):
            labels[i] = 2  # default to FLAT for invalid
            continue

        lvl = int(levels[i])
        # Use per-level boundaries, fall back to global (-1)
        bounds = boundaries.get(lvl, boundaries.get(-1))
        if bounds is None:
            labels[i] = 2
            continue

        # np.searchsorted: returns index where ret would be inserted
        # For boundaries [b0, b1, b2, b3]:
        #   ret < b0 → class 0 (strong_down)
        #   b0 <= ret < b1 → class 1 (down)
        #   b1 <= ret < b2 → class 2 (flat)
        #   b2 <= ret < b3 → class 3 (up)
        #   ret >= b3 → class 4 (strong_up)
        labels[i] = int(np.searchsorted(bounds, ret))

    return labels


def compute_return_targets(
    token_positions: NDArray[np.int64],
    levels: NDArray[np.int64],
    price_values: NDArray[np.float64],
    n_coeffs: NDArray[np.int64],
    n_symbols: NDArray[np.int64],
    percentiles: list[float] | None = None,
    per_level: bool = True,
) -> ReturnTargetResult:
    """End-to-end return target computation.

    Computes returns, boundaries, and quantile labels in one call.

    Args:
        token_positions: Token position per sample.
        levels: DWT level per sample.
        price_values: Raw price array.
        n_coeffs: Number of coefficients per sample's level.
        n_symbols: Number of SAX symbols per sample's level.
        percentiles: Quantile boundary percentiles.
        per_level: Whether to use per-level boundaries.

    Returns:
        ReturnTargetResult with all computed fields.
    """
    returns, valid_mask = compute_sample_returns(
        token_positions, levels, price_values, n_coeffs, n_symbols
    )

    boundaries = compute_quantile_boundaries(
        returns, levels, valid_mask, percentiles, per_level
    )

    labels = assign_quantile_labels(returns, levels, boundaries)

    return ReturnTargetResult(
        returns=returns,
        quantile_labels=labels,
        boundaries=boundaries,
        valid_mask=valid_mask,
    )
