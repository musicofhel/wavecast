"""Experiment 3.1: Fractional Differencing.

Replace np.diff(coefficients) with fracdiff(coefficients, d=d_optimal)
to preserve long-range memory while achieving stationarity.

Branch: exp3/wave1-fracdiff
"""

from __future__ import annotations

import logging

import numpy as np
from scripts.exp3.evaluate_representation import (
    evaluate_representation,
)
from scripts.feature_tests.harness import (
    CONTEXT_LENGTH,
    DETAIL_LEVELS,
    N_AUX_FEATURES,
    N_CLASSES,
    PERCENTILES,
    _ohlcv_to_timeseries,
)

from wavecast.core.universe import DEFAULT_UNIVERSE
from wavecast.data.auxiliary_features import compute_detail_auxiliary_features
from wavecast.data.continuous_dataset import build_continuous_dataset
from wavecast.data.fracdiff import fracdiff_coefficients
from wavecast.experiments.runner import SECTOR_ID_MAP
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
)
from wavecast.wavelets.dwt import decompose

logger = logging.getLogger(__name__)


def _build_fracdiff_pipeline(
    prices, ohlcv, tickers, d_values=None,
):
    """Build D1 pipeline with fracdiff instead of np.diff.

    Returns (X, returns, valid, levels, d_per_series) using the
    same structure as harness._build_d1_pipeline.
    """
    ac_map = {
        a.ticker: SECTOR_ID_MAP.get(a.sector.value, 0)
        for a in DEFAULT_UNIVERSE.assets if a.sector
    }

    coeff_series = {}
    aux_series = {}
    d_used = {}

    for ticker in tickers:
        ts = prices[ticker]
        decomp = decompose(ts, level=5)
        for lvl in DETAIL_LEVELS:
            detail = decomp.detail_at_level(lvl)
            if len(detail) <= CONTEXT_LENGTH + 1:
                continue

            # Fractional differencing instead of np.diff
            d_val = d_values.get((ticker, lvl)) if d_values else None
            fd, d_actual = fracdiff_coefficients(detail, d=d_val)
            d_used[(ticker, lvl)] = d_actual

            # fracdiff returns same length as input; drop first element
            # to match np.diff behavior (n-1 elements)
            deltas = fd[1:]

            if len(deltas) <= CONTEXT_LENGTH:
                continue

            coeff_series[(ticker, lvl)] = deltas

            # Aux features still computed from original detail coefficients
            existing_aux = compute_detail_auxiliary_features(
                detail, decomp.approximation,
            )
            aux_series[(ticker, lvl)] = existing_aux

    if not coeff_series:
        empty = np.empty((0,))
        return empty, empty, np.empty((0,), dtype=np.bool_), np.empty((0,), dtype=np.int64), d_used

    ds = build_continuous_dataset(coeff_series, CONTEXT_LENGTH, ac_map, normalize=True)

    # Returns computation (same as harness)
    rets = np.full(len(ds.windows), np.nan)
    valid = np.zeros(len(ds.windows), dtype=np.bool_)
    for i, w in enumerate(ds.windows):
        key = (w.ticker, w.level)
        if key not in coeff_series or w.ticker not in prices:
            continue
        target_pos = w.token_position
        if target_pos >= len(coeff_series[key]):
            continue
        span = 2 ** w.level
        bar_start = target_pos * span
        bar_end = bar_start + span
        pv = prices[w.ticker].values
        if bar_end < len(pv) and pv[bar_start] > 0:
            rets[i] = (pv[bar_end] - pv[bar_start]) / pv[bar_start]
            valid[i] = True

    lvls = np.array([w.level for w in ds.windows], dtype=np.int64)

    # Build X with aux features
    ctx_arr, lvl_arr, ac_arr = ds.to_arrays()
    aux_win = _build_aux_windows(aux_series, CONTEXT_LENGTH, N_AUX_FEATURES)
    aux_flat = aux_win.reshape(len(aux_win), -1)
    X = np.column_stack([ctx_arr, aux_flat, lvl_arr, ac_arr])

    return X, rets, valid, lvls, d_used


def _build_aux_windows(aux_series, context_length, n_features):
    windows = []
    for (_t, _l), aux in sorted(aux_series.items()):
        if len(aux) <= context_length:
            continue
        for i in range(len(aux) - context_length):
            windows.append(aux[i:i + context_length])
    if not windows:
        return np.empty((0, context_length, n_features), dtype=np.float64)
    return np.array(windows, dtype=np.float64)


def build_fracdiff_dataset(train_ohlcv, test_ohlcv):
    """Build fracdiff dataset for evaluate_representation."""
    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))

    # Find optimal d on TRAINING data only
    logger.info("Finding optimal d values on training data...")
    X_train, tr_rets, tr_valid, tr_lvls, d_values = _build_fracdiff_pipeline(
        train_prices, train_ohlcv, tickers, d_values=None,
    )

    # Log d values
    d_vals = list(d_values.values())
    logger.info("d values: min=%.2f, max=%.2f, mean=%.2f, median=%.2f",
                min(d_vals), max(d_vals), np.mean(d_vals), np.median(d_vals))
    for (ticker, lvl), d in sorted(d_values.items()):
        logger.info("  %s L%d: d=%.3f", ticker, lvl, d)

    # Apply same d values to test data
    X_test, te_rets, te_valid, te_lvls, _ = _build_fracdiff_pipeline(
        test_prices, test_ohlcv, tickers, d_values=d_values,
    )

    # Compute quantile boundaries from training
    boundaries = compute_quantile_boundaries(
        tr_rets, tr_lvls, tr_valid, list(PERCENTILES), per_level=True,
    )
    y_train = assign_quantile_labels(tr_rets, tr_lvls, boundaries)
    y_test = assign_quantile_labels(te_rets, te_lvls, boundaries)

    meta = {
        "context_length": CONTEXT_LENGTH,
        "n_classes": N_CLASSES,
        "n_aux": N_AUX_FEATURES,
        "detail_levels": list(DETAIL_LEVELS),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "d_values": {f"{t}_L{lvl}": d for (t, lvl), d in d_values.items()},
        "d_mean": float(np.mean(d_vals)),
        "d_median": float(np.median(d_vals)),
    }

    return (X_train, y_train, tr_rets, tr_valid, tr_lvls,
            X_test, y_test, te_rets, te_valid, te_lvls, meta)


if __name__ == "__main__":
    results = evaluate_representation(
        name="fracdiff",
        build_dataset_fn=build_fracdiff_dataset,
        n_seeds=3,
        run_baseline=True,
    )
    print(f"\nFinal verdict: {results['verdict']}")
