"""Experiment 3.6: MODWT Replacement.

Replace standard DWT (which decimates) with MODWT (Maximal Overlap DWT),
which produces N coefficients at every level without downsampling.

Benefits: shift-invariant, more data, direct temporal alignment.
Risk: redundancy from overlapping coefficients may cause overfitting.

Branch: exp3/wave2-modwt
"""

from __future__ import annotations

import logging

import numpy as np
import pywt
from numpy.typing import NDArray
from scripts.exp3.evaluate_representation import evaluate_representation
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
from wavecast.experiments.runner import SECTOR_ID_MAP
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
)

logger = logging.getLogger(__name__)


def _modwt_decompose(values: NDArray, wavelet: str = "db4", level: int = 5):
    """MODWT via PyWavelets stationary wavelet transform.

    Returns dict mapping level -> detail coefficients (full length, no decimation).
    Also returns approximation coefficients.
    """
    # Pad to next power of 2 if needed (swt requirement)
    n = len(values)
    padded_len = 1
    while padded_len < n:
        padded_len *= 2

    padded = np.pad(values, (0, padded_len - n), mode="reflect") if padded_len > n else values

    try:
        result = pywt.swt(padded, wavelet, level=level, trim_approx=True)
    except ValueError:
        # If level is too high for the data length, reduce
        max_level = pywt.swt_max_level(padded_len)
        actual_level = min(level, max_level)
        if actual_level < 1:
            return {}, np.array([])
        result = pywt.swt(padded, wavelet, level=actual_level, trim_approx=True)

    # result is list: [approx, detail_level_N, ..., detail_level_1]
    # Trim back to original length
    details = {}
    approx = result[0][:n]
    for i, coeff in enumerate(result[1:]):
        lvl = len(result) - 1 - i  # Levels from coarsest to finest
        details[lvl] = coeff[:n]

    return details, approx


def _build_modwt_pipeline(prices, tickers, ctx_len=CONTEXT_LENGTH):
    """Build D1-equivalent pipeline using MODWT instead of DWT."""
    ac_map = {
        a.ticker: SECTOR_ID_MAP.get(a.sector.value, 0)
        for a in DEFAULT_UNIVERSE.assets if a.sector
    }

    coeff_series = {}
    aux_series = {}

    for ticker in tickers:
        ts = prices[ticker]
        details, approx = _modwt_decompose(ts.values)

        for lvl in DETAIL_LEVELS:
            if lvl not in details:
                continue
            detail = details[lvl]
            deltas = np.diff(detail) if len(detail) > 1 else detail
            if len(deltas) <= ctx_len:
                continue

            coeff_series[(ticker, lvl)] = deltas

            # Aux features from MODWT coefficients
            aux_series[(ticker, lvl)] = compute_detail_auxiliary_features(
                detail, approx,
            )

    if not coeff_series:
        empty = np.empty((0,))
        return empty, empty, np.empty((0,), dtype=np.bool_), np.empty((0,), dtype=np.int64)

    ds = build_continuous_dataset(coeff_series, ctx_len, ac_map, normalize=True)

    # Returns
    rets = np.full(len(ds.windows), np.nan)
    valid = np.zeros(len(ds.windows), dtype=np.bool_)
    for i, w in enumerate(ds.windows):
        key = (w.ticker, w.level)
        if key not in coeff_series or w.ticker not in prices:
            continue
        target_pos = w.token_position
        if target_pos >= len(coeff_series[key]):
            continue
        # For MODWT, coefficient index maps 1:1 to price bar index
        # (no decimation, so span=1 for level 1, not 2^level)
        # Actually, return should still be measured over the level's timescale
        span = 2 ** w.level
        bar_start = target_pos
        bar_end = bar_start + span
        pv = prices[w.ticker].values
        if bar_end < len(pv) and pv[bar_start] > 0:
            rets[i] = (pv[bar_end] - pv[bar_start]) / pv[bar_start]
            valid[i] = True

    lvls = np.array([w.level for w in ds.windows], dtype=np.int64)

    # Build X
    ctx_arr, lvl_arr, ac_arr = ds.to_arrays()
    aux_wins = []
    for (_t, _l), aux in sorted(aux_series.items()):
        if len(aux) <= ctx_len:
            continue
        for j in range(len(aux) - ctx_len):
            aux_wins.append(aux[j:j + ctx_len])
    if aux_wins:
        aux_win = np.array(aux_wins, dtype=np.float64)
    else:
        aux_win = np.empty((0, ctx_len, N_AUX_FEATURES), dtype=np.float64)

    aux_flat = aux_win.reshape(len(aux_win), -1)
    X = np.column_stack([ctx_arr, aux_flat, lvl_arr, ac_arr])

    return X, rets, valid, lvls


def build_modwt_dataset(train_ohlcv, test_ohlcv):
    """Build MODWT dataset for evaluate_representation."""
    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))

    X_train, tr_rets, tr_valid, tr_lvls = _build_modwt_pipeline(
        train_prices, tickers,
    )
    X_test, te_rets, te_valid, te_lvls = _build_modwt_pipeline(
        test_prices, tickers,
    )

    boundaries = compute_quantile_boundaries(
        tr_rets, tr_lvls, tr_valid, list(PERCENTILES), per_level=True,
    )
    y_train = assign_quantile_labels(tr_rets, tr_lvls, boundaries)
    y_test = assign_quantile_labels(te_rets, te_lvls, boundaries)

    logger.info("MODWT: %d train, %d test (vs ~124K for DWT)", len(X_train), len(X_test))

    meta = {
        "context_length": CONTEXT_LENGTH,
        "n_classes": N_CLASSES,
        "n_aux": N_AUX_FEATURES,
        "detail_levels": list(DETAIL_LEVELS),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "transform": "MODWT (stationary wavelet transform)",
    }

    return (X_train, y_train, tr_rets, tr_valid, tr_lvls,
            X_test, y_test, te_rets, te_valid, te_lvls, meta)


if __name__ == "__main__":
    results = evaluate_representation(
        name="modwt",
        build_dataset_fn=build_modwt_dataset,
        n_seeds=3,
        run_baseline=True,
    )
    print(f"\nFinal verdict: {results['verdict']}")
