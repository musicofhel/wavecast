"""Experiment 3.5: Range-DWT as Second Channel.

In addition to DWT of close prices, compute DWT of bar range (high - low).
Stack as two-channel input: [close_delta_L1[t], range_delta_L1[t]].

Branch: exp3/wave2-range-dwt
"""

from __future__ import annotations

import logging

import numpy as np
from scripts.exp3.evaluate_representation import evaluate_representation
from scripts.feature_tests.harness import (
    CONTEXT_LENGTH,
    DETAIL_LEVELS,
    N_CLASSES,
    PERCENTILES,
    _ohlcv_to_timeseries,
)

from wavecast.core.types import TimeSeries
from wavecast.core.universe import DEFAULT_UNIVERSE
from wavecast.data.auxiliary_features import compute_detail_auxiliary_features
from wavecast.data.continuous_dataset import build_continuous_dataset
from wavecast.experiments.runner import SECTOR_ID_MAP
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
)
from wavecast.wavelets.dwt import decompose

logger = logging.getLogger(__name__)

# 4 base aux + 1 range delta = 5 aux features
N_AUX_WITH_RANGE = 5


def _build_range_dwt_pipeline(prices, ohlcv, tickers, ctx_len=CONTEXT_LENGTH):
    """Build D1 pipeline with range-DWT as additional aux feature."""
    ac_map = {
        a.ticker: SECTOR_ID_MAP.get(a.sector.value, 0)
        for a in DEFAULT_UNIVERSE.assets if a.sector
    }

    coeff_series = {}
    aux_series = {}

    for ticker in tickers:
        ts = prices[ticker]
        decomp = decompose(ts, level=5)

        # Also decompose bar range
        if ticker not in ohlcv:
            continue
        df = ohlcv[ticker]
        bar_range = (df["high"] - df["low"]).to_numpy(dtype=np.float64)
        if len(bar_range) < 100:
            continue

        range_ts = TimeSeries(
            values=bar_range,
            timestamps=df["timestamp"].to_numpy(dtype="datetime64[ns]"),
            ticker=ticker, interval="1h",
        )
        range_decomp = decompose(range_ts, level=5)

        for lvl in DETAIL_LEVELS:
            detail = decomp.detail_at_level(lvl)
            close_deltas = np.diff(detail) if len(detail) > 1 else detail
            if len(close_deltas) <= ctx_len:
                continue

            range_detail = range_decomp.detail_at_level(lvl)
            range_deltas = np.diff(range_detail) if len(range_detail) > 1 else range_detail

            coeff_series[(ticker, lvl)] = close_deltas

            # Standard 4 aux features
            existing_aux = compute_detail_auxiliary_features(
                detail, decomp.approximation,
            )

            # Add range delta as 5th aux feature
            n_deltas = len(close_deltas)
            if len(range_deltas) != n_deltas:
                # Resample range deltas to match close deltas
                indices = np.linspace(0, len(range_deltas) - 1, n_deltas)
                range_deltas_resampled = np.interp(
                    indices, np.arange(len(range_deltas)), range_deltas,
                )
            else:
                range_deltas_resampled = range_deltas

            # Z-normalize range deltas
            std = np.std(range_deltas_resampled)
            if std > 1e-10:
                range_deltas_resampled = (
                    range_deltas_resampled - np.mean(range_deltas_resampled)
                ) / std

            # Concatenate as 5th feature
            range_col = range_deltas_resampled.reshape(-1, 1)
            aux_series[(ticker, lvl)] = np.concatenate(
                [existing_aux, range_col], axis=1,
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
        span = 2 ** w.level
        bar_start = target_pos * span
        bar_end = bar_start + span
        pv = prices[w.ticker].values
        if bar_end < len(pv) and pv[bar_start] > 0:
            rets[i] = (pv[bar_end] - pv[bar_start]) / pv[bar_start]
            valid[i] = True

    lvls = np.array([w.level for w in ds.windows], dtype=np.int64)

    # Build X
    ctx_arr, lvl_arr, ac_arr = ds.to_arrays()
    # Build aux windows
    aux_wins = []
    for (_t, _l), aux in sorted(aux_series.items()):
        if len(aux) <= ctx_len:
            continue
        for j in range(len(aux) - ctx_len):
            aux_wins.append(aux[j:j + ctx_len])
    if aux_wins:
        aux_win = np.array(aux_wins, dtype=np.float64)
    else:
        aux_win = np.empty((0, ctx_len, N_AUX_WITH_RANGE), dtype=np.float64)

    aux_flat = aux_win.reshape(len(aux_win), -1)
    X = np.column_stack([ctx_arr, aux_flat, lvl_arr, ac_arr])

    return X, rets, valid, lvls


def build_range_dwt_dataset(train_ohlcv, test_ohlcv):
    """Build range-DWT dataset for evaluate_representation."""
    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))

    X_train, tr_rets, tr_valid, tr_lvls = _build_range_dwt_pipeline(
        train_prices, train_ohlcv, tickers,
    )
    X_test, te_rets, te_valid, te_lvls = _build_range_dwt_pipeline(
        test_prices, test_ohlcv, tickers,
    )

    boundaries = compute_quantile_boundaries(
        tr_rets, tr_lvls, tr_valid, list(PERCENTILES), per_level=True,
    )
    y_train = assign_quantile_labels(tr_rets, tr_lvls, boundaries)
    y_test = assign_quantile_labels(te_rets, te_lvls, boundaries)

    logger.info("Range-DWT: %d train, %d test", len(X_train), len(X_test))

    meta = {
        "context_length": CONTEXT_LENGTH,
        "n_classes": N_CLASSES,
        "n_aux": N_AUX_WITH_RANGE,
        "detail_levels": list(DETAIL_LEVELS),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return (X_train, y_train, tr_rets, tr_valid, tr_lvls,
            X_test, y_test, te_rets, te_valid, te_lvls, meta)


if __name__ == "__main__":
    results = evaluate_representation(
        name="range_dwt",
        build_dataset_fn=build_range_dwt_dataset,
        n_seeds=3,
        run_baseline=True,
    )
    print(f"\nFinal verdict: {results['verdict']}")
