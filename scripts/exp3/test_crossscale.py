"""Experiment 3.4: Multi-Channel Cross-Scale Input.

Instead of processing each DWT level as a separate sequence,
construct a SINGLE sequence where each timestep has all levels as channels.

Current: [delta_L1[t]] per sequence (scalar, one level at a time)
New:     [delta_L1[t], delta_L2[t], delta_L5[t], approx_dir[t]] per timestep

Branch: exp3/wave2-crossscale
"""

from __future__ import annotations

import logging

import numpy as np
from scripts.exp3.evaluate_representation import evaluate_representation
from scripts.feature_tests.harness import (
    CONTEXT_LENGTH,
    N_CLASSES,
    PERCENTILES,
    _ohlcv_to_timeseries,
)

from wavecast.core.universe import DEFAULT_UNIVERSE
from wavecast.experiments.runner import SECTOR_ID_MAP
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
)
from wavecast.wavelets.dwt import decompose

logger = logging.getLogger(__name__)

CROSS_LEVELS = [1, 2, 5]
N_CHANNELS = len(CROSS_LEVELS) + 1  # +1 for approx_direction


def _build_crossscale_pipeline(prices, tickers, ctx_len=CONTEXT_LENGTH):
    """Build multi-channel cross-scale dataset.

    Downsamples fine levels to match the coarsest level's resolution.
    Each timestep has 4 channels: [delta_L1, delta_L2, delta_L5, approx_dir].
    """
    ac_map = {
        a.ticker: SECTOR_ID_MAP.get(a.sector.value, 0)
        for a in DEFAULT_UNIVERSE.assets if a.sector
    }

    all_X = []
    all_rets = []
    all_valid = []
    all_lvls = []

    for ticker in tickers:
        ts = prices[ticker]
        decomp = decompose(ts, level=5)

        # Get detail coefficients at each level
        level_deltas = {}
        for lvl in CROSS_LEVELS:
            detail = decomp.detail_at_level(lvl)
            if len(detail) > 1:
                level_deltas[lvl] = np.diff(detail)

        if not all(lvl in level_deltas for lvl in CROSS_LEVELS):
            continue

        # Approx direction
        approx = decomp.approximation
        approx_delta = np.diff(approx) if len(approx) > 1 else np.zeros(1)

        # Downsample all to the COARSEST level's length (level 5)
        target_len = len(level_deltas[max(CROSS_LEVELS)])
        if target_len <= ctx_len:
            continue

        channels = np.zeros((target_len, N_CHANNELS), dtype=np.float64)
        for ch_idx, lvl in enumerate(CROSS_LEVELS):
            src = level_deltas[lvl]
            if len(src) == target_len:
                channels[:, ch_idx] = src
            else:
                # Downsample via averaging blocks
                ratio = len(src) / target_len
                for t in range(target_len):
                    start = int(t * ratio)
                    end = int((t + 1) * ratio)
                    end = max(end, start + 1)
                    channels[t, ch_idx] = np.mean(src[start:end])

        # Approx direction channel
        if len(approx_delta) == target_len:
            channels[:, -1] = np.sign(approx_delta)
        else:
            ratio = len(approx_delta) / target_len
            for t in range(target_len):
                idx = min(int(t * ratio), len(approx_delta) - 1)
                channels[t, -1] = np.sign(approx_delta[idx])

        # Z-normalize each channel independently
        for ch in range(N_CHANNELS - 1):  # Don't normalize sign channel
            std = np.std(channels[:, ch])
            if std > 1e-10:
                channels[:, ch] = (channels[:, ch] - np.mean(channels[:, ch])) / std

        ac_id = ac_map.get(ticker, 0)

        # Sliding windows
        for i in range(target_len - ctx_len):
            window = channels[i:i + ctx_len]  # (ctx_len, N_CHANNELS)

            # Flatten: first ctx_len values are channel 0, etc.
            # Layout: [ch0_t0..ch0_tL, ch1_t0..ch1_tL, ..., chN_t0..chN_tL, level, ac]
            flat = window.T.ravel()  # (N_CHANNELS * ctx_len,)
            row = np.concatenate([flat, [0, ac_id]])  # level=0 (cross-scale)
            all_X.append(row)

            # Return from coarsest level (level 5) position
            target_pos = i + ctx_len
            span = 2 ** max(CROSS_LEVELS)
            bar_start = target_pos * span
            bar_end = bar_start + span
            pv = ts.values
            if bar_end < len(pv) and pv[bar_start] > 0:
                ret = (pv[bar_end] - pv[bar_start]) / pv[bar_start]
                all_rets.append(ret)
                all_valid.append(True)
            else:
                all_rets.append(np.nan)
                all_valid.append(False)

            all_lvls.append(0)

    X = np.array(all_X, dtype=np.float64)
    rets = np.array(all_rets, dtype=np.float64)
    valid = np.array(all_valid, dtype=np.bool_)
    lvls = np.array(all_lvls, dtype=np.int64)

    return X, rets, valid, lvls


def build_crossscale_dataset(train_ohlcv, test_ohlcv):
    """Build cross-scale dataset for evaluate_representation."""
    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))

    X_train, tr_rets, tr_valid, tr_lvls = _build_crossscale_pipeline(
        train_prices, tickers,
    )
    X_test, te_rets, te_valid, te_lvls = _build_crossscale_pipeline(
        test_prices, tickers,
    )

    logger.info("Cross-scale: %d train, %d test (N_CHANNELS=%d)",
                len(X_train), len(X_test), N_CHANNELS)

    # Use global boundaries (single level=0)
    boundaries = compute_quantile_boundaries(
        tr_rets, tr_lvls, tr_valid, list(PERCENTILES), per_level=False,
    )
    y_train = assign_quantile_labels(tr_rets, tr_lvls, boundaries)
    y_test = assign_quantile_labels(te_rets, te_lvls, boundaries)

    meta = {
        "context_length": CONTEXT_LENGTH,
        "n_classes": N_CLASSES,
        "n_aux": N_CHANNELS - 1,  # channel 0 is the main context, rest are aux
        "detail_levels": CROSS_LEVELS,
        "n_channels": N_CHANNELS,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return (X_train, y_train, tr_rets, tr_valid, tr_lvls,
            X_test, y_test, te_rets, te_valid, te_lvls, meta)


if __name__ == "__main__":
    results = evaluate_representation(
        name="crossscale",
        build_dataset_fn=build_crossscale_dataset,
        n_seeds=3,
        model_config={"n_aux_features": N_CHANNELS - 1},
        run_baseline=True,
    )
    print(f"\nFinal verdict: {results['verdict']}")
