"""Experiment 3.3: Extended Context Window.

Increase context_length from 16 to 32, 48, 64. Same D1 representation.
D1 has 124K training windows — 6.5x more data than Phase 3's SAX test
where context=64 hurt due to sample starvation.

Branch: exp3/wave1-context
"""

from __future__ import annotations

import logging

import numpy as np
from scripts.exp3.evaluate_representation import (
    evaluate_representation,
)
from scripts.feature_tests.harness import (
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
from wavecast.wavelets.dwt import decompose

logger = logging.getLogger(__name__)


def _make_context_builder(ctx_len: int):
    """Create a dataset builder for a specific context length."""

    def build_context_dataset(train_ohlcv, test_ohlcv):
        train_prices = _ohlcv_to_timeseries(train_ohlcv)
        test_prices = _ohlcv_to_timeseries(test_ohlcv)
        tickers = sorted(set(train_ohlcv) & set(test_ohlcv))

        ac_map = {
            a.ticker: SECTOR_ID_MAP.get(a.sector.value, 0)
            for a in DEFAULT_UNIVERSE.assets if a.sector
        }

        def _build_pipeline(prices, tkrs):
            coeff_series = {}
            aux_series = {}
            for ticker in tkrs:
                ts = prices[ticker]
                decomp = decompose(ts, level=5)
                for lvl in DETAIL_LEVELS:
                    detail = decomp.detail_at_level(lvl)
                    deltas = np.diff(detail) if len(detail) > 1 else detail
                    if len(deltas) <= ctx_len:
                        continue
                    coeff_series[(ticker, lvl)] = deltas
                    aux_series[(ticker, lvl)] = compute_detail_auxiliary_features(
                        detail, decomp.approximation,
                    )
            if not coeff_series:
                empty = np.empty((0,))
                return empty, empty, np.empty((0,), dtype=np.bool_), np.empty((0,), dtype=np.int64)

            ds = build_continuous_dataset(coeff_series, ctx_len, ac_map, normalize=True)

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

            ctx_arr, lvl_arr, ac_arr = ds.to_arrays()
            # Build aux windows for this context length
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

        X_train, tr_rets, tr_valid, tr_lvls = _build_pipeline(train_prices, tickers)
        X_test, te_rets, te_valid, te_lvls = _build_pipeline(test_prices, tickers)

        boundaries = compute_quantile_boundaries(
            tr_rets, tr_lvls, tr_valid, list(PERCENTILES), per_level=True,
        )
        y_train = assign_quantile_labels(tr_rets, tr_lvls, boundaries)
        y_test = assign_quantile_labels(te_rets, te_lvls, boundaries)

        logger.info("Context=%d: %d train, %d test (baseline ctx=16 has ~124K train)",
                    ctx_len, len(X_train), len(X_test))

        meta = {
            "context_length": ctx_len,
            "n_classes": N_CLASSES,
            "n_aux": N_AUX_FEATURES,
            "detail_levels": list(DETAIL_LEVELS),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        return (X_train, y_train, tr_rets, tr_valid, tr_lvls,
                X_test, y_test, te_rets, te_valid, te_lvls, meta)

    return build_context_dataset


if __name__ == "__main__":
    context_lengths = [32, 48, 64]
    all_results = {}

    for ctx in context_lengths:
        logger.info("\n" + "=" * 80)
        logger.info("Testing context_length=%d", ctx)
        logger.info("=" * 80)

        builder = _make_context_builder(ctx)
        results = evaluate_representation(
            name=f"context_{ctx}",
            build_dataset_fn=builder,
            n_seeds=3,
            run_baseline=True,
        )
        all_results[f"context_{ctx}"] = results

    # Summary
    print("\n" + "=" * 80)
    print("EXTENDED CONTEXT SUMMARY")
    print("=" * 80)
    for name, r in all_results.items():
        c = r["challenger_avg"]
        print(f"  {name}: econ_dir={c['econ_dir']:.1%} trans={c['transition_acc']:.1%} "
              f"large={c['large_move_acc']:.1%} flat={c['pred_dist']['flat']:.1%} "
              f"sharpe={c['sharpe_costs']:+.3f} → {r['verdict']}")
