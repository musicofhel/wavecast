"""Experiment 3.2: Fine-Grained Return Discretization.

Replace 5-class targets (strong_down/down/flat/up/strong_up) with
7-class or 11-class targets. The flat region becomes narrower,
reducing the flat attractor that plagued 7/46 prior experiments.

Tests 4 variants:
  - 7 bins, narrow flat (bin 3 only = flat)
  - 7 bins, wide flat (bins 2-4 = flat)
  - 11 bins, narrow flat (bin 5 only = flat)
  - 11 bins, wide flat (bins 4-6 = flat)

Branch: exp3/wave1-finegrain
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scripts.exp3.evaluate_representation import (
    evaluate_representation,
)
from scripts.feature_tests.harness import (
    CONTEXT_LENGTH,
    DETAIL_LEVELS,
    N_AUX_FEATURES,
    _build_d1_pipeline,
    _ohlcv_to_timeseries,
)

logger = logging.getLogger(__name__)


def _compute_fine_boundaries(
    returns: NDArray, levels: NDArray, valid: NDArray, n_bins: int,
) -> dict[int, NDArray]:
    """Compute quantile boundaries for n_bins classes."""
    percentiles = [100.0 * (i + 1) / n_bins for i in range(n_bins - 1)]
    boundaries = {}
    unique_levels = sorted(set(int(v) for v in levels[valid]))
    for lvl in unique_levels:
        mask = valid & (levels == lvl)
        level_returns = returns[mask]
        if len(level_returns) < 5:
            level_returns = returns[valid]
        boundaries[lvl] = np.percentile(level_returns, percentiles).astype(np.float64)
    return boundaries


def _assign_fine_labels(
    returns: NDArray, levels: NDArray, boundaries: dict[int, NDArray],
) -> NDArray:
    """Assign fine-grained bin labels."""
    n = len(returns)
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        ret = returns[i]
        if np.isnan(ret):
            # Default to middle bin
            mid_bin = len(next(iter(boundaries.values()))) // 2
            labels[i] = mid_bin
            continue
        lvl = int(levels[i])
        bounds = boundaries.get(lvl, next(iter(boundaries.values())))
        labels[i] = int(np.searchsorted(bounds, ret))
    return labels


def _make_finegrain_builder(n_bins: int):
    """Create a dataset builder for n_bins fine-grained classes."""

    def build_finegrain_dataset(train_ohlcv, test_ohlcv):
        train_prices = _ohlcv_to_timeseries(train_ohlcv)
        test_prices = _ohlcv_to_timeseries(test_ohlcv)
        tickers = sorted(set(train_ohlcv) & set(test_ohlcv))

        # Build X arrays using standard D1 pipeline
        X_train, tr_rets, tr_valid, tr_lvls, _, _ = _build_d1_pipeline(
            train_prices, None, tickers, None, 0,
        )
        X_test, te_rets, te_valid, te_lvls, _, _ = _build_d1_pipeline(
            test_prices, None, tickers, None, 0,
        )

        # Compute fine-grained boundaries from training data
        boundaries = _compute_fine_boundaries(tr_rets, tr_lvls, tr_valid, n_bins)
        y_train = _assign_fine_labels(tr_rets, tr_lvls, boundaries)
        y_test = _assign_fine_labels(te_rets, te_lvls, boundaries)

        # Log class distribution
        valid_labels = y_train[tr_valid]
        dist = np.bincount(valid_labels, minlength=n_bins)
        logger.info("Training class distribution (%d bins):", n_bins)
        for c in range(n_bins):
            logger.info("  bin %d: %d (%.1f%%)", c, dist[c],
                       100.0 * dist[c] / len(valid_labels) if len(valid_labels) > 0 else 0)

        meta = {
            "context_length": CONTEXT_LENGTH,
            "n_classes": n_bins,
            "n_aux": N_AUX_FEATURES,
            "detail_levels": list(DETAIL_LEVELS),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_bins": n_bins,
            "boundaries": {str(k): v.tolist() for k, v in boundaries.items()},
        }

        return (X_train, y_train, tr_rets, tr_valid, tr_lvls,
                X_test, y_test, te_rets, te_valid, te_lvls, meta)

    return build_finegrain_dataset


if __name__ == "__main__":
    configs = [
        (7, "finegrain_7"),
        (11, "finegrain_11"),
    ]

    all_results = {}
    for n_bins, name in configs:
        logger.info("\n" + "=" * 80)
        logger.info("Testing %d-bin fine-grained discretization", n_bins)
        logger.info("=" * 80)

        builder = _make_finegrain_builder(n_bins)
        results = evaluate_representation(
            name=name,
            build_dataset_fn=builder,
            n_seeds=3,
            model_config={"n_output_classes": n_bins},
            run_baseline=True,
        )
        all_results[name] = results

    # Summary
    print("\n" + "=" * 80)
    print("FINE-GRAINED DISCRETIZATION SUMMARY")
    print("=" * 80)
    for name, r in all_results.items():
        c = r["challenger_avg"]
        print(f"  {name}: econ_dir={c['econ_dir']:.1%} trans={c['transition_acc']:.1%} "
              f"large={c['large_move_acc']:.1%} flat={c['pred_dist']['flat']:.1%} "
              f"sharpe={c['sharpe_costs']:+.3f} → {r['verdict']}")
