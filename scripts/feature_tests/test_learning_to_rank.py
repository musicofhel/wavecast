#!/usr/bin/env python3
"""Feature Test 20: Learning to Rank — Pairwise RankNet Loss.

Hypothesis: Replace classification loss with pairwise ranking loss.
Instead of "which class is correct?", optimize "rank the quantiles
correctly." Paper shows 3x Sharpe improvement in financial trading.

Usage:
    python -m scripts.feature_tests.test_learning_to_rank
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scripts.feature_tests.harness import (
    CONTEXT_LENGTH,
    MODEL_KWARGS,
    N_AUX_FEATURES,
    N_CLASSES,
    PERCENTILES,
    _build_d1_pipeline,
    _evaluate,
    _load_ohlcv,
    _ohlcv_to_timeseries,
    _split_ohlcv,
)

from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
)

logger = logging.getLogger(__name__)


def run_ranknet_test():
    """Run Learning-to-Rank experiment: CE vs RankNet loss."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Learning-to-Rank experiment: CE vs RankNet")

    # Load data
    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))
    logger.info("Loaded %d tickers", len(tickers))

    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)

    logger.info("Building D1 dataset...")
    X_train, tr_rets, tr_valid, tr_lvls, _, _ = _build_d1_pipeline(
        train_prices, None, tickers, None, 0,
    )
    X_test, te_rets, te_valid, te_lvls, _, _ = _build_d1_pipeline(
        test_prices, None, tickers, None, 0,
    )
    logger.info("Train: %d windows, Test: %d windows", len(X_train), len(X_test))

    boundaries = compute_quantile_boundaries(tr_rets, tr_lvls, tr_valid, PERCENTILES, per_level=True)
    y_train = assign_quantile_labels(tr_rets, tr_lvls, boundaries)
    y_test = assign_quantile_labels(te_rets, te_lvls, boundaries)

    # Class weights
    valid_labels = y_train[tr_valid]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    y_tr = y_train.astype(np.float64)

    n_seeds = 3
    baseline_results = []
    ranknet_results = []

    import torch
    for seed in range(n_seeds):
        logger.info("--- Seed %d/%d ---", seed + 1, n_seeds)
        np.random.seed(seed * 42 + 7)
        torch.manual_seed(seed * 42 + 7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 42 + 7)

        # --- Baseline: CE ---
        logger.info("  Training CE baseline...")
        model_base = WaveletGPT(
            vocab_size=1, context_length=CONTEXT_LENGTH,
            task="return_quantile", n_output_classes=N_CLASSES,
            class_weights=class_weights, input_mode="continuous",
            n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
        )
        metrics_base = model_base.fit(X_train, y_tr)
        pred_base = model_base.predict(X_test).astype(np.int64)
        base_eval = _evaluate(pred_base, te_rets, y_test, te_valid, metrics_base["train_loss"], 0.0)
        baseline_results.append(base_eval)
        logger.info("  Baseline: econ_dir=%.1f%% sharpe=%.3f",
                     base_eval.econ_dir_accuracy * 100, base_eval.sharpe_with_costs)

        # --- RankNet ---
        logger.info("  Training RankNet...")
        model_rank = WaveletGPT(
            vocab_size=1, context_length=CONTEXT_LENGTH,
            task="return_quantile", n_output_classes=N_CLASSES,
            class_weights=class_weights, input_mode="continuous",
            n_aux_features=N_AUX_FEATURES,
            loss_type="ranknet",
            loss_kwargs={"sigma": 1.0, "n_pairs": 256},
            **MODEL_KWARGS,
        )
        metrics_rank = model_rank.fit(X_train, y_tr)
        pred_rank = model_rank.predict(X_test).astype(np.int64)
        rank_eval = _evaluate(pred_rank, te_rets, y_test, te_valid, metrics_rank["train_loss"], 0.0)
        ranknet_results.append(rank_eval)
        logger.info("  RankNet: econ_dir=%.1f%% sharpe=%.3f",
                     rank_eval.econ_dir_accuracy * 100, rank_eval.sharpe_with_costs)

    # Summary
    base_econ = np.mean([r.econ_dir_accuracy for r in baseline_results])
    rank_econ = np.mean([r.econ_dir_accuracy for r in ranknet_results])
    base_sharpe = np.mean([r.sharpe_with_costs for r in baseline_results])
    rank_sharpe = np.mean([r.sharpe_with_costs for r in ranknet_results])
    base_trans = np.mean([r.transition_accuracy for r in baseline_results])
    rank_trans = np.mean([r.transition_accuracy for r in ranknet_results])

    d_econ = rank_econ - base_econ
    d_sharpe = rank_sharpe - base_sharpe
    d_trans = rank_trans - base_trans

    improved_sharpe = d_sharpe > 0.3
    degraded_econ = d_econ < -0.01
    degraded_trans = d_trans < -0.01

    if improved_sharpe and not degraded_econ:
        verdict = "PASS"
    elif degraded_econ or degraded_trans:
        verdict = "FAIL"
    else:
        verdict = "NO EFFECT"

    w = 100
    print()
    print("=" * w)
    print("LOSS TEST: Learning to Rank (RankNet)")
    print(f"sigma=1.0, n_pairs=256, Seeds: {n_seeds}")
    print("=" * w)
    print(f"\n{'Metric':<20} {'CE':>12} {'RankNet':>12} {'Delta':>10}")
    print("-" * w)
    print(f"{'Econ Dir Acc':<20} {base_econ:>11.1%} {rank_econ:>11.1%} {d_econ:>+9.1%}")
    print(f"{'Sharpe (+costs)':<20} {base_sharpe:>+11.3f} {rank_sharpe:>+11.3f} {d_sharpe:>+9.3f}")
    print(f"{'Transition Acc':<20} {base_trans:>11.1%} {rank_trans:>11.1%} {d_trans:>+9.1%}")
    print("-" * w)
    print(f"\nVERDICT: {verdict}")
    print("=" * w)

    # Save
    results_dir = Path.home() / ".wavecast" / "audit" / "feature_tests"
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "name": "learning_to_rank",
        "n_seeds": n_seeds,
        "verdict": verdict,
        "baseline_mean": {"econ_dir": base_econ, "sharpe_costs": base_sharpe, "transition": base_trans},
        "ranknet_mean": {"econ_dir": rank_econ, "sharpe_costs": rank_sharpe, "transition": rank_trans},
        "delta": {"econ_dir": d_econ, "sharpe_costs": d_sharpe, "transition": d_trans},
        "per_seed": {
            "baseline": [
                {"econ_dir": r.econ_dir_accuracy, "sharpe_costs": r.sharpe_with_costs, "transition": r.transition_accuracy}
                for r in baseline_results
            ],
            "ranknet": [
                {"econ_dir": r.econ_dir_accuracy, "sharpe_costs": r.sharpe_with_costs, "transition": r.transition_accuracy}
                for r in ranknet_results
            ],
        },
    }
    out_path = results_dir / "learning_to_rank_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)

    return output


if __name__ == "__main__":
    run_ranknet_test()
