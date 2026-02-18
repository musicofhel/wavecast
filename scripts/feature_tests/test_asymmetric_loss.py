#!/usr/bin/env python3
"""Experiment 4: Asymmetric (Cost-Weighted) Loss.

Source: LINEX loss (Khatun & Matin 2020); cost-sensitive ensembles (Lboro)
Hypothesis: Standard CE treats false buys and missed buys equally. In trading,
false positives (wrong trades) are costlier than false negatives (missed opps).
Cost-weighted CE with explicit cost matrix penalizes directional errors more.

Pass criteria: Sharpe +0.3 (primary), econ dir no degradation.

Usage:
    python -m scripts.feature_tests.test_asymmetric_loss
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
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


def run_asymmetric_loss_test(n_seeds: int = 3) -> dict:
    """Compare CE vs asymmetric loss with FP/FN ratio of 5:1."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger.info("Asymmetric Loss experiment, %d seeds", n_seeds)

    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))

    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)

    X_train, tr_rets, tr_valid, tr_lvls, _, _ = _build_d1_pipeline(
        train_prices, None, tickers, None, 0
    )
    X_test, te_rets, te_valid, te_lvls, _, _ = _build_d1_pipeline(
        test_prices, None, tickers, None, 0
    )

    boundaries = compute_quantile_boundaries(
        tr_rets, tr_lvls, tr_valid, PERCENTILES, per_level=True
    )
    y_train = assign_quantile_labels(tr_rets, tr_lvls, boundaries)
    y_test = assign_quantile_labels(te_rets, te_lvls, boundaries)

    valid_labels = y_train[tr_valid]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    y_tr = y_train.astype(np.float64)

    baseline_results = []
    asym_results = []

    for seed in range(n_seeds):
        logger.info("--- Seed %d/%d ---", seed + 1, n_seeds)

        # Baseline
        np.random.seed(seed * 42 + 7)
        torch.manual_seed(seed * 42 + 7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 42 + 7)

        model_base = WaveletGPT(
            vocab_size=1, context_length=CONTEXT_LENGTH,
            task="return_quantile", n_output_classes=N_CLASSES,
            class_weights=class_weights, input_mode="continuous",
            n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
        )
        t0 = time.time()
        metrics = model_base.fit(X_train, y_tr)
        base_time = time.time() - t0
        pred_base = model_base.predict(X_test).astype(np.int64)
        base_eval = _evaluate(pred_base, te_rets, y_test, te_valid, metrics["train_loss"], base_time)
        baseline_results.append(base_eval)
        logger.info("  Baseline: econ=%.1f%% sharpe=%.3f", base_eval.econ_dir_accuracy * 100, base_eval.sharpe_with_costs)

        # Asymmetric
        np.random.seed(seed * 42 + 7)
        torch.manual_seed(seed * 42 + 7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 42 + 7)

        model_asym = WaveletGPT(
            vocab_size=1, context_length=CONTEXT_LENGTH,
            task="return_quantile", n_output_classes=N_CLASSES,
            class_weights=class_weights, input_mode="continuous",
            n_aux_features=N_AUX_FEATURES,
            loss_type="asymmetric", loss_kwargs={"fp_fn_ratio": 5.0},
            **MODEL_KWARGS,
        )
        t0 = time.time()
        metrics = model_asym.fit(X_train, y_tr)
        asym_time = time.time() - t0
        pred_asym = model_asym.predict(X_test).astype(np.int64)
        asym_eval = _evaluate(pred_asym, te_rets, y_test, te_valid, metrics["train_loss"], asym_time)
        asym_results.append(asym_eval)
        logger.info("  Asymmetric: econ=%.1f%% sharpe=%.3f", asym_eval.econ_dir_accuracy * 100, asym_eval.sharpe_with_costs)

    base_econ = np.mean([r.econ_dir_accuracy for r in baseline_results])
    asym_econ = np.mean([r.econ_dir_accuracy for r in asym_results])
    base_sharpe = np.mean([r.sharpe_with_costs for r in baseline_results])
    asym_sharpe = np.mean([r.sharpe_with_costs for r in asym_results])
    base_trans = np.mean([r.transition_accuracy for r in baseline_results])
    asym_trans = np.mean([r.transition_accuracy for r in asym_results])

    d_econ = asym_econ - base_econ
    d_sharpe = asym_sharpe - base_sharpe
    d_trans = asym_trans - base_trans

    verdict = "PASS" if d_sharpe > 0.3 and d_econ > -0.01 else "FAIL"

    w = 100
    print()
    print("=" * w)
    print("EXPERIMENT: Asymmetric Loss (FP/FN ratio=5:1)")
    print("=" * w)
    print(f"\n{'Metric':<20} {'Baseline':>10} {'Asymmetric':>12} {'Delta':>10}")
    print("-" * w)
    print(f"{'Econ Dir Acc':<20} {base_econ:>9.1%} {asym_econ:>11.1%} {d_econ:>+9.1%}")
    print(f"{'Sharpe (+costs)':<20} {base_sharpe:>+9.3f} {asym_sharpe:>+11.3f} {d_sharpe:>+9.3f}")
    print(f"{'Transition Acc':<20} {base_trans:>9.1%} {asym_trans:>11.1%} {d_trans:>+9.1%}")
    print("-" * w)
    print(f"\nVERDICT: {verdict}")
    print("=" * w)

    results_dir = Path.home() / ".wavecast" / "audit" / "feature_tests"
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "name": "asymmetric_loss",
        "experiment_type": "loss_function",
        "fp_fn_ratio": 5.0,
        "n_seeds": n_seeds,
        "verdict": verdict,
        "baseline_mean": {"econ_dir": float(base_econ), "sharpe_costs": float(base_sharpe), "transition": float(base_trans)},
        "challenger_mean": {"econ_dir": float(asym_econ), "sharpe_costs": float(asym_sharpe), "transition": float(asym_trans)},
        "delta": {"econ_dir": float(d_econ), "sharpe_costs": float(d_sharpe), "transition": float(d_trans)},
        "per_seed": {
            "baseline": [{"econ_dir": r.econ_dir_accuracy, "sharpe_costs": r.sharpe_with_costs, "transition": r.transition_accuracy, "train_loss": r.train_loss} for r in baseline_results],
            "challenger": [{"econ_dir": r.econ_dir_accuracy, "sharpe_costs": r.sharpe_with_costs, "transition": r.transition_accuracy, "train_loss": r.train_loss} for r in asym_results],
        },
    }
    out_path = results_dir / "asymmetric_loss_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)
    return output


if __name__ == "__main__":
    run_asymmetric_loss_test()
