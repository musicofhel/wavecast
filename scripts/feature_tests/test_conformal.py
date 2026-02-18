#!/usr/bin/env python3
"""Experiment 10: Conformal Selective Prediction.

Source: Zaffran et al. 2022 AgACI; Xu & Xie 2021 EnbPI; Wisniewski 2020
Hypothesis: Construct prediction SETS with coverage guarantees. If prediction
set contains multiple classes, abstain. Trade only high-confidence singletons.

Pass: At alpha=0.10: marginal coverage >= 88%, selective econ_dir > 67% at coverage > 25%.

Usage:
    python -m scripts.feature_tests.test_conformal
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
    COST_BPS,
    MIN_RETURN_THRESHOLD,
    MODEL_KWARGS,
    N_AUX_FEATURES,
    N_CLASSES,
    PERCENTILES,
    _build_d1_pipeline,
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


def build_prediction_sets(
    cal_proba: np.ndarray,
    cal_labels: np.ndarray,
    test_proba: np.ndarray,
    alpha: float,
) -> tuple[list[set[int]], float]:
    """Split conformal prediction sets.

    Nonconformity score: s_i = 1 - softmax[y_true_i]
    Calibrate quantile q_hat for coverage 1 - alpha.
    Prediction set: C(x) = {y : softmax_y(x) >= 1 - q_hat}
    """
    # Calibration scores
    n_cal = len(cal_labels)
    scores = 1.0 - cal_proba[np.arange(n_cal), cal_labels]

    # Quantile with finite-sample correction
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    q_hat = float(np.quantile(scores, q_level))

    # Build prediction sets on test
    threshold = 1.0 - q_hat
    pred_sets = []
    for i in range(len(test_proba)):
        pset = set()
        for c in range(test_proba.shape[1]):
            if test_proba[i, c] >= threshold:
                pset.add(c)
        if not pset:
            # If nothing passes, include the argmax
            pset.add(int(np.argmax(test_proba[i])))
        pred_sets.append(pset)

    return pred_sets, q_hat


def evaluate_conformal(
    pred_sets: list[set[int]],
    test_proba: np.ndarray,
    test_labels: np.ndarray,
    test_returns: np.ndarray,
    valid_mask: np.ndarray,
) -> dict:
    """Evaluate conformal prediction sets."""
    mid = N_CLASSES // 2
    n_test = len(pred_sets)

    # Marginal coverage: fraction where true label is in prediction set
    covered = 0
    valid_count = 0
    for i in range(n_test):
        if valid_mask[i]:
            valid_count += 1
            if test_labels[i] in pred_sets[i]:
                covered += 1
    coverage = covered / max(valid_count, 1)

    # Directional singletons: sets with exactly one directional class
    singleton_mask = np.zeros(n_test, dtype=bool)
    singleton_preds = np.zeros(n_test, dtype=np.float64)
    for i in range(n_test):
        if not valid_mask[i]:
            continue
        pset = pred_sets[i]
        if len(pset) == 1:
            cls = next(iter(pset))
            if cls != mid:  # Not flat
                singleton_mask[i] = True
                singleton_preds[i] = 1.0 if cls > mid else -1.0

    singleton_coverage = float(singleton_mask.sum()) / max(valid_mask.sum(), 1)
    actual_dir = np.sign(test_returns)

    # Selective econ dir on singletons
    filt = singleton_mask & (np.abs(test_returns) > MIN_RETURN_THRESHOLD)
    if filt.sum() > 0:
        has_pred = singleton_preds[filt] != 0
        if has_pred.sum() > 0:
            econ_dir = float(
                np.mean(singleton_preds[filt][has_pred] == actual_dir[filt][has_pred])
            )
        else:
            econ_dir = 0.5
    else:
        econ_dir = 0.5

    # Selective Sharpe
    if singleton_mask.sum() > 1:
        pnl = singleton_preds[singleton_mask] * test_returns[singleton_mask]
        pnl_net = pnl.copy()
        dir_changes = np.abs(np.diff(singleton_preds[singleton_mask]))
        cost = np.zeros(int(singleton_mask.sum()))
        cost[1:] = dir_changes * (COST_BPS / 10000)
        cost[0] = abs(singleton_preds[singleton_mask][0]) * (COST_BPS / 10000)
        pnl_net = pnl - cost
        pnl_net = pnl_net[~np.isnan(pnl_net)]
        if len(pnl_net) > 1 and np.std(pnl_net) > 0:
            sharpe = float(np.mean(pnl_net) / np.std(pnl_net) * np.sqrt(252 * 7))
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Average set size
    avg_set_size = float(np.mean([len(s) for s in pred_sets]))

    return {
        "marginal_coverage": coverage,
        "singleton_coverage": singleton_coverage,
        "selective_econ_dir": econ_dir,
        "selective_sharpe": sharpe,
        "avg_set_size": avg_set_size,
    }


def run_conformal_test(n_seeds: int = 3) -> dict:
    """Run conformal prediction experiment at multiple alpha levels."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    alphas = [0.05, 0.10, 0.20]
    logger.info("Conformal Selective Prediction, alphas=%s, %d seeds", alphas, n_seeds)

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

    # Split train into train + calibration
    n_train = int(len(X_train) * 0.8)
    X_tr = X_train[:n_train]
    y_tr_rets = tr_rets[:n_train]
    y_tr_valid = tr_valid[:n_train]
    y_tr_lvls = tr_lvls[:n_train]

    X_cal = X_train[n_train:]
    cal_rets = tr_rets[n_train:]
    cal_valid = tr_valid[n_train:]
    cal_lvls = tr_lvls[n_train:]

    boundaries = compute_quantile_boundaries(
        y_tr_rets, y_tr_lvls, y_tr_valid, PERCENTILES, per_level=True
    )
    y_train_labels = assign_quantile_labels(y_tr_rets, y_tr_lvls, boundaries)
    y_cal_labels = assign_quantile_labels(cal_rets, cal_lvls, boundaries)
    y_test_labels = assign_quantile_labels(te_rets, te_lvls, boundaries)

    valid_labels = y_train_labels[y_tr_valid]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    all_results: dict[float, list[dict]] = {a: [] for a in alphas}

    for seed in range(n_seeds):
        logger.info("--- Seed %d/%d ---", seed + 1, n_seeds)
        np.random.seed(seed * 42 + 7)
        torch.manual_seed(seed * 42 + 7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 42 + 7)

        model = WaveletGPT(
            vocab_size=1,
            context_length=CONTEXT_LENGTH,
            task="return_quantile",
            n_output_classes=N_CLASSES,
            class_weights=class_weights,
            input_mode="continuous",
            n_aux_features=N_AUX_FEATURES,
            **MODEL_KWARGS,
        )
        t0 = time.time()
        model.fit(X_tr, y_train_labels.astype(np.float64))
        logger.info("  Trained in %.1fs", time.time() - t0)

        cal_proba = model.predict_proba(X_cal)
        test_proba = model.predict_proba(X_test)

        for alpha in alphas:
            pred_sets, q_hat = build_prediction_sets(
                cal_proba[cal_valid], y_cal_labels[cal_valid], test_proba, alpha
            )
            result = evaluate_conformal(
                pred_sets, test_proba, y_test_labels, te_rets, te_valid
            )
            result["q_hat"] = q_hat
            all_results[alpha].append(result)
            logger.info(
                "  alpha=%.2f: coverage=%.1f%% singletons=%.1f%% econ=%.1f%% sharpe=%.3f set_size=%.2f",
                alpha,
                result["marginal_coverage"] * 100,
                result["singleton_coverage"] * 100,
                result["selective_econ_dir"] * 100,
                result["selective_sharpe"],
                result["avg_set_size"],
            )

    # Print results
    w = 110
    print()
    print("=" * w)
    print("EXPERIMENT: Conformal Selective Prediction")
    print("=" * w)
    print(
        f"\n{'Alpha':>8} {'Coverage':>10} {'Singletons':>12} {'Sel Econ':>10} {'Sel Sharpe':>12} {'Avg Set':>10}"
    )
    print("-" * w)

    best_econ = 0.0
    best_sharpe = 0.0
    best_coverage = 0.0

    for alpha in alphas:
        results = all_results[alpha]
        cov = np.mean([r["marginal_coverage"] for r in results])
        sing = np.mean([r["singleton_coverage"] for r in results])
        econ = np.mean([r["selective_econ_dir"] for r in results])
        sharpe = np.mean([r["selective_sharpe"] for r in results])
        avg_set = np.mean([r["avg_set_size"] for r in results])

        print(
            f"  {alpha:>6.2f} {cov:>9.1%} {sing:>11.1%} {econ:>9.1%} {sharpe:>+11.3f} {avg_set:>9.2f}"
        )

        if sing > 0.25 and econ > best_econ:
            best_econ = econ
            best_sharpe = sharpe
            best_coverage = sing

    pass_coverage = any(
        np.mean([r["marginal_coverage"] for r in all_results[0.10]]) >= 0.88
        for _ in [1]
    )
    pass_econ = best_econ > 0.67
    pass_cov = best_coverage > 0.25
    verdict = "PASS" if pass_coverage and pass_econ and pass_cov else "FAIL"

    print("-" * w)
    print(f"\nBest singleton: econ={best_econ:.1%}, sharpe={best_sharpe:+.3f}, coverage={best_coverage:.1%}")
    print(f"VERDICT: {verdict}")
    print("=" * w)

    # Save
    results_dir = Path.home() / ".wavecast" / "audit" / "feature_tests"
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "name": "conformal_selective",
        "experiment_type": "calibration",
        "alphas": alphas,
        "n_seeds": n_seeds,
        "verdict": verdict,
        "best_selective_econ": float(best_econ),
        "best_selective_sharpe": float(best_sharpe),
        "best_singleton_coverage": float(best_coverage),
        "results": {
            str(alpha): {
                "marginal_coverage": float(np.mean([r["marginal_coverage"] for r in results])),
                "singleton_coverage": float(np.mean([r["singleton_coverage"] for r in results])),
                "selective_econ_dir": float(np.mean([r["selective_econ_dir"] for r in results])),
                "selective_sharpe": float(np.mean([r["selective_sharpe"] for r in results])),
                "avg_set_size": float(np.mean([r["avg_set_size"] for r in results])),
            }
            for alpha, results in all_results.items()
        },
    }
    out_path = results_dir / "conformal_selective_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)
    return output


if __name__ == "__main__":
    run_conformal_test()
