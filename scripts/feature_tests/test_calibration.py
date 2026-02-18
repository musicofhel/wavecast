#!/usr/bin/env python3
"""Experiment 8: Post-Hoc Calibration Suite.

Source: Kuleshov et al. ICML 2018; smooth isotonic (PMC); ROC-regularized isotonic
Hypothesis: Softmax confidence is useless for trading. Post-hoc calibration
methods can fix confidence estimates without retraining.

Compares: Temperature scaling, Platt scaling (sigmoid), Isotonic regression
Sweeps abstention thresholds: 0%, 10%, 20%, 30%, 50%

Pass: At any threshold with coverage > 30%: selective econ_dir > 67% AND selective Sharpe > +5.0

Usage:
    python -m scripts.feature_tests.test_calibration
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax as scipy_softmax
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
from sklearn.isotonic import IsotonicRegression

from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
)

logger = logging.getLogger(__name__)


def temperature_scale(logits: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to logits."""
    return scipy_softmax(logits / T, axis=1)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit temperature parameter by minimizing NLL on validation set."""
    def nll(T: float) -> float:
        proba = temperature_scale(logits, T)
        log_proba = np.log(np.clip(proba[np.arange(len(labels)), labels], 1e-10, 1.0))
        return -np.mean(log_proba)

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def fit_platt(confidences: np.ndarray, correct: np.ndarray) -> tuple[float, float]:
    """Fit Platt scaling: P(correct | confidence) = sigmoid(a * confidence + b)."""
    from scipy.optimize import minimize

    def loss(params: np.ndarray) -> float:
        a, b = params
        p = 1.0 / (1.0 + np.exp(-(a * confidences + b)))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.mean(correct * np.log(p) + (1 - correct) * np.log(1 - p))

    result = minimize(loss, x0=[1.0, 0.0], method="Nelder-Mead")
    return float(result.x[0]), float(result.x[1])


def selective_evaluate(
    pred_labels: np.ndarray,
    actual_returns: np.ndarray,
    actual_labels: np.ndarray,
    valid_mask: np.ndarray,
    calibrated_conf: np.ndarray,
    threshold: float,
) -> dict:
    """Evaluate metrics on subset where calibrated confidence > threshold."""
    mid = N_CLASSES // 2
    pred_dir = np.zeros(len(pred_labels), dtype=np.float64)
    pred_dir[pred_labels > mid] = 1.0
    pred_dir[pred_labels < mid] = -1.0
    actual_dir = np.sign(actual_returns)

    # Select predictions above threshold
    select = valid_mask & (calibrated_conf >= threshold)
    coverage = float(select.sum()) / max(valid_mask.sum(), 1)

    if select.sum() == 0:
        return {"coverage": 0.0, "econ_dir": 0.5, "sharpe": 0.0}

    filt = select & (np.abs(actual_returns) > MIN_RETURN_THRESHOLD)
    if filt.sum() > 0:
        has_pred = pred_dir[filt] != 0
        if has_pred.sum() > 0:
            econ_dir = float(
                np.mean(pred_dir[filt][has_pred] == actual_dir[filt][has_pred])
            )
        else:
            econ_dir = 0.5
    else:
        econ_dir = 0.5

    # Sharpe with costs
    pnl = pred_dir[select] * actual_returns[select]
    dir_changes = np.abs(np.diff(pred_dir[select]))
    cost = np.zeros(int(select.sum()))
    cost[1:] = dir_changes * (COST_BPS / 10000)
    cost[0] = abs(pred_dir[select][0]) * (COST_BPS / 10000)
    pnl_net = pnl - cost
    pnl_net = pnl_net[~np.isnan(pnl_net)]
    if len(pnl_net) > 1 and np.std(pnl_net) > 0:
        sharpe = float(np.mean(pnl_net) / np.std(pnl_net) * np.sqrt(252 * 7))
    else:
        sharpe = 0.0

    return {"coverage": coverage, "econ_dir": econ_dir, "sharpe": sharpe}


def run_calibration_test(n_seeds: int = 3) -> dict:
    """Compare temperature, Platt, and isotonic calibration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger.info("Post-hoc Calibration experiment, %d seeds", n_seeds)

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

    # Split train into train + val for calibration fitting
    n_train = int(len(X_train) * 0.8)
    X_tr = X_train[:n_train]
    y_tr_rets = tr_rets[:n_train]
    y_tr_valid = tr_valid[:n_train]
    y_tr_lvls = tr_lvls[:n_train]

    X_val = X_train[n_train:]
    val_rets = tr_rets[n_train:]
    val_valid = tr_valid[n_train:]
    val_lvls = tr_lvls[n_train:]

    boundaries = compute_quantile_boundaries(
        y_tr_rets, y_tr_lvls, y_tr_valid, PERCENTILES, per_level=True
    )
    y_train_labels = assign_quantile_labels(y_tr_rets, y_tr_lvls, boundaries)
    y_val_labels = assign_quantile_labels(val_rets, val_lvls, boundaries)
    y_test_labels = assign_quantile_labels(te_rets, te_lvls, boundaries)

    valid_labels = y_train_labels[y_tr_valid]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]
    methods = ["uncalibrated", "temperature", "platt", "isotonic"]
    all_results: dict[str, dict[float, list[dict]]] = {
        m: {t: [] for t in thresholds} for m in methods
    }

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

        # Get raw logits on val and test
        val_proba = model.predict_proba(X_val)
        test_proba = model.predict_proba(X_test)
        val_preds = np.argmax(val_proba, axis=1)
        test_preds = np.argmax(test_proba, axis=1)

        # Uncalibrated confidence
        uncal_conf_test = np.max(test_proba, axis=1)

        # Temperature scaling
        # We need raw logits, but predict_proba returns softmax.
        # Approximate: logits = log(proba + eps)
        val_logits = np.log(np.clip(val_proba, 1e-10, 1.0))
        test_logits = np.log(np.clip(test_proba, 1e-10, 1.0))

        T = fit_temperature(val_logits, y_val_labels[val_valid])
        temp_proba = temperature_scale(test_logits, T)
        temp_conf = np.max(temp_proba, axis=1)
        logger.info("  Temperature: T=%.3f", T)

        # Platt scaling
        val_correct = (val_preds == y_val_labels).astype(np.float64)
        val_max_conf = np.max(val_proba, axis=1)
        a, b = fit_platt(val_max_conf[val_valid], val_correct[val_valid])
        platt_conf = 1.0 / (1.0 + np.exp(-(a * np.max(test_proba, axis=1) + b)))
        logger.info("  Platt: a=%.3f, b=%.3f", a, b)

        # Isotonic regression
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_max_conf[val_valid], val_correct[val_valid])
        iso_conf = iso.predict(np.max(test_proba, axis=1))

        # Evaluate at each threshold
        conf_map = {
            "uncalibrated": uncal_conf_test,
            "temperature": temp_conf,
            "platt": platt_conf,
            "isotonic": iso_conf,
        }

        for method, conf in conf_map.items():
            for thresh in thresholds:
                result = selective_evaluate(
                    test_preds, te_rets, y_test_labels, te_valid, conf, thresh
                )
                all_results[method][thresh].append(result)

    # Print results
    w = 110
    print()
    print("=" * w)
    print("EXPERIMENT: Post-Hoc Calibration Suite")
    print("=" * w)

    best_selective_econ = 0.0
    best_selective_sharpe = 0.0
    best_config = ""

    for method in methods:
        print(f"\n  {method.upper()}")
        print(f"  {'Threshold':>10} {'Coverage':>10} {'Econ Dir':>10} {'Sharpe':>10}")
        print("  " + "-" * 50)
        for thresh in thresholds:
            results = all_results[method][thresh]
            avg_cov = np.mean([r["coverage"] for r in results])
            avg_econ = np.mean([r["econ_dir"] for r in results])
            avg_sharpe = np.mean([r["sharpe"] for r in results])
            print(
                f"  {thresh:>10.0%} {avg_cov:>9.1%} {avg_econ:>9.1%} {avg_sharpe:>+9.3f}"
            )
            if avg_cov > 0.30 and avg_econ > best_selective_econ:
                best_selective_econ = avg_econ
                best_selective_sharpe = avg_sharpe
                best_config = f"{method}@{thresh:.0%}"

    pass_econ = best_selective_econ > 0.67
    pass_sharpe = best_selective_sharpe > 5.0
    verdict = "PASS" if pass_econ and pass_sharpe else "FAIL"

    print("-" * w)
    print(f"\nBest (coverage>30%): {best_config} — econ={best_selective_econ:.1%}, sharpe={best_selective_sharpe:+.3f}")
    print(f"VERDICT: {verdict}")
    print("=" * w)

    # Save
    results_dir = Path.home() / ".wavecast" / "audit" / "feature_tests"
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "name": "post_hoc_calibration",
        "experiment_type": "calibration",
        "n_seeds": n_seeds,
        "verdict": verdict,
        "best_config": best_config,
        "best_selective_econ": float(best_selective_econ),
        "best_selective_sharpe": float(best_selective_sharpe),
        "results": {
            method: {
                str(thresh): {
                    "coverage": float(np.mean([r["coverage"] for r in results])),
                    "econ_dir": float(np.mean([r["econ_dir"] for r in results])),
                    "sharpe": float(np.mean([r["sharpe"] for r in results])),
                }
                for thresh, results in method_results.items()
            }
            for method, method_results in all_results.items()
        },
    }
    out_path = results_dir / "post_hoc_calibration_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)
    return output


if __name__ == "__main__":
    run_calibration_test()
