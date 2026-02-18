#!/usr/bin/env python3
"""Experiment 12: Error Regularization.

Source: "The Art of Abstention" ACL 2021
Hypothesis: Penalize high confidence on wrong predictions with
L = CE + λ · |max_softmax - is_correct|. Forces model to be confident
only when correct, improving calibration for selective trading.

Pass criteria: ECE reduction > 30%. Selective econ_dir > 67% at coverage > 50%.

Usage:
    python -m scripts.feature_tests.test_error_reg
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


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = correct[mask].mean()
            ece += mask.sum() / len(labels) * abs(avg_acc - avg_conf)
    return float(ece)


def run_error_reg_test(n_seeds: int = 3) -> dict:
    """Compare CE vs error-regularized loss at λ ∈ {0.1, 0.5, 1.0}."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger.info("Error Regularization experiment, %d seeds", n_seeds)

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

    lambdas = [0.1, 0.5, 1.0]
    baseline_results = []
    ereg_results = {lv: [] for lv in lambdas}

    valid_mask = te_valid.astype(bool)

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
        base_probs = model_base.predict_proba(X_test)
        base_eval = _evaluate(
            pred_base, te_rets, y_test, te_valid,
            metrics["train_loss"], base_time,
        )
        base_ece = compute_ece(base_probs[valid_mask], y_test[valid_mask])

        # Base selective at 50% coverage
        base_conf = base_probs[valid_mask].max(axis=1)
        base_preds_v = pred_base[valid_mask]
        base_rets_v = te_rets[valid_mask]
        thr50 = np.quantile(base_conf, 0.5)
        keep = base_conf >= thr50
        mid = N_CLASSES // 2
        if keep.sum() > 0:
            pd = np.where(base_preds_v[keep] > mid, 1, np.where(base_preds_v[keep] < mid, -1, 0))
            ad = np.sign(base_rets_v[keep])
            dm = (pd != 0) & (ad != 0)
            base_sel = float((pd[dm] == ad[dm]).mean()) if dm.sum() > 0 else 0.0
        else:
            base_sel = 0.0

        baseline_results.append({
            "eval": base_eval,
            "ece": base_ece,
            "sel_50": base_sel,
        })
        logger.info(
            "  Baseline: econ=%.1f%% ECE=%.4f sel@50%%=%.1f%%",
            base_eval.econ_dir_accuracy * 100, base_ece, base_sel * 100,
        )

        for lam in lambdas:
            np.random.seed(seed * 42 + 7)
            torch.manual_seed(seed * 42 + 7)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed * 42 + 7)

            model_er = WaveletGPT(
                vocab_size=1, context_length=CONTEXT_LENGTH,
                task="return_quantile", n_output_classes=N_CLASSES,
                class_weights=class_weights, input_mode="continuous",
                n_aux_features=N_AUX_FEATURES,
                loss_type="error_reg", loss_kwargs={"lam": lam},
                **MODEL_KWARGS,
            )
            t0 = time.time()
            metrics = model_er.fit(X_train, y_tr)
            er_time = time.time() - t0
            pred_er = model_er.predict(X_test).astype(np.int64)
            er_probs = model_er.predict_proba(X_test)
            er_eval = _evaluate(
                pred_er, te_rets, y_test, te_valid,
                metrics["train_loss"], er_time,
            )
            er_ece = compute_ece(er_probs[valid_mask], y_test[valid_mask])

            # Selective at 50%
            er_conf = er_probs[valid_mask].max(axis=1)
            er_preds_v = pred_er[valid_mask]
            er_rets_v = te_rets[valid_mask]
            er_thr50 = np.quantile(er_conf, 0.5)
            keep_er = er_conf >= er_thr50
            if keep_er.sum() > 0:
                pd_er = np.where(er_preds_v[keep_er] > mid, 1, np.where(er_preds_v[keep_er] < mid, -1, 0))
                ad_er = np.sign(er_rets_v[keep_er])
                dm_er = (pd_er != 0) & (ad_er != 0)
                er_sel = float((pd_er[dm_er] == ad_er[dm_er]).mean()) if dm_er.sum() > 0 else 0.0
            else:
                er_sel = 0.0

            ereg_results[lam].append({
                "eval": er_eval,
                "ece": er_ece,
                "sel_50": er_sel,
            })
            logger.info(
                "  λ=%.1f: econ=%.1f%% ECE=%.4f sel@50%%=%.1f%%",
                lam, er_eval.econ_dir_accuracy * 100, er_ece, er_sel * 100,
            )

    # Aggregate
    avg_base_ece = float(np.mean([r["ece"] for r in baseline_results]))
    avg_base_econ = float(np.mean([r["eval"].econ_dir_accuracy for r in baseline_results]))
    avg_base_sharpe = float(np.mean([r["eval"].sharpe_with_costs for r in baseline_results]))
    avg_base_sel = float(np.mean([r["sel_50"] for r in baseline_results]))

    lam_summaries = {}
    best_lam = None
    best_ece_reduction = 0.0
    best_sel = 0.0

    for lam in lambdas:
        avg_ece = float(np.mean([r["ece"] for r in ereg_results[lam]]))
        avg_econ = float(np.mean([r["eval"].econ_dir_accuracy for r in ereg_results[lam]]))
        avg_sharpe = float(np.mean([r["eval"].sharpe_with_costs for r in ereg_results[lam]]))
        avg_sel = float(np.mean([r["sel_50"] for r in ereg_results[lam]]))
        ece_red = (avg_base_ece - avg_ece) / (avg_base_ece + 1e-10)

        lam_summaries[str(lam)] = {
            "econ_dir": avg_econ,
            "sharpe_costs": avg_sharpe,
            "ece": avg_ece,
            "ece_reduction": ece_red,
            "sel_50": avg_sel,
        }
        if ece_red > best_ece_reduction:
            best_ece_reduction = ece_red
            best_lam = lam
            best_sel = avg_sel

    verdict = "PASS" if best_ece_reduction > 0.30 and best_sel > 0.67 else "FAIL"

    w = 100
    print()
    print("=" * w)
    print("EXPERIMENT: Error Regularization")
    print("=" * w)
    print(f"\n{'λ':<6} {'Econ Dir':>10} {'Sharpe':>10} {'ECE':>10} {'ECE Δ%':>10} {'Sel@50%':>10}")
    print("-" * w)
    print(f"{'base':<6} {avg_base_econ:>9.1%} {avg_base_sharpe:>+9.3f} {avg_base_ece:>9.4f} {'':>10} {avg_base_sel:>9.1%}")
    for lam in lambdas:
        s = lam_summaries[str(lam)]
        print(
            f"{lam:<6.1f} {s['econ_dir']:>9.1%} {s['sharpe_costs']:>+9.3f} "
            f"{s['ece']:>9.4f} {s['ece_reduction']:>+9.0%} {s['sel_50']:>9.1%}"
        )
    print("-" * w)
    print(f"\nVERDICT: {verdict}")
    print("=" * w)

    results_dir = Path.home() / ".wavecast" / "audit" / "feature_tests"
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "name": "error_regularization",
        "experiment_type": "selective_prediction",
        "n_seeds": n_seeds,
        "verdict": verdict,
        "baseline": {
            "econ_dir": avg_base_econ,
            "sharpe_costs": avg_base_sharpe,
            "ece": avg_base_ece,
            "sel_50": avg_base_sel,
        },
        "lambda_summaries": lam_summaries,
        "best_lambda": best_lam,
        "best_ece_reduction": best_ece_reduction,
        "best_selective": best_sel,
    }
    out_path = results_dir / "error_reg_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)
    return output


if __name__ == "__main__":
    run_error_reg_test()
