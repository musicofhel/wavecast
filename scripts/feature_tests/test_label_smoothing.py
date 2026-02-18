#!/usr/bin/env python3
"""Experiment 3: Label Smoothing.

Source: Calibration survey (2308.01222)
Hypothesis: Hard 0/1 targets cause overconfident softmax. Label smoothing
replaces targets with (1-eps)*one_hot + eps/K, regularizing the model to
be less sure. Reduces overconfidence -> better calibration.

Pass criteria: ECE reduction >50%, econ dir no degradation, transition > -1pp

Usage:
    python -m scripts.feature_tests.test_label_smoothing
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


def compute_ece(proba: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error."""
    confidences = np.max(proba, axis=1)
    predictions = np.argmax(proba, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)
    return float(ece)


def run_label_smoothing_test(n_seeds: int = 3) -> dict:
    """Compare baseline CE vs label smoothing (eps=0.05, 0.1, 0.2)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    epsilons = [0.05, 0.1, 0.2]
    logger.info(
        "Label Smoothing experiment: CE vs eps=%s, %d seeds", epsilons, n_seeds
    )

    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))
    logger.info("Loaded %d tickers", len(tickers))

    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)

    X_train, tr_rets, tr_valid, tr_lvls, _, _ = _build_d1_pipeline(
        train_prices, None, tickers, None, 0
    )
    X_test, te_rets, te_valid, te_lvls, _, _ = _build_d1_pipeline(
        test_prices, None, tickers, None, 0
    )
    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

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

    all_results: dict[str, list] = {"baseline": [], **{f"eps_{e}": [] for e in epsilons}}
    all_eces: dict[str, list] = {"baseline": [], **{f"eps_{e}": [] for e in epsilons}}

    for seed in range(n_seeds):
        logger.info("--- Seed %d/%d ---", seed + 1, n_seeds)

        # Baseline
        np.random.seed(seed * 42 + 7)
        torch.manual_seed(seed * 42 + 7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 42 + 7)

        model_base = WaveletGPT(
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
        metrics = model_base.fit(X_train, y_tr)
        base_time = time.time() - t0
        pred_base = model_base.predict(X_test).astype(np.int64)
        proba_base = model_base.predict_proba(X_test)
        base_eval = _evaluate(
            pred_base, te_rets, y_test, te_valid, metrics["train_loss"], base_time
        )
        base_ece = compute_ece(proba_base[te_valid], y_test[te_valid])
        all_results["baseline"].append(base_eval)
        all_eces["baseline"].append(base_ece)
        logger.info(
            "  Baseline: econ=%.1f%% sharpe=%.3f ECE=%.4f",
            base_eval.econ_dir_accuracy * 100,
            base_eval.sharpe_with_costs,
            base_ece,
        )

        # Sweep epsilons
        for eps in epsilons:
            np.random.seed(seed * 42 + 7)
            torch.manual_seed(seed * 42 + 7)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed * 42 + 7)

            model_ls = WaveletGPT(
                vocab_size=1,
                context_length=CONTEXT_LENGTH,
                task="return_quantile",
                n_output_classes=N_CLASSES,
                class_weights=class_weights,
                input_mode="continuous",
                n_aux_features=N_AUX_FEATURES,
                loss_type="label_smoothing",
                loss_kwargs={"epsilon": eps},
                **MODEL_KWARGS,
            )
            t0 = time.time()
            metrics = model_ls.fit(X_train, y_tr)
            ls_time = time.time() - t0
            pred_ls = model_ls.predict(X_test).astype(np.int64)
            proba_ls = model_ls.predict_proba(X_test)
            ls_eval = _evaluate(
                pred_ls, te_rets, y_test, te_valid, metrics["train_loss"], ls_time
            )
            ls_ece = compute_ece(proba_ls[te_valid], y_test[te_valid])
            key = f"eps_{eps}"
            all_results[key].append(ls_eval)
            all_eces[key].append(ls_ece)
            logger.info(
                "  eps=%.2f: econ=%.1f%% sharpe=%.3f ECE=%.4f",
                eps,
                ls_eval.econ_dir_accuracy * 100,
                ls_eval.sharpe_with_costs,
                ls_ece,
            )

    # Aggregate
    w = 100
    print()
    print("=" * w)
    print("EXPERIMENT: Label Smoothing")
    print("=" * w)
    print(
        f"\n{'Config':<15} {'Econ Dir':>10} {'Sharpe':>10} {'Trans':>10} {'ECE':>10} {'ECE Red':>10}"
    )
    print("-" * w)

    base_econ = np.mean([r.econ_dir_accuracy for r in all_results["baseline"]])
    base_sharpe = np.mean([r.sharpe_with_costs for r in all_results["baseline"]])
    base_trans = np.mean([r.transition_accuracy for r in all_results["baseline"]])
    base_ece = np.mean(all_eces["baseline"])

    print(
        f"{'Baseline':<15} {base_econ:>9.1%} {base_sharpe:>+9.3f} {base_trans:>9.1%} {base_ece:>9.4f} {'':>10}"
    )

    best_key = None
    best_ece_reduction = 0.0

    for eps in epsilons:
        key = f"eps_{eps}"
        econ = np.mean([r.econ_dir_accuracy for r in all_results[key]])
        sharpe = np.mean([r.sharpe_with_costs for r in all_results[key]])
        trans = np.mean([r.transition_accuracy for r in all_results[key]])
        ece = np.mean(all_eces[key])
        ece_red = (base_ece - ece) / base_ece if base_ece > 0 else 0.0

        print(
            f"{'eps=' + str(eps):<15} {econ:>9.1%} {sharpe:>+9.3f} {trans:>9.1%} {ece:>9.4f} {ece_red:>+9.1%}"
        )

        if ece_red > best_ece_reduction:
            best_ece_reduction = ece_red
            best_key = key

    # Verdict on best epsilon
    if best_key:
        best_econ = np.mean([r.econ_dir_accuracy for r in all_results[best_key]])
        best_trans = np.mean([r.transition_accuracy for r in all_results[best_key]])
        d_econ = best_econ - base_econ
        d_trans = best_trans - base_trans
        pass_ece = best_ece_reduction > 0.5
        pass_econ = d_econ > -0.01
        pass_trans = d_trans > -0.01
        verdict = "PASS" if pass_ece and pass_econ and pass_trans else "FAIL"
    else:
        verdict = "FAIL"
        d_econ = 0.0
        d_trans = 0.0

    print("-" * w)
    print(f"\nBest: {best_key} (ECE reduction {best_ece_reduction:.1%})")
    print(f"VERDICT: {verdict}")
    print("=" * w)

    # Save
    results_dir = Path.home() / ".wavecast" / "audit" / "feature_tests"
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "name": "label_smoothing",
        "experiment_type": "loss_function",
        "epsilons_tested": epsilons,
        "n_seeds": n_seeds,
        "verdict": verdict,
        "baseline_mean": {
            "econ_dir": float(base_econ),
            "sharpe_costs": float(base_sharpe),
            "transition": float(base_trans),
            "ece": float(base_ece),
        },
        "best_epsilon": best_key,
        "best_ece_reduction": float(best_ece_reduction),
        "per_epsilon": {},
    }

    for eps in epsilons:
        key = f"eps_{eps}"
        output["per_epsilon"][key] = {
            "econ_dir": float(np.mean([r.econ_dir_accuracy for r in all_results[key]])),
            "sharpe_costs": float(np.mean([r.sharpe_with_costs for r in all_results[key]])),
            "transition": float(np.mean([r.transition_accuracy for r in all_results[key]])),
            "ece": float(np.mean(all_eces[key])),
        }

    out_path = results_dir / "label_smoothing_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)
    return output


if __name__ == "__main__":
    run_label_smoothing_test()
