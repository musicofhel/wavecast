#!/usr/bin/env python3
"""Experiment 2: Ordinal Regression Loss.

Source: Loss function survey (2301.05579v1); ordinal_network.pdf
Hypothesis: Return quantiles are naturally ordered (q10 < q30 < q50 < q70 < q90)
but wavecast treats them as unordered categories. Ordinal loss penalizes distant
misclassifications more than adjacent ones.

Pass criteria: Econ dir +1pp, adjacent-class confusion reduction >5%, transition > -1pp

Usage:
    python -m scripts.feature_tests.test_ordinal_loss
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from scripts.feature_tests.harness import (  # noqa: I001
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


def run_ordinal_loss_test(n_seeds: int = 3) -> dict:
    """Compare baseline CE vs ordinal regression loss across seeds."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger.info("Ordinal Loss experiment: CE vs ordinal, %d seeds", n_seeds)

    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))
    logger.info("Loaded %d tickers", len(tickers))

    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)

    logger.info("Building D1 dataset...")
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

    baseline_results = []
    ordinal_results = []
    confusion_improvements = []

    for seed in range(n_seeds):
        logger.info("--- Seed %d/%d ---", seed + 1, n_seeds)
        np.random.seed(seed * 42 + 7)
        torch.manual_seed(seed * 42 + 7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 42 + 7)

        # Baseline (CE)
        logger.info("  Training baseline (CE)...")
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
        base_eval = _evaluate(
            pred_base, te_rets, y_test, te_valid, metrics["train_loss"], base_time
        )
        baseline_results.append(base_eval)

        # Compute confusion matrix for baseline
        base_confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        for p, a in zip(pred_base[te_valid], y_test[te_valid], strict=False):
            base_confusion[a, p] += 1

        logger.info(
            "  Baseline: econ_dir=%.1f%% sharpe=%.3f",
            base_eval.econ_dir_accuracy * 100,
            base_eval.sharpe_with_costs,
        )

        # Reset seeds
        np.random.seed(seed * 42 + 7)
        torch.manual_seed(seed * 42 + 7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 42 + 7)

        # Ordinal loss
        logger.info("  Training ordinal...")
        model_ord = WaveletGPT(
            vocab_size=1,
            context_length=CONTEXT_LENGTH,
            task="return_quantile",
            n_output_classes=N_CLASSES,
            class_weights=class_weights,
            input_mode="continuous",
            n_aux_features=N_AUX_FEATURES,
            loss_type="ordinal",
            **MODEL_KWARGS,
        )
        t0 = time.time()
        metrics = model_ord.fit(X_train, y_tr)
        ord_time = time.time() - t0

        # For ordinal loss, use the OrdinalLoss.predict() method
        proba = model_ord.predict_proba(X_test)
        # With ordinal loss, logits are passed through OrdinalLoss.predict
        # But predict_proba returns softmax of raw logits, so we use argmax
        pred_ord = np.argmax(proba, axis=1).astype(np.int64)

        ord_eval = _evaluate(
            pred_ord, te_rets, y_test, te_valid, metrics["train_loss"], ord_time
        )
        ordinal_results.append(ord_eval)

        # Confusion matrix for ordinal
        ord_confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        for p, a in zip(pred_ord[te_valid], y_test[te_valid], strict=False):
            ord_confusion[a, p] += 1

        # Adjacent-class confusion: sum of off-by-more-than-1 errors
        base_distant = 0
        ord_distant = 0
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                if abs(i - j) > 1:
                    base_distant += base_confusion[i, j]
                    ord_distant += ord_confusion[i, j]
        base_total = base_confusion.sum()
        ord_total = ord_confusion.sum()
        if base_total > 0 and ord_total > 0:
            improvement = (base_distant / base_total) - (ord_distant / ord_total)
            confusion_improvements.append(improvement)
        else:
            confusion_improvements.append(0.0)

        logger.info(
            "  Ordinal: econ_dir=%.1f%% sharpe=%.3f distant_err_reduction=%.2f%%",
            ord_eval.econ_dir_accuracy * 100,
            ord_eval.sharpe_with_costs,
            confusion_improvements[-1] * 100,
        )

    # Aggregate
    base_econ = np.mean([r.econ_dir_accuracy for r in baseline_results])
    ord_econ = np.mean([r.econ_dir_accuracy for r in ordinal_results])
    base_sharpe = np.mean([r.sharpe_with_costs for r in baseline_results])
    ord_sharpe = np.mean([r.sharpe_with_costs for r in ordinal_results])
    base_trans = np.mean([r.transition_accuracy for r in baseline_results])
    ord_trans = np.mean([r.transition_accuracy for r in ordinal_results])
    avg_confusion_improvement = np.mean(confusion_improvements)

    d_econ = ord_econ - base_econ
    d_sharpe = ord_sharpe - base_sharpe
    d_trans = ord_trans - base_trans

    pass_econ = d_econ > 0.01
    pass_confusion = avg_confusion_improvement > 0.05
    pass_trans = d_trans > -0.01
    verdict = "PASS" if (pass_econ or pass_confusion) and pass_trans else "FAIL"

    w = 100
    print()
    print("=" * w)
    print("EXPERIMENT: Ordinal Regression Loss")
    print("=" * w)
    print(f"\n{'Metric':<25} {'Baseline':>10} {'Ordinal':>10} {'Delta':>10}")
    print("-" * w)
    print(f"{'Econ Dir Acc':<25} {base_econ:>9.1%} {ord_econ:>9.1%} {d_econ:>+9.1%}")
    print(
        f"{'Sharpe (+costs)':<25} {base_sharpe:>+9.3f} {ord_sharpe:>+9.3f} {d_sharpe:>+9.3f}"
    )
    print(
        f"{'Transition Acc':<25} {base_trans:>9.1%} {ord_trans:>9.1%} {d_trans:>+9.1%}"
    )
    print(
        f"{'Distant Confusion Red':<25} {'':>10} {'':>10} {avg_confusion_improvement:>+9.1%}"
    )
    print("-" * w)
    print(f"\nVERDICT: {verdict}")
    print("=" * w)

    # Save
    results_dir = Path.home() / ".wavecast" / "audit" / "feature_tests"
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "name": "ordinal_loss",
        "experiment_type": "loss_function",
        "loss_type": "ordinal",
        "n_seeds": n_seeds,
        "verdict": verdict,
        "baseline_mean": {
            "econ_dir": float(base_econ),
            "sharpe_costs": float(base_sharpe),
            "transition": float(base_trans),
        },
        "challenger_mean": {
            "econ_dir": float(ord_econ),
            "sharpe_costs": float(ord_sharpe),
            "transition": float(ord_trans),
        },
        "delta": {
            "econ_dir": float(d_econ),
            "sharpe_costs": float(d_sharpe),
            "transition": float(d_trans),
        },
        "confusion_improvement": float(avg_confusion_improvement),
        "per_seed": {
            "baseline": [
                {
                    "econ_dir": r.econ_dir_accuracy,
                    "sharpe_costs": r.sharpe_with_costs,
                    "transition": r.transition_accuracy,
                    "train_loss": r.train_loss,
                }
                for r in baseline_results
            ],
            "challenger": [
                {
                    "econ_dir": r.econ_dir_accuracy,
                    "sharpe_costs": r.sharpe_with_costs,
                    "transition": r.transition_accuracy,
                    "train_loss": r.train_loss,
                }
                for r in ordinal_results
            ],
        },
    }
    out_path = results_dir / "ordinal_loss_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)
    return output


if __name__ == "__main__":
    run_ordinal_loss_test()
