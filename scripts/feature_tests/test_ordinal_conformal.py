#!/usr/bin/env python3
"""Ordinal Conformal Prediction: contiguous prediction sets respecting class ordering.

Standard conformal prediction builds arbitrary prediction sets. For ordinal
targets (down < flat < up), we enforce contiguity: sets must be intervals
{k_min, ..., k_max}. Only trade directional singletons ({0 or 1} = down,
{3 or 4} = up) for high-conviction selective trading.

For each alpha, we:
1. Compute ordinal nonconformity scores on calibration data.
2. Build contiguous prediction sets on test data.
3. Only trade singleton directional sets, abstaining on ambiguous intervals.

Usage:
    python -m scripts.feature_tests.test_ordinal_conformal
"""

from __future__ import annotations  # noqa: I001

import logging
import time

import numpy as np
import torch

from scripts.feature_tests.exp2_helpers import (
    MID,
    evaluate_5_metrics,
    print_5_metrics,
    save_results,
)
from scripts.feature_tests.harness import (
    CONTEXT_LENGTH,
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
from wavecast.targets.returns import assign_quantile_labels, compute_quantile_boundaries

logger = logging.getLogger(__name__)

ALPHAS = [0.05, 0.10, 0.20]
CAL_FRAC = 0.20  # fraction of training data held out for calibration


def ordinal_nonconformity_score(probs: np.ndarray, true_class: int) -> float:
    """Ordinal nonconformity: 1 minus cumulative probability of the smallest
    contiguous interval around the true class that contains it.

    For ordinal conformal, the score measures how much probability mass must
    be accumulated outward from the true class to cover it.
    """
    return 1.0 - float(probs[true_class])


def build_ordinal_prediction_set(
    probs: np.ndarray, threshold: float, n_classes: int,
) -> list[int]:
    """Build a contiguous ordinal prediction set by expanding outward from argmax.

    Start at the most probable class. Expand to adjacent classes (left or right,
    whichever has higher probability) until the accumulated score <= threshold,
    i.e. 1 - sum(probs in set) <= threshold, or equivalently sum >= 1 - threshold.

    Returns:
        Sorted list of class indices in the prediction set.
    """
    center = int(np.argmax(probs))
    pred_set = {center}
    cumulative = float(probs[center])
    lo, hi = center, center

    while cumulative < (1.0 - threshold) and (lo > 0 or hi < n_classes - 1):
        # Expand to whichever adjacent class has higher probability
        left_prob = float(probs[lo - 1]) if lo > 0 else -1.0
        right_prob = float(probs[hi + 1]) if hi < n_classes - 1 else -1.0

        if left_prob >= right_prob and lo > 0:
            lo -= 1
            pred_set.add(lo)
            cumulative += left_prob
        elif hi < n_classes - 1:
            hi += 1
            pred_set.add(hi)
            cumulative += right_prob
        else:
            break

    return sorted(pred_set)


def is_directional_singleton(pred_set: list[int], mid: int) -> bool:
    """Check if prediction set is a singleton that is directional (not flat).

    Directional: class 0 or 1 (down) or class 3 or 4 (up).
    """
    if len(pred_set) != 1:
        return False
    cls = pred_set[0]
    return cls != mid


def selective_econ_dir(
    pred_sets: list[list[int]],
    actual_returns: np.ndarray,
    valid_mask: np.ndarray,
    mid: int,
    min_return_threshold: float = 0.001,
) -> tuple[float, float, int]:
    """Compute selective economic directional accuracy on traded samples.

    Returns:
        (selective_accuracy, coverage_fraction, n_traded)
    """
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        return 0.5, 0.0, 0

    traded_correct = 0
    traded_total = 0
    total_valid = 0

    for i in range(len(pred_sets)):
        if not valid_mask[i]:
            continue
        if abs(actual_returns[i]) <= min_return_threshold:
            continue
        total_valid += 1

        pred_set = pred_sets[i]
        if not is_directional_singleton(pred_set, mid):
            continue

        cls = pred_set[0]
        pred_dir = 1.0 if cls > mid else -1.0
        actual_dir = float(np.sign(actual_returns[i]))
        if actual_dir == 0:
            continue

        traded_total += 1
        if pred_dir == actual_dir:
            traded_correct += 1

    if traded_total == 0:
        return 0.5, 0.0, 0

    accuracy = traded_correct / traded_total
    coverage = traded_total / max(total_valid, 1)
    return accuracy, coverage, traded_total


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Ordinal conformal prediction experiment")

    # --- Data ---
    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))
    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)

    X_train_full, tr_rets, tr_valid, tr_lvls, _, _ = _build_d1_pipeline(
        train_prices, None, tickers, None, 0,
    )
    X_test, te_rets, te_valid, te_lvls, _, _ = _build_d1_pipeline(
        test_prices, None, tickers, None, 0,
    )
    logger.info("Train: %d windows, Test: %d windows", len(X_train_full), len(X_test))

    boundaries = compute_quantile_boundaries(tr_rets, tr_lvls, tr_valid, PERCENTILES, per_level=True)
    y_train_full = assign_quantile_labels(tr_rets, tr_lvls, boundaries)
    y_test = assign_quantile_labels(te_rets, te_lvls, boundaries)

    # --- Split training into proper train (80%) and calibration (20%) ---
    n_total = len(X_train_full)
    n_cal = int(n_total * CAL_FRAC)
    n_train = n_total - n_cal

    rng = np.random.default_rng(42)
    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]
    cal_idx = perm[n_train:]

    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]
    X_cal = X_train_full[cal_idx]
    y_cal = y_train_full[cal_idx]
    cal_valid = tr_valid[cal_idx]
    logger.info("Proper train: %d, Calibration: %d", n_train, n_cal)

    # Class weights from proper training set
    valid_labels = y_train[tr_valid[train_idx]]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    # --- Train CE baseline ---
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("Training CE baseline on proper train split...")
    model = WaveletGPT(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=N_CLASSES,
        class_weights=class_weights, input_mode="continuous",
        n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
    )
    t0 = time.time()
    model.fit(X_train, y_train.astype(np.float64))
    train_time = time.time() - t0

    # --- Calibration: compute nonconformity scores ---
    logger.info("Computing calibration nonconformity scores...")
    cal_probs = model.predict_proba(X_cal)  # (n_cal, N_CLASSES)
    cal_scores = np.array([
        ordinal_nonconformity_score(cal_probs[i], y_cal[i])
        for i in range(n_cal)
        if cal_valid[i]
    ])
    logger.info("Calibration scores: %d samples, mean=%.4f, std=%.4f",
                len(cal_scores), cal_scores.mean(), cal_scores.std())

    # --- Test predictions ---
    test_probs = model.predict_proba(X_test)
    pred_labels = test_probs.argmax(axis=1).astype(np.int64)

    # Full 5-metric evaluation (not selective)
    m_full = evaluate_5_metrics(pred_labels, te_rets, y_test, te_valid)
    print_5_metrics("CE Baseline (full test set)", m_full)

    # --- Sweep alpha levels ---
    alpha_results = {}
    w = 100

    print("\n" + "=" * w)
    print("  ORDINAL CONFORMAL PREDICTION RESULTS")
    print("=" * w)
    print(f"  {'Alpha':<8} {'Threshold':>10} {'Sel Econ Dir':>13} {'Coverage':>10}"
          f" {'N Traded':>10} {'Avg Set Size':>13}")
    print("-" * w)

    for alpha in ALPHAS:
        # Threshold: (1-alpha) quantile of calibration scores
        q_level = min(1.0, (1.0 - alpha) * (1 + 1.0 / len(cal_scores)))
        threshold = float(np.quantile(cal_scores, min(q_level, 1.0)))

        # Build prediction sets for test data
        pred_sets: list[list[int]] = []
        for i in range(len(X_test)):
            ps = build_ordinal_prediction_set(test_probs[i], threshold, N_CLASSES)
            pred_sets.append(ps)

        # Selective trading metrics
        sel_acc, coverage, n_traded = selective_econ_dir(
            pred_sets, te_rets, te_valid, MID,
        )

        # Average set size
        valid_sets = [pred_sets[i] for i in range(len(pred_sets)) if te_valid[i]]
        avg_set_size = float(np.mean([len(s) for s in valid_sets])) if valid_sets else 0.0

        # Coverage check: how often true label is in prediction set
        marginal_coverage = 0.0
        n_covered = 0
        n_checked = 0
        for i in range(len(pred_sets)):
            if not te_valid[i]:
                continue
            n_checked += 1
            if y_test[i] in pred_sets[i]:
                n_covered += 1
        if n_checked > 0:
            marginal_coverage = n_covered / n_checked

        alpha_results[f"alpha_{alpha:.2f}"] = {
            "alpha": alpha,
            "threshold": threshold,
            "selective_econ_dir": sel_acc,
            "coverage": coverage,
            "n_traded": n_traded,
            "avg_set_size": avg_set_size,
            "marginal_coverage": marginal_coverage,
        }

        print(f"  {alpha:<8.2f} {threshold:>10.4f} {sel_acc:>12.1%} {coverage:>9.1%}"
              f" {n_traded:>10d} {avg_set_size:>13.2f}")

    print("-" * w)

    # --- Pass/Fail ---
    r010 = alpha_results["alpha_0.10"]
    passed = r010["selective_econ_dir"] > 0.67 and r010["coverage"] > 0.25
    verdict = "PASS" if passed else "FAIL"

    print("\n  Marginal coverages: "
          + "  ".join(f"a={a:.2f}: {alpha_results[f'alpha_{a:.2f}']['marginal_coverage']:.1%}"
                      for a in ALPHAS))
    print("\n  Pass criteria (alpha=0.10): selective_econ_dir > 67%, coverage > 25%")
    print(f"  Actual: selective_econ_dir = {r010['selective_econ_dir']:.1%},"
          f" coverage = {r010['coverage']:.1%}")
    print(f"\n  VERDICT: {verdict}")
    print("=" * w)

    save_results("ordinal_conformal", {
        "full_metrics": m_full,
        "alpha_results": alpha_results,
        "n_cal_scores": len(cal_scores),
        "train_time": train_time,
        "verdict": verdict,
    })


if __name__ == "__main__":
    main()
