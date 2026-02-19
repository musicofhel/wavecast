#!/usr/bin/env python3
"""ERAPS: Non-Exchangeable Conformal via Adaptive Conformal Inference (AgACI).

Standard conformal prediction assumes exchangeability between calibration and
test data. Financial time series violate this: distribution shifts, volatility
clustering, and regime changes break the iid assumption.

AgACI (Gibbs 2021) maintains a running alpha_t that adapts online:
  alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
where err_t = 1 if y_t not in C_t. This adapts the coverage guarantee to
non-stationary data.

We also apply exponential reweighting of calibration scores:
  w_i = exp(-beta * (t - i))
giving more weight to recent scores and less to stale ones.

Compare vs standard conformal (equal weights, fixed alpha).

Usage:
    python -m scripts.feature_tests.test_eraps
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
from wavecast.targets.returns import assign_quantile_labels, compute_quantile_boundaries

logger = logging.getLogger(__name__)

TARGET_ALPHA = 0.10  # target miscoverage rate (1 - coverage = 0.90)
AGACI_GAMMA = 0.01   # learning rate for adaptive alpha
BETA = 0.01          # exponential decay rate for reweighting
CAL_FRAC = 0.20      # calibration fraction of training data
ROLLING_WINDOW = 200  # window for coverage stability analysis


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Compute weighted quantile.

    Args:
        values: Array of values.
        weights: Corresponding weights (need not sum to 1).
        q: Quantile level in [0, 1].

    Returns:
        Weighted quantile value.
    """
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cum_weights = np.cumsum(sorted_weights)
    cum_weights /= cum_weights[-1]  # normalize to [0, 1]
    # Find first index where cumulative weight exceeds q
    idx = np.searchsorted(cum_weights, q)
    idx = min(idx, len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def build_prediction_set(probs: np.ndarray, threshold: float) -> list[int]:
    """Build prediction set: classes whose softmax prob exceeds 1 - threshold.

    We use the APS (Adaptive Prediction Sets) approach: include classes in
    descending probability order until cumulative prob >= 1 - threshold.
    """
    sorted_idx = np.argsort(-probs)  # descending
    pred_set = []
    cumulative = 0.0
    target = 1.0 - threshold

    for idx in sorted_idx:
        pred_set.append(int(idx))
        cumulative += probs[idx]
        if cumulative >= target:
            break

    return sorted(pred_set)


def run_standard_conformal(
    cal_scores: np.ndarray,
    test_probs: np.ndarray,
    y_test: np.ndarray,
    te_valid: np.ndarray,
    alpha: float,
) -> dict:
    """Standard conformal with fixed alpha and equal weights."""
    # Threshold at (1-alpha)(1 + 1/n) quantile of calibration scores
    q_level = min(1.0, (1.0 - alpha) * (1 + 1.0 / len(cal_scores)))
    threshold = float(np.quantile(cal_scores, min(q_level, 1.0)))

    pred_sets: list[list[int]] = []
    for i in range(len(test_probs)):
        ps = build_prediction_set(test_probs[i], threshold)
        pred_sets.append(ps)

    # Marginal coverage
    n_covered = 0
    n_checked = 0
    for i in range(len(pred_sets)):
        if not te_valid[i]:
            continue
        n_checked += 1
        if y_test[i] in pred_sets[i]:
            n_covered += 1

    marginal_coverage = n_covered / max(n_checked, 1)
    avg_set_size = float(np.mean([
        len(pred_sets[i]) for i in range(len(pred_sets)) if te_valid[i]
    ]))

    return {
        "method": "standard",
        "threshold": threshold,
        "marginal_coverage": marginal_coverage,
        "avg_set_size": avg_set_size,
        "pred_sets": pred_sets,
    }


def run_agaci_conformal(
    cal_scores: np.ndarray,
    cal_times: np.ndarray,
    test_probs: np.ndarray,
    y_test: np.ndarray,
    te_valid: np.ndarray,
    alpha: float,
    gamma: float,
    beta: float,
) -> dict:
    """AgACI: Adaptive conformal with exponential reweighting.

    Processes test samples sequentially, adapting alpha_t online.
    """
    n_cal = len(cal_scores)
    n_test = len(test_probs)

    # Initialize
    alpha_t = alpha
    pred_sets: list[list[int]] = []
    coverages: list[float] = []
    alpha_history: list[float] = []

    for t in range(n_test):
        # Compute exponentially weighted calibration scores
        weights = np.exp(-beta * (n_cal + t - cal_times))
        weights = weights / weights.sum()

        # Weighted quantile of calibration scores at (1 - alpha_t) level
        q_level = min(1.0 - alpha_t, 0.999)
        q_level = max(q_level, 0.001)
        threshold = weighted_quantile(cal_scores, weights, q_level)

        # Build prediction set
        ps = build_prediction_set(test_probs[t], threshold)
        pred_sets.append(ps)

        # Check coverage and adapt alpha
        if te_valid[t]:
            err_t = 0.0 if y_test[t] in ps else 1.0
            coverages.append(1.0 - err_t)
            # AgACI update: increase alpha if covered (shrink sets), decrease if missed
            alpha_t = alpha_t + gamma * (alpha - err_t)
            # Clamp to valid range
            alpha_t = max(0.001, min(0.999, alpha_t))
        else:
            coverages.append(float("nan"))

        alpha_history.append(alpha_t)

    # Compute metrics
    valid_coverages = [c for c in coverages if not np.isnan(c)]
    marginal_coverage = float(np.mean(valid_coverages)) if valid_coverages else 0.0

    # Coverage stability: max deviation from target in rolling windows
    max_deviation = 0.0
    target_coverage = 1.0 - alpha
    if len(valid_coverages) >= ROLLING_WINDOW:
        for start in range(0, len(valid_coverages) - ROLLING_WINDOW + 1, ROLLING_WINDOW // 4):
            window = valid_coverages[start:start + ROLLING_WINDOW]
            window_cov = float(np.mean(window))
            dev = abs(window_cov - target_coverage)
            max_deviation = max(max_deviation, dev)
    elif valid_coverages:
        max_deviation = abs(marginal_coverage - target_coverage)

    avg_set_size = float(np.mean([
        len(pred_sets[i]) for i in range(len(pred_sets)) if te_valid[i]
    ]))

    return {
        "method": "agaci",
        "marginal_coverage": marginal_coverage,
        "coverage_stability": max_deviation,
        "avg_set_size": avg_set_size,
        "final_alpha": alpha_t,
        "alpha_range": (min(alpha_history), max(alpha_history)),
        "pred_sets": pred_sets,
    }


def selective_econ_dir(
    pred_sets: list[list[int]],
    actual_returns: np.ndarray,
    valid_mask: np.ndarray,
    mid: int,
) -> tuple[float, float, int]:
    """Selective econ dir: only trade singleton directional sets."""
    traded_correct = 0
    traded_total = 0
    total_valid = 0

    for i in range(len(pred_sets)):
        if not valid_mask[i]:
            continue
        if abs(actual_returns[i]) <= MIN_RETURN_THRESHOLD:
            continue
        total_valid += 1

        ps = pred_sets[i]
        if len(ps) != 1 or ps[0] == mid:
            continue

        cls = ps[0]
        pred_dir = 1.0 if cls > mid else -1.0
        actual_dir = float(np.sign(actual_returns[i]))
        if actual_dir == 0:
            continue

        traded_total += 1
        if pred_dir == actual_dir:
            traded_correct += 1

    if traded_total == 0:
        return 0.5, 0.0, 0
    return traded_correct / traded_total, traded_total / max(total_valid, 1), traded_total


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("ERAPS: Non-exchangeable conformal via AgACI")

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

    # --- Split training into proper train + calibration ---
    n_total = len(X_train_full)
    n_cal = int(n_total * CAL_FRAC)
    n_train = n_total - n_cal

    # Keep temporal ordering: last CAL_FRAC for calibration
    train_idx = np.arange(n_train)
    cal_idx = np.arange(n_train, n_total)

    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]
    X_cal = X_train_full[cal_idx]
    y_cal = y_train_full[cal_idx]
    cal_valid = tr_valid[cal_idx]

    logger.info("Proper train: %d, Calibration: %d", n_train, n_cal)

    # Class weights
    valid_labels = y_train[tr_valid[train_idx]]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    # --- Train model ---
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("Training CE baseline...")
    model = WaveletGPT(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=N_CLASSES,
        class_weights=class_weights, input_mode="continuous",
        n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
    )
    t0 = time.time()
    model.fit(X_train, y_train.astype(np.float64))
    train_time = time.time() - t0

    # --- Calibration scores ---
    logger.info("Computing calibration scores...")
    cal_probs = model.predict_proba(X_cal)
    cal_scores_all = np.array([
        1.0 - cal_probs[i, y_cal[i]] for i in range(n_cal)
    ])
    # Use only valid calibration samples
    valid_cal_idx = np.where(cal_valid)[0]
    cal_scores = cal_scores_all[valid_cal_idx]
    cal_times = valid_cal_idx.astype(np.float64)  # temporal indices

    logger.info("Valid calibration scores: %d, mean=%.4f", len(cal_scores), cal_scores.mean())

    # --- Test probabilities ---
    test_probs = model.predict_proba(X_test)
    pred_labels = test_probs.argmax(axis=1).astype(np.int64)

    # Full 5-metric evaluation
    m_full = evaluate_5_metrics(pred_labels, te_rets, y_test, te_valid)
    print_5_metrics("CE Baseline (full test set)", m_full)

    # --- Standard conformal ---
    logger.info("Running standard conformal (alpha=%.2f)...", TARGET_ALPHA)
    std_result = run_standard_conformal(cal_scores, test_probs, y_test, te_valid, TARGET_ALPHA)
    std_sel_acc, std_sel_cov, std_n_traded = selective_econ_dir(
        std_result["pred_sets"], te_rets, te_valid, MID,
    )

    # --- AgACI conformal ---
    logger.info("Running AgACI conformal (alpha=%.2f, gamma=%.3f, beta=%.3f)...",
                TARGET_ALPHA, AGACI_GAMMA, BETA)
    agaci_result = run_agaci_conformal(
        cal_scores, cal_times, test_probs, y_test, te_valid,
        TARGET_ALPHA, AGACI_GAMMA, BETA,
    )
    agaci_sel_acc, agaci_sel_cov, agaci_n_traded = selective_econ_dir(
        agaci_result["pred_sets"], te_rets, te_valid, MID,
    )

    # --- Results ---
    w = 100
    print("\n" + "=" * w)
    print("  ERAPS: NON-EXCHANGEABLE CONFORMAL PREDICTION")
    print("=" * w)
    print(f"  {'Method':<20} {'Marg Cov':>10} {'Stab (<5pp)':>12} {'Set Size':>10}"
          f" {'Sel Econ':>10} {'Sel Cov':>9} {'N Traded':>10}")
    print("-" * w)
    print(f"  {'Standard':<20} {std_result['marginal_coverage']:>9.1%}"
          f" {'  N/A':>12} {std_result['avg_set_size']:>10.2f}"
          f" {std_sel_acc:>9.1%} {std_sel_cov:>8.1%} {std_n_traded:>10d}")
    print(f"  {'AgACI':<20} {agaci_result['marginal_coverage']:>9.1%}"
          f" {agaci_result['coverage_stability']:>11.1%} {agaci_result['avg_set_size']:>10.2f}"
          f" {agaci_sel_acc:>9.1%} {agaci_sel_cov:>8.1%} {agaci_n_traded:>10d}")
    print("-" * w)

    print(f"\n  AgACI alpha range: [{agaci_result['alpha_range'][0]:.4f},"
          f" {agaci_result['alpha_range'][1]:.4f}]")
    print(f"  AgACI final alpha: {agaci_result['final_alpha']:.4f}")

    # Pass criteria
    agaci_cov_ok = agaci_result["marginal_coverage"] >= 0.88
    agaci_stab_ok = agaci_result["coverage_stability"] < 0.05
    agaci_sel_ok = agaci_sel_acc > 0.66
    passed = agaci_cov_ok and agaci_stab_ok and agaci_sel_ok
    verdict = "PASS" if passed else "FAIL"

    reasons = []
    if not agaci_cov_ok:
        reasons.append(f"marginal_coverage {agaci_result['marginal_coverage']:.1%} < 88%")
    if not agaci_stab_ok:
        reasons.append(f"stability {agaci_result['coverage_stability']:.1%} >= 5pp")
    if not agaci_sel_ok:
        reasons.append(f"selective_econ_dir {agaci_sel_acc:.1%} <= 66%")

    print("\n  Pass criteria: marginal_coverage >= 88%, stability < 5pp, selective_econ_dir > 66%")
    if reasons:
        print(f"  Failed: {'; '.join(reasons)}")
    print(f"\n  VERDICT: {verdict}")
    print("=" * w)

    save_results("eraps", {
        "full_metrics": m_full,
        "standard_conformal": {
            k: v for k, v in std_result.items() if k != "pred_sets"
        },
        "standard_selective": {
            "econ_dir": std_sel_acc, "coverage": std_sel_cov, "n_traded": std_n_traded,
        },
        "agaci_conformal": {
            k: v for k, v in agaci_result.items() if k != "pred_sets"
        },
        "agaci_selective": {
            "econ_dir": agaci_sel_acc, "coverage": agaci_sel_cov, "n_traded": agaci_n_traded,
        },
        "config": {
            "alpha": TARGET_ALPHA, "gamma": AGACI_GAMMA, "beta": BETA,
            "cal_frac": CAL_FRAC, "rolling_window": ROLLING_WINDOW,
        },
        "train_time": train_time,
        "verdict": verdict,
    })


if __name__ == "__main__":
    main()
