#!/usr/bin/env python3
"""EnCQR: Bootstrap Ensemble Conformal — removes exchangeability assumption.

Ensemble Conformalized Quantile Regression (Xu & Xie 2021, adapted for
classification). Train B=5 bootstrap models, use out-of-bag (OOB) predictions
for calibration scores. On test: ensemble via averaged softmax, abstain when
models disagree (prediction interval width > threshold).

Key insight: OOB calibration scores do NOT require exchangeability because
each sample's score comes from models that never saw it, creating a valid
split-conformal setup without explicit data splitting.

Usage:
    python -m scripts.feature_tests.test_encqr
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

B = 5                    # number of bootstrap models
BOOTSTRAP_FRAC = 0.80   # fraction of training data per bootstrap (with replacement)
ABSTENTION_THRESHOLDS = [0, 1, 2, 3]  # max class spread to allow trading
SEED_BASE = 42


def create_bootstrap_indices(
    n_total: int, n_bootstrap: int, frac: float, rng: np.random.Generator,
) -> list[np.ndarray]:
    """Create B bootstrap sample index arrays (with replacement)."""
    n_sample = int(n_total * frac)
    indices = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_total, size=n_sample, replace=True)
        indices.append(idx)
    return indices


def get_oob_mask(bootstrap_indices: list[np.ndarray], n_total: int) -> np.ndarray:
    """For each sample, return a boolean mask of which models are OOB.

    Returns:
        (n_total, B) boolean array. oob[i, b] = True if sample i was NOT
        in bootstrap b's training set.
    """
    n_bootstrap = len(bootstrap_indices)
    oob = np.ones((n_total, n_bootstrap), dtype=bool)
    for b, idx in enumerate(bootstrap_indices):
        unique_idx = np.unique(idx)
        oob[unique_idx, b] = False
    return oob


def selective_econ_dir_with_abstention(
    ensemble_preds: np.ndarray,
    model_preds: np.ndarray,
    actual_returns: np.ndarray,
    valid_mask: np.ndarray,
    max_spread: int,
    mid: int,
) -> tuple[float, float, int]:
    """Selective econ dir with abstention based on model disagreement.

    Args:
        ensemble_preds: Ensemble prediction labels (N,).
        model_preds: Per-model predictions (N, B).
        actual_returns: Actual returns (N,).
        valid_mask: Boolean validity mask (N,).
        max_spread: Maximum allowed class spread (max - min across models).
        mid: Middle class index.

    Returns:
        (accuracy, coverage, n_traded)
    """
    n = len(ensemble_preds)
    traded_correct = 0
    traded_total = 0
    total_valid = 0

    for i in range(n):
        if not valid_mask[i]:
            continue
        if abs(actual_returns[i]) <= MIN_RETURN_THRESHOLD:
            continue
        total_valid += 1

        # Check model spread
        spread = int(model_preds[i].max() - model_preds[i].min())
        if spread > max_spread:
            continue  # abstain

        # Check directionality
        pred = ensemble_preds[i]
        if pred == mid:
            continue  # flat prediction, skip

        pred_dir = 1.0 if pred > mid else -1.0
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
    logger.info("EnCQR: Bootstrap ensemble conformal prediction (B=%d)", B)

    # --- Data ---
    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))
    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)

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

    # Class weights from full training set
    valid_labels = y_train[tr_valid]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    # --- Create bootstrap samples ---
    rng = np.random.default_rng(SEED_BASE)
    n_total = len(X_train)
    bootstrap_indices = create_bootstrap_indices(n_total, B, BOOTSTRAP_FRAC, rng)
    oob_mask = get_oob_mask(bootstrap_indices, n_total)
    logger.info("Bootstrap samples created: %d models, %.0f%% each",
                B, BOOTSTRAP_FRAC * 100)

    # Report OOB statistics
    oob_counts = oob_mask.sum(axis=1)
    logger.info("OOB model counts per sample: mean=%.1f, min=%d, max=%d",
                oob_counts.mean(), oob_counts.min(), oob_counts.max())
    samples_with_oob = (oob_counts > 0).sum()
    logger.info("Samples with at least 1 OOB model: %d / %d (%.1f%%)",
                samples_with_oob, n_total, 100 * samples_with_oob / n_total)

    # --- Train B bootstrap models ---
    models: list[WaveletGPT] = []
    train_times: list[float] = []

    for b in range(B):
        seed = SEED_BASE + b * 17
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        logger.info("Training bootstrap model %d/%d (seed=%d)...", b + 1, B, seed)
        idx = bootstrap_indices[b]
        X_b = X_train[idx]
        y_b = y_train[idx]

        model = WaveletGPT(
            vocab_size=1, context_length=CONTEXT_LENGTH,
            task="return_quantile", n_output_classes=N_CLASSES,
            class_weights=class_weights, input_mode="continuous",
            n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
        )
        t0 = time.time()
        model.fit(X_b, y_b.astype(np.float64))
        elapsed = time.time() - t0
        models.append(model)
        train_times.append(elapsed)
        logger.info("  Model %d trained in %.1fs", b + 1, elapsed)

    total_train_time = sum(train_times)
    logger.info("Total training time: %.1fs", total_train_time)

    # --- OOB calibration scores ---
    logger.info("Computing OOB calibration scores...")
    # For each training sample, get predictions from OOB models
    # OOB nonconformity = 1 - (fraction of OOB models predicting true class)
    train_probs_per_model = []
    for b in range(B):
        probs = models[b].predict_proba(X_train)  # (n_total, N_CLASSES)
        train_probs_per_model.append(probs)

    oob_scores = np.full(n_total, np.nan)
    for i in range(n_total):
        oob_models = np.where(oob_mask[i])[0]
        if len(oob_models) == 0:
            continue
        # Average softmax from OOB models
        oob_probs = np.mean([train_probs_per_model[b][i] for b in oob_models], axis=0)
        oob_scores[i] = 1.0 - oob_probs[y_train[i]]

    valid_oob = ~np.isnan(oob_scores) & tr_valid
    n_valid_oob = int(valid_oob.sum())
    logger.info("Valid OOB scores: %d (%.1f%% of training data)",
                n_valid_oob, 100 * n_valid_oob / n_total)

    # --- Test predictions ---
    logger.info("Computing ensemble test predictions...")
    test_probs_per_model = np.zeros((len(X_test), B, N_CLASSES))
    test_preds_per_model = np.zeros((len(X_test), B), dtype=np.int64)

    for b in range(B):
        probs = models[b].predict_proba(X_test)
        test_probs_per_model[:, b, :] = probs
        test_preds_per_model[:, b] = probs.argmax(axis=1)

    # Ensemble: averaged softmax
    ensemble_probs = test_probs_per_model.mean(axis=1)  # (N, C)
    ensemble_preds = ensemble_probs.argmax(axis=1).astype(np.int64)

    # Majority vote
    from scipy.stats import mode as scipy_mode
    majority_preds = scipy_mode(test_preds_per_model, axis=1, keepdims=False).mode.astype(np.int64)

    # Model spread (disagreement)
    spreads = test_preds_per_model.max(axis=1) - test_preds_per_model.min(axis=1)

    # --- Full 5-metric evaluation on ensemble ---
    m_ensemble = evaluate_5_metrics(ensemble_preds, te_rets, y_test, te_valid)
    print_5_metrics("Ensemble (averaged softmax)", m_ensemble)

    m_majority = evaluate_5_metrics(majority_preds, te_rets, y_test, te_valid)
    print_5_metrics("Ensemble (majority vote)", m_majority)

    # --- Sweep abstention thresholds ---
    w = 100
    print("\n" + "=" * w)
    print("  EnCQR: BOOTSTRAP ENSEMBLE SELECTIVE TRADING")
    print("=" * w)
    print(f"  {'Max Spread':<12} {'Sel Econ Dir':>13} {'Coverage':>10}"
          f" {'N Traded':>10} {'Pct Spread=0':>13}")
    print("-" * w)

    threshold_results = {}
    for max_s in ABSTENTION_THRESHOLDS:
        sel_acc, sel_cov, n_traded = selective_econ_dir_with_abstention(
            ensemble_preds, test_preds_per_model, te_rets, te_valid, max_s, MID,
        )
        pct_low_spread = float((spreads[te_valid] <= max_s).mean())

        threshold_results[f"max_spread_{max_s}"] = {
            "max_spread": max_s,
            "selective_econ_dir": sel_acc,
            "coverage": sel_cov,
            "n_traded": n_traded,
            "pct_within_spread": pct_low_spread,
        }

        print(f"  {max_s:<12d} {sel_acc:>12.1%} {sel_cov:>9.1%}"
              f" {n_traded:>10d} {pct_low_spread:>12.1%}")

    print("-" * w)

    # Spread statistics
    valid_spreads = spreads[te_valid]
    print(f"\n  Spread stats: mean={valid_spreads.mean():.2f},"
          f" median={np.median(valid_spreads):.1f},"
          f" max={valid_spreads.max()}")
    for s_val in range(N_CLASSES):
        pct = float((valid_spreads == s_val).mean())
        if pct > 0.001:
            print(f"    spread={s_val}: {pct:.1%}")

    # --- Pass criteria ---
    # Find best threshold with coverage > 30%
    best_result = None
    for max_s in ABSTENTION_THRESHOLDS:
        r = threshold_results[f"max_spread_{max_s}"]
        if r["coverage"] > 0.30 and (best_result is None
                                      or r["selective_econ_dir"] > best_result["selective_econ_dir"]):
            best_result = r

    if best_result is not None:
        passed = best_result["selective_econ_dir"] > 0.67
        verdict_detail = (f"Best: spread<={best_result['max_spread']},"
                          f" econ_dir={best_result['selective_econ_dir']:.1%},"
                          f" coverage={best_result['coverage']:.1%}")
    else:
        passed = False
        verdict_detail = "No threshold achieves coverage > 30%"

    verdict = "PASS" if passed else "FAIL"

    print("\n  Pass criteria: selective_econ_dir > 67% at coverage > 30%")
    print(f"  {verdict_detail}")
    print(f"\n  VERDICT: {verdict}")
    print("=" * w)

    save_results("encqr", {
        "ensemble_avg_metrics": m_ensemble,
        "ensemble_majority_metrics": m_majority,
        "threshold_results": threshold_results,
        "spread_stats": {
            "mean": float(valid_spreads.mean()),
            "median": float(np.median(valid_spreads)),
            "max": int(valid_spreads.max()),
        },
        "oob_stats": {
            "n_valid_oob": n_valid_oob,
            "oob_score_mean": float(oob_scores[valid_oob].mean()),
        },
        "n_bootstrap": B,
        "bootstrap_frac": BOOTSTRAP_FRAC,
        "train_times": train_times,
        "total_train_time": total_train_time,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
    })


if __name__ == "__main__":
    main()
