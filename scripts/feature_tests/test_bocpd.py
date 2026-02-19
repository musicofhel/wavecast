#!/usr/bin/env python3
"""Feature Test: BOCPD (Bayesian Online Changepoint Detection).

Hypothesis: Run-length posterior on D1 coefficient deltas captures regime
structure that raw deltas miss. The model should improve transition accuracy
because BOCPD is specifically designed to detect changepoints in streaming data.

2 new features per timestep:
  - run_length_mean: E[r] = expected run length from the posterior.
    High = stable regime, low = recent changepoint.
  - changepoint_prob: P(r_t = 0) = probability that a changepoint occurred
    at this exact timestep.

Implements simplified BOCPD (Adams & MacKay 2007) with a Gaussian
observation model and constant hazard rate h = 1/250 (mean run length
~1 month of hourly bars).

Usage:
    python -m scripts.feature_tests.test_bocpd
"""

from __future__ import annotations  # noqa: I001

import logging
import time

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from scripts.feature_tests.exp2_helpers import (
    evaluate_5_metrics,
    print_5_metrics,
    save_results,
    verdict_from_metrics,
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

N_NEW_FEATURES = 2
HAZARD_RATE = 1.0 / 250.0  # Mean run length ~250 bars (1 month hourly)
MAX_RUN_LENGTH = 300  # Truncate posterior for efficiency
PRIOR_MU = 0.0
PRIOR_KAPPA = 1.0  # Prior precision (observations per prior "sample")
PRIOR_ALPHA = 1.0  # Inverse-gamma shape
PRIOR_BETA = 0.05  # Inverse-gamma scale (controls initial variance estimate)


def _bocpd_run_length_posterior(
    data: NDArray,
    hazard: float = HAZARD_RATE,
    max_run: int = MAX_RUN_LENGTH,
) -> tuple[NDArray, NDArray]:
    """Compute BOCPD run-length posterior on a 1-D sequence.

    Uses a Normal-Inverse-Gamma conjugate model for the observation
    likelihood, with online sufficient-statistic updates.

    Args:
        data: 1-D array of observations (coefficient deltas).
        hazard: Constant hazard rate (probability of changepoint per step).
        max_run: Maximum run length to track (truncation for efficiency).

    Returns:
        run_length_mean: (T,) array of E[r_t] at each timestep.
        changepoint_prob: (T,) array of P(r_t = 0) at each timestep.
    """
    n = len(data)
    run_length_mean = np.zeros(n, dtype=np.float64)
    changepoint_prob = np.zeros(n, dtype=np.float64)

    # Sufficient statistics for each run-length hypothesis.
    # We track arrays of length (max_run + 1) for run lengths 0..max_run.
    mu = np.full(max_run + 1, PRIOR_MU, dtype=np.float64)
    kappa = np.full(max_run + 1, PRIOR_KAPPA, dtype=np.float64)
    alpha = np.full(max_run + 1, PRIOR_ALPHA, dtype=np.float64)
    beta = np.full(max_run + 1, PRIOR_BETA, dtype=np.float64)

    # Run-length probabilities: R[r] = P(run_length = r)
    # Start with R[0] = 1 (we begin with a fresh run).
    R = np.zeros(max_run + 1, dtype=np.float64)
    R[0] = 1.0

    for t in range(n):
        x = data[t]

        # --- Predictive probability under each run-length hypothesis ---
        # Student-t predictive: T_{2*alpha}(mu, beta*(kappa+1)/(alpha*kappa))
        pred_var = beta * (kappa + 1.0) / (alpha * kappa)
        pred_var = np.maximum(pred_var, 1e-12)
        dof = 2.0 * alpha
        z = (x - mu) ** 2 / pred_var
        # Student-t log-pdf (up to a constant that cancels in normalization)
        log_pred = -0.5 * (dof + 1.0) * np.log1p(z / dof)

        # Convert to probabilities (relative scale is fine, we normalize)
        pred_prob = np.exp(log_pred - log_pred.max())
        pred_prob = np.maximum(pred_prob, 1e-300)

        # --- Growth and changepoint probabilities ---
        # Growth: P(r_t = r+1) = P(r_{t-1}=r) * pred(x|r) * (1-h)
        grow = R * pred_prob * (1.0 - hazard)
        # Changepoint: P(r_t = 0) = sum P(r_{t-1}=r) * pred(x|r) * h
        cp = np.sum(R * pred_prob * hazard)

        # Shift grow probabilities (run length r+1 comes from old r)
        R_new = np.zeros(max_run + 1, dtype=np.float64)
        R_new[0] = cp
        R_new[1:] = grow[:-1]

        # Normalize
        total = R_new.sum()
        if total > 0:
            R_new /= total
        else:
            R_new[0] = 1.0
        R = R_new

        # --- Extract features ---
        run_indices = np.arange(max_run + 1, dtype=np.float64)
        run_length_mean[t] = np.dot(R, run_indices)
        changepoint_prob[t] = R[0]

        # --- Update sufficient statistics (Normal-Inverse-Gamma) ---
        # For each run-length hypothesis, incorporate the new observation.
        new_mu = (kappa * mu + x) / (kappa + 1.0)
        new_kappa = kappa + 1.0
        new_alpha = alpha + 0.5
        new_beta = beta + 0.5 * kappa * (x - mu) ** 2 / (kappa + 1.0)

        # Shift: run-length r+1 inherits stats from old r
        mu_shifted = np.empty(max_run + 1, dtype=np.float64)
        kappa_shifted = np.empty(max_run + 1, dtype=np.float64)
        alpha_shifted = np.empty(max_run + 1, dtype=np.float64)
        beta_shifted = np.empty(max_run + 1, dtype=np.float64)

        # Run length 0 resets to prior
        mu_shifted[0] = PRIOR_MU
        kappa_shifted[0] = PRIOR_KAPPA
        alpha_shifted[0] = PRIOR_ALPHA
        beta_shifted[0] = PRIOR_BETA

        # Run lengths 1..max_run inherit updated stats from 0..max_run-1
        mu_shifted[1:] = new_mu[:-1]
        kappa_shifted[1:] = new_kappa[:-1]
        alpha_shifted[1:] = new_alpha[:-1]
        beta_shifted[1:] = new_beta[:-1]

        mu = mu_shifted
        kappa = kappa_shifted
        alpha = alpha_shifted
        beta = beta_shifted

    return run_length_mean, changepoint_prob


def bocpd_features(
    ohlcv_df: pd.DataFrame,
    detail_coeffs: NDArray,
    approx_coeffs: NDArray,
    level: int,
) -> NDArray:
    """BOCPD features on D1 coefficient deltas.

    Returns (n_deltas, 2): [run_length_mean, changepoint_prob].
    """
    n_deltas = len(detail_coeffs) - 1
    if n_deltas <= 0:
        return np.empty((0, N_NEW_FEATURES), dtype=np.float64)

    deltas = np.diff(detail_coeffs)
    rl_mean, cp_prob = _bocpd_run_length_posterior(deltas)

    # Normalize run_length_mean to [0, 1] range for model consumption
    rl_max = rl_mean.max()
    rl_mean_norm = rl_mean / rl_max if rl_max > 0 else rl_mean

    return np.column_stack([rl_mean_norm, cp_prob])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("BOCPD feature test (2 features: run_length_mean, changepoint_prob)")

    # --- Data ---
    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))
    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)
    logger.info("Loaded %d tickers", len(tickers))

    # --- Build pipelines ---
    logger.info("Building baseline dataset (4 aux features)...")
    X_train_base, tr_rets, tr_valid, tr_lvls, _, _ = _build_d1_pipeline(
        train_prices, None, tickers, None, 0,
    )
    X_test_base, te_rets, te_valid, te_lvls, _, _ = _build_d1_pipeline(
        test_prices, None, tickers, None, 0,
    )

    logger.info("Building BOCPD challenger dataset (4 + 2 aux features)...")
    X_train_chal, _, _, _, _, _ = _build_d1_pipeline(
        train_prices, train_ohlcv, tickers, bocpd_features, N_NEW_FEATURES,
    )
    X_test_chal, _, _, _, _, _ = _build_d1_pipeline(
        test_prices, test_ohlcv, tickers, bocpd_features, N_NEW_FEATURES,
    )

    logger.info("Train: %d windows, Test: %d windows", len(X_train_base), len(X_test_base))

    # --- Labels and class weights ---
    boundaries = compute_quantile_boundaries(tr_rets, tr_lvls, tr_valid, PERCENTILES, per_level=True)
    y_train = assign_quantile_labels(tr_rets, tr_lvls, boundaries)
    y_test = assign_quantile_labels(te_rets, te_lvls, boundaries)

    valid_labels = y_train[tr_valid]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Baseline ---
    logger.info("Training CE baseline (4 aux)...")
    model_base = WaveletGPT(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=N_CLASSES,
        class_weights=class_weights, input_mode="continuous",
        n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
    )
    t0 = time.time()
    model_base.fit(X_train_base, y_train.astype(np.float64))
    base_time = time.time() - t0
    pred_base = model_base.predict(X_test_base).astype(np.int64)
    m_base = evaluate_5_metrics(pred_base, te_rets, y_test, te_valid)
    print_5_metrics("CE Baseline (4 aux)", m_base)

    # --- BOCPD Challenger ---
    logger.info("Training BOCPD challenger (4 + 2 aux)...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_chal = WaveletGPT(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=N_CLASSES,
        class_weights=class_weights, input_mode="continuous",
        n_aux_features=N_AUX_FEATURES + N_NEW_FEATURES, **MODEL_KWARGS,
    )
    t0 = time.time()
    model_chal.fit(X_train_chal, y_train.astype(np.float64))
    chal_time = time.time() - t0
    pred_chal = model_chal.predict(X_test_chal).astype(np.int64)
    m_chal = evaluate_5_metrics(pred_chal, te_rets, y_test, te_valid)
    print_5_metrics("BOCPD Challenger (6 aux)", m_chal)

    # --- Comparison ---
    verdict = verdict_from_metrics(m_base, m_chal)

    w = 100
    print("\n" + "=" * w)
    print("  BOCPD FEATURE TEST COMPARISON")
    print("=" * w)
    print(f"  {'Metric':<22} {'Baseline':>10} {'Challenger':>12} {'Delta':>10}")
    print("  " + "-" * (w - 4))
    for key, label in [
        ("econ_dir", "Econ Dir Acc"),
        ("transition_acc", "Transition Acc"),
        ("large_move_acc", "Large-Move Acc"),
        ("sharpe_costs", "Sharpe (+costs)"),
        ("quantile_acc", "Quantile Acc"),
    ]:
        bv = m_base[key]
        cv = m_chal[key]
        delta = cv - bv
        if key == "sharpe_costs":
            print(f"  {label:<22} {bv:>+9.3f} {cv:>+11.3f} {delta:>+9.3f}")
        else:
            print(f"  {label:<22} {bv:>9.1%} {cv:>11.1%} {delta:>+9.1%}")
    print("  " + "-" * (w - 4))
    print(f"  Pred Dist (base):  up={m_base['pred_dist']['up']:.1%}"
          f"  flat={m_base['pred_dist']['flat']:.1%}"
          f"  down={m_base['pred_dist']['down']:.1%}")
    print(f"  Pred Dist (chal):  up={m_chal['pred_dist']['up']:.1%}"
          f"  flat={m_chal['pred_dist']['flat']:.1%}"
          f"  down={m_chal['pred_dist']['down']:.1%}")
    print(f"\n  Baseline train time:    {base_time:.1f}s")
    print(f"  Challenger train time:  {chal_time:.1f}s")
    print(f"\n  VERDICT: {verdict}")
    print("=" * w)

    save_results("bocpd", {
        "baseline": m_base,
        "challenger": m_chal,
        "verdict": verdict,
        "baseline_train_time": base_time,
        "challenger_train_time": chal_time,
        "bocpd_config": {
            "hazard_rate": HAZARD_RATE,
            "max_run_length": MAX_RUN_LENGTH,
            "prior_mu": PRIOR_MU,
            "prior_kappa": PRIOR_KAPPA,
            "prior_alpha": PRIOR_ALPHA,
            "prior_beta": PRIOR_BETA,
        },
    })


if __name__ == "__main__":
    main()
