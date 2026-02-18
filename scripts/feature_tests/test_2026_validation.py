#!/usr/bin/env python3
"""2026 holdout validation + prediction distribution diagnostic.

Trains CE baseline, heteroscedastic, and hetero+SelectiveNet on ≤2024-06-30.
Tests on both 2025 (comparison) and 2026 (holdout).

Key diagnostic: is heteroscedastic achieving higher econ_dir by shifting
prediction distribution toward continuation (same direction as previous bar)
rather than actually getting smarter?

Reports:
- Prediction distribution (up/flat/down counts)
- Continuation vs reversal prediction rates
- Accuracy on continuations vs reversals separately
- Whether selective filtering selects continuation-biased predictions
- Full coverage-accuracy curves

Usage:
    python -m scripts.feature_tests.test_2026_validation
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scripts.feature_tests.harness import (
    CONTEXT_LENGTH,
    COST_BPS,
    MIN_RETURN_THRESHOLD,
    MODEL_KWARGS,
    N_AUX_FEATURES,
    N_CLASSES,
    PERCENTILES,
    TICKERS,
    TRAIN_END,
    _build_d1_pipeline,
    _ohlcv_to_timeseries,
)

from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
)

logger = logging.getLogger(__name__)

TEST_START = "2025-01-01"
TEST_MID = "2025-07-01"  # Split 2025 into H1 (seen) and H2 (holdout)
COVERAGE_TARGETS = [0.3, 0.5, 0.7, 0.9]


def load_and_split_ohlcv() -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    """Load OHLCV and split into train, test_H1, test_H2.

    Cached data runs to ~2026-01-01. We split 2025 into:
    - H1: Jan-Jun 2025 (same period used in ephemeral branch experiments)
    - H2: Jul-Dec 2025 (never used in any experiment — true holdout)
    """
    cache_dir = Path.home() / ".wavecast" / "cache"
    train: dict[str, pd.DataFrame] = {}
    test_h1: dict[str, pd.DataFrame] = {}
    test_h2: dict[str, pd.DataFrame] = {}

    train_end = pd.Timestamp(TRAIN_END)
    t_start = pd.Timestamp(TEST_START)
    t_mid = pd.Timestamp(TEST_MID)

    for ticker in TICKERS:
        path = cache_dir / f"{ticker}_1h_ohlcv.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        tr = df[df["timestamp"] <= train_end]
        te_h1 = df[(df["timestamp"] >= t_start) & (df["timestamp"] < t_mid)]
        te_h2 = df[df["timestamp"] >= t_mid]

        if len(tr) > 200 and len(te_h1) > 50 and len(te_h2) > 50:
            train[ticker] = tr.reset_index(drop=True)
            test_h1[ticker] = te_h1.reset_index(drop=True)
            test_h2[ticker] = te_h2.reset_index(drop=True)

    return train, test_h1, test_h2


def analyze_predictions(
    pred_labels: NDArray,
    actual_returns: NDArray,
    valid_mask: NDArray,
    variance: NDArray | None = None,
) -> dict:
    """Full diagnostic: distribution, continuation bias, transition accuracy."""
    mid = N_CLASSES // 2
    valid = valid_mask.astype(bool)

    # Predicted direction
    pred_dir = np.zeros(len(pred_labels), dtype=np.float64)
    pred_dir[pred_labels > mid] = 1.0
    pred_dir[pred_labels < mid] = -1.0
    actual_dir = np.sign(actual_returns)

    vp = pred_dir[valid]
    va = actual_dir[valid]
    vr = actual_returns[valid]

    # --- Prediction distribution ---
    n_up = int((vp == 1).sum())
    n_flat = int((vp == 0).sum())
    n_down = int((vp == -1).sum())
    n_total = len(vp)

    # --- Continuation vs reversal ---
    # Previous actual direction (shifted by 1)
    prev_actual = np.zeros_like(va)
    prev_actual[1:] = va[:-1]
    prev_actual[0] = 0  # no previous for first sample

    has_prev = prev_actual != 0
    has_pred = vp != 0
    has_actual = va != 0
    usable = has_prev & has_pred & has_actual

    # Is prediction a continuation (same as prev actual) or reversal?
    is_continuation_pred = (vp == prev_actual) & usable
    is_reversal_pred = (vp != prev_actual) & usable

    n_cont_pred = int(is_continuation_pred.sum())
    n_rev_pred = int(is_reversal_pred.sum())
    n_usable = int(usable.sum())
    cont_rate = n_cont_pred / n_usable if n_usable > 0 else 0.0

    # Is actual a continuation or reversal?
    is_continuation_actual = (va == prev_actual) & usable
    is_reversal_actual = (va != prev_actual) & usable
    actual_cont_rate = float(is_continuation_actual.sum()) / n_usable if n_usable > 0 else 0.0

    # Accuracy on continuations vs reversals (by actual regime)
    correct = (vp == va)
    cont_mask = is_continuation_actual & has_pred
    rev_mask = is_reversal_actual & has_pred
    acc_on_continuations = float(correct[cont_mask].mean()) if cont_mask.sum() > 0 else 0.0
    acc_on_reversals = float(correct[rev_mask].mean()) if rev_mask.sum() > 0 else 0.0

    # --- Standard econ dir ---
    filt = valid & (np.abs(actual_returns) > MIN_RETURN_THRESHOLD)
    if filt.sum() > 0:
        has_p = pred_dir[filt] != 0
        econ_dir = float(
            np.mean(pred_dir[filt][has_p] == actual_dir[filt][has_p]),
        ) if has_p.sum() > 0 else 0.5
    else:
        econ_dir = 0.5

    # --- Transition accuracy ---
    vi = np.where(valid)[0]
    transitions = np.where(np.diff(np.sign(actual_dir[vi])) != 0)[0] + 1
    if len(transitions) > 0:
        trans_acc = float(
            np.mean(pred_dir[vi][transitions] == actual_dir[vi][transitions]),
        )
        n_transitions = len(transitions)
    else:
        trans_acc = 0.5
        n_transitions = 0

    # --- Sharpe with costs ---
    if valid.sum() > 0:
        pnl_raw = pred_dir[valid] * actual_returns[valid]
        pnl_raw = pnl_raw[~np.isnan(pnl_raw)]
        dir_changes = np.abs(np.diff(pred_dir[valid]))
        cost = np.zeros(int(valid.sum()))
        cost[1:] = dir_changes * (COST_BPS / 10000)
        cost[0] = abs(pred_dir[valid][0]) * (COST_BPS / 10000)
        pnl_net = pred_dir[valid] * actual_returns[valid] - cost
        pnl_net = pnl_net[~np.isnan(pnl_net)]
        sharpe = float(
            np.mean(pnl_net) / np.std(pnl_net) * np.sqrt(252 * 7),
        ) if len(pnl_net) > 1 and np.std(pnl_net) > 0 else 0.0
    else:
        sharpe = 0.0

    # --- Selective metrics (if variance provided) ---
    selective = {}
    if variance is not None:
        for cov in COVERAGE_TARGETS:
            v_var = variance[valid]
            threshold = np.quantile(v_var, cov)
            keep = v_var <= threshold
            actual_cov = float(keep.mean())

            if keep.sum() == 0:
                selective[str(cov)] = {
                    "coverage": actual_cov, "econ_dir": 0.0,
                    "transition": 0.0, "cont_rate": 0.0,
                }
                continue

            # Econ dir on kept predictions
            k_pred = vp[keep]
            k_actual = va[keep]
            k_ret = vr[keep]
            k_filt = (np.abs(k_ret) > MIN_RETURN_THRESHOLD) & (k_pred != 0) & (k_actual != 0)
            sel_econ = float((k_pred[k_filt] == k_actual[k_filt]).mean()) if k_filt.sum() > 0 else 0.5

            # Transition accuracy on kept predictions
            k_transitions = np.where(np.diff(np.sign(k_actual)) != 0)[0] + 1
            sel_trans = float(
                np.mean(k_pred[k_transitions] == k_actual[k_transitions]),
            ) if len(k_transitions) > 0 else 0.5

            # Continuation rate in kept predictions
            k_prev = np.zeros_like(k_actual)
            k_prev[1:] = k_actual[:-1]
            k_usable = (k_prev != 0) & (k_pred != 0) & (k_actual != 0)
            k_cont = float(
                ((k_pred == k_prev) & k_usable).sum() / k_usable.sum(),
            ) if k_usable.sum() > 0 else 0.0

            selective[str(cov)] = {
                "coverage": actual_cov,
                "econ_dir": sel_econ,
                "transition": sel_trans,
                "cont_rate": k_cont,
                "n_samples": int(keep.sum()),
                "n_transitions": int(len(k_transitions)),
            }

    return {
        "econ_dir": econ_dir,
        "sharpe_costs": sharpe,
        "transition_accuracy": trans_acc,
        "n_transitions": n_transitions,
        "prediction_distribution": {
            "up": n_up, "flat": n_flat, "down": n_down, "total": n_total,
            "up_pct": n_up / n_total if n_total > 0 else 0.0,
            "flat_pct": n_flat / n_total if n_total > 0 else 0.0,
            "down_pct": n_down / n_total if n_total > 0 else 0.0,
        },
        "continuation_bias": {
            "pred_continuation_rate": cont_rate,
            "actual_continuation_rate": actual_cont_rate,
            "n_continuation_preds": n_cont_pred,
            "n_reversal_preds": n_rev_pred,
            "n_usable": n_usable,
        },
        "regime_accuracy": {
            "acc_on_continuations": acc_on_continuations,
            "acc_on_reversals": acc_on_reversals,
        },
        "selective": selective,
    }


def run_validation(n_seeds: int = 3) -> dict:
    """Run holdout validation with full diagnostics.

    H1 = 2025-H1 (Jan-Jun, same as experiment test period)
    H2 = 2025-H2 (Jul-Dec, true holdout — never seen)
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info("Holdout validation + prediction distribution diagnostic")

    # Load data
    train_ohlcv, test_h1_ohlcv, test_h2_ohlcv = load_and_split_ohlcv()
    tickers = sorted(
        set(train_ohlcv) & set(test_h1_ohlcv) & set(test_h2_ohlcv),
    )
    logger.info("Tickers with all 3 splits: %d", len(tickers))

    # Filter to common tickers
    train_ohlcv = {t: train_ohlcv[t] for t in tickers}
    test_h1_ohlcv = {t: test_h1_ohlcv[t] for t in tickers}
    test_h2_ohlcv = {t: test_h2_ohlcv[t] for t in tickers}

    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_h1_prices = _ohlcv_to_timeseries(test_h1_ohlcv)
    test_h2_prices = _ohlcv_to_timeseries(test_h2_ohlcv)

    # Build datasets
    logger.info("Building datasets...")
    X_train, tr_rets, tr_valid, tr_lvls, _, _ = _build_d1_pipeline(
        train_prices, None, tickers, None, 0,
    )
    X_test_h1, te_h1_rets, te_h1_valid, te_h1_lvls, _, _ = _build_d1_pipeline(
        test_h1_prices, None, tickers, None, 0,
    )
    X_test_h2, te_h2_rets, te_h2_valid, te_h2_lvls, _, _ = _build_d1_pipeline(
        test_h2_prices, None, tickers, None, 0,
    )

    boundaries = compute_quantile_boundaries(
        tr_rets, tr_lvls, tr_valid, PERCENTILES, per_level=True,
    )
    y_train = assign_quantile_labels(tr_rets, tr_lvls, boundaries)

    valid_labels = y_train[tr_valid]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    y_tr = y_train.astype(np.float64)

    logger.info(
        "Train: %d, Test-H1: %d, Test-H2: %d windows",
        len(X_train), len(X_test_h1), len(X_test_h2),
    )

    configs = {
        "baseline": {"loss_type": "ce", "loss_kwargs": {}},
        "hetero": {"loss_type": "heteroscedastic", "loss_kwargs": {}},
        "hetero_sn": {
            "loss_type": "heteroscedastic_selectivenet",
            "loss_kwargs": {"target_coverage": 0.7, "lam": 32.0},
        },
    }

    all_results: dict[str, dict] = {}

    for cfg_key, cfg in configs.items():
        logger.info("=== Config: %s ===", cfg_key)
        results_h1: list[dict] = []
        results_h2: list[dict] = []

        for seed in range(n_seeds):
            logger.info("  Seed %d/%d", seed + 1, n_seeds)

            np.random.seed(seed * 42 + 7)
            torch.manual_seed(seed * 42 + 7)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed * 42 + 7)

            model_kwargs = dict(MODEL_KWARGS)
            if cfg["loss_type"] != "ce":
                model_kwargs["loss_type"] = cfg["loss_type"]
                model_kwargs["loss_kwargs"] = cfg["loss_kwargs"]

            model = WaveletGPT(
                vocab_size=1, context_length=CONTEXT_LENGTH,
                task="return_quantile", n_output_classes=N_CLASSES,
                class_weights=class_weights, input_mode="continuous",
                n_aux_features=N_AUX_FEATURES, **model_kwargs,
            )
            t0 = time.time()
            model.fit(X_train, y_tr)
            train_time = time.time() - t0
            logger.info("    Trained in %.1fs", train_time)

            has_variance = cfg["loss_type"] != "ce"

            # Test on H1 (2025 Jan-Jun, same as experiments)
            pred_h1 = model.predict(X_test_h1).astype(np.int64)
            var_h1 = (
                model.predict_variance(X_test_h1) if has_variance else None
            )
            diag_h1 = analyze_predictions(
                pred_h1, te_h1_rets, te_h1_valid, var_h1,
            )
            results_h1.append(diag_h1)
            logger.info(
                "    H1: econ=%.1f%% trans=%.1f%% cont_rate=%.1f%%",
                diag_h1["econ_dir"] * 100,
                diag_h1["transition_accuracy"] * 100,
                diag_h1["continuation_bias"]["pred_continuation_rate"] * 100,
            )

            # Test on H2 (2025 Jul-Dec, holdout)
            pred_h2 = model.predict(X_test_h2).astype(np.int64)
            var_h2 = (
                model.predict_variance(X_test_h2) if has_variance else None
            )
            diag_h2 = analyze_predictions(
                pred_h2, te_h2_rets, te_h2_valid, var_h2,
            )
            results_h2.append(diag_h2)
            logger.info(
                "    H2: econ=%.1f%% trans=%.1f%% cont_rate=%.1f%%",
                diag_h2["econ_dir"] * 100,
                diag_h2["transition_accuracy"] * 100,
                diag_h2["continuation_bias"]["pred_continuation_rate"] * 100,
            )

        all_results[cfg_key] = {
            "results_h1": results_h1,
            "results_h2": results_h2,
        }

    # Print diagnostic report
    w = 115
    print()
    print("=" * w)
    print("HOLDOUT VALIDATION + PREDICTION DISTRIBUTION DIAGNOSTIC")
    print("=" * w)

    for period, period_key in [
        ("2025-H1 (Jan-Jun, experiment period)", "results_h1"),
        ("2025-H2 (Jul-Dec, HOLDOUT)", "results_h2"),
    ]:
        print(f"\n{'=' * w}")
        print(f"  {period} TEST SET")
        print(f"{'=' * w}")

        print(f"\n{'Config':<20} {'Econ Dir':>10} {'Sharpe':>8} {'Trans':>8} "
              f"{'Cont Rate':>10} {'Actual CR':>10} "
              f"{'Acc@Cont':>10} {'Acc@Rev':>10} "
              f"{'Up%':>6} {'Flat%':>6} {'Down%':>6}")
        print("-" * w)

        for cfg_key in configs:
            results = all_results[cfg_key][period_key]
            econ = np.mean([r["econ_dir"] for r in results])
            sharpe = np.mean([r["sharpe_costs"] for r in results])
            trans = np.mean([r["transition_accuracy"] for r in results])
            cont_rate = np.mean([
                r["continuation_bias"]["pred_continuation_rate"]
                for r in results
            ])
            actual_cr = np.mean([
                r["continuation_bias"]["actual_continuation_rate"]
                for r in results
            ])
            acc_cont = np.mean([
                r["regime_accuracy"]["acc_on_continuations"]
                for r in results
            ])
            acc_rev = np.mean([
                r["regime_accuracy"]["acc_on_reversals"]
                for r in results
            ])
            up_pct = np.mean([
                r["prediction_distribution"]["up_pct"] for r in results
            ])
            flat_pct = np.mean([
                r["prediction_distribution"]["flat_pct"] for r in results
            ])
            down_pct = np.mean([
                r["prediction_distribution"]["down_pct"] for r in results
            ])

            print(
                f"{cfg_key:<20} {econ:>9.1%} {sharpe:>+7.3f} {trans:>7.1%} "
                f"{cont_rate:>9.1%} {actual_cr:>9.1%} "
                f"{acc_cont:>9.1%} {acc_rev:>9.1%} "
                f"{up_pct:>5.1%} {flat_pct:>5.1%} {down_pct:>5.1%}",
            )

        # Selective metrics with transition + continuation
        print("\n  Selective metrics (variance-based filtering):")
        print(f"  {'Config':<20} {'Cov':>6} {'Econ Dir':>10} "
              f"{'Trans':>8} {'Cont Rate':>10}")
        print(f"  {'-' * 60}")
        for cfg_key in ["hetero", "hetero_sn"]:
            results = all_results[cfg_key][period_key]
            for cov in COVERAGE_TARGETS:
                cov_s = str(cov)
                sel_econ = np.mean([
                    r["selective"][cov_s]["econ_dir"] for r in results
                ])
                sel_trans = np.mean([
                    r["selective"][cov_s]["transition"] for r in results
                ])
                sel_cont = np.mean([
                    r["selective"][cov_s]["cont_rate"] for r in results
                ])
                print(
                    f"  {cfg_key:<20} {cov:>5.0%} {sel_econ:>9.1%} "
                    f"{sel_trans:>7.1%} {sel_cont:>9.1%}",
                )
            print()

    print("=" * w)

    # Save
    results_dir = Path.home() / ".wavecast" / "audit" / "feature_tests"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "2026_validation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    return all_results


if __name__ == "__main__":
    run_validation()
