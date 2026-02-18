#!/usr/bin/env python3
"""Three diagnostics to determine if any real selective prediction signal exists.

1. Platt scaling prediction distribution — does the 70.4%@37.7% come from
   balanced up/down/flat or from flat-selecting?
2. CE softmax entropy as abstention signal — low/med/high entropy thirds
3. CE error analysis — return magnitude when the model is wrong

Usage:
    python -m scripts.feature_tests.test_three_diagnostics
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
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
from sklearn.isotonic import IsotonicRegression

from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
)

logger = logging.getLogger(__name__)

MID = N_CLASSES // 2


def pred_distribution(preds: np.ndarray) -> dict:
    """Count up/flat/down predictions."""
    up = int((preds > MID).sum())
    flat = int((preds == MID).sum())
    down = int((preds < MID).sum())
    total = len(preds)
    return {
        "up": up, "flat": flat, "down": down, "total": total,
        "up_pct": up / total if total else 0,
        "flat_pct": flat / total if total else 0,
        "down_pct": down / total if total else 0,
    }


def econ_dir_on_subset(
    preds: np.ndarray, returns: np.ndarray, valid: np.ndarray,
) -> dict:
    """Compute econ dir, transition acc, and pred distribution on a subset."""
    v = valid.astype(bool)
    if v.sum() == 0:
        return {
            "econ_dir": 0.5, "transition": 0.5, "n_valid": 0,
            "distribution": pred_distribution(np.array([])),
        }

    vp = preds[v]
    vr = returns[v]

    # Prediction distribution
    dist = pred_distribution(vp)

    # Econ dir (only non-flat preds, only returns > threshold)
    pred_dir = np.where(vp > MID, 1, np.where(vp < MID, -1, 0))
    actual_dir = np.sign(vr)
    filt = (np.abs(vr) > MIN_RETURN_THRESHOLD) & (pred_dir != 0) & (actual_dir != 0)
    econ = float(np.mean(pred_dir[filt] == actual_dir[filt])) if filt.sum() > 0 else 0.5
    n_directional = int(filt.sum())

    # Transition accuracy
    if v.sum() > 10:
        ad_seq = np.sign(vr)
        transitions = np.where(np.diff(ad_seq) != 0)[0] + 1
        if len(transitions) > 0:
            trans = float(np.mean(pred_dir[transitions] == ad_seq[transitions]))
            n_trans = len(transitions)
        else:
            trans = 0.5
            n_trans = 0
    else:
        trans = 0.5
        n_trans = 0

    return {
        "econ_dir": econ,
        "n_directional": n_directional,
        "transition": trans,
        "n_transitions": n_trans,
        "n_valid": int(v.sum()),
        "distribution": dist,
    }


def run_diagnostics(n_seeds: int = 3) -> dict:
    """Run all three diagnostics."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info("Three diagnostics, %d seeds", n_seeds)

    # Load data
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

    # 80/20 train/val split (same as calibration experiment)
    n_train = int(len(X_train_full) * 0.8)
    X_train = X_train_full[:n_train]
    X_val = X_train_full[n_train:]
    val_rets = tr_rets[n_train:]
    val_valid = tr_valid[n_train:]
    val_lvls = tr_lvls[n_train:]

    boundaries = compute_quantile_boundaries(
        tr_rets[:n_train], tr_lvls[:n_train], tr_valid[:n_train],
        PERCENTILES, per_level=True,
    )
    y_train = assign_quantile_labels(
        tr_rets[:n_train], tr_lvls[:n_train], boundaries,
    )
    y_val = assign_quantile_labels(val_rets, val_lvls, boundaries)

    valid_labels = y_train[tr_valid[:n_train]]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    logger.info(
        "Train: %d, Val: %d, Test: %d, Tickers: %d",
        n_train, len(X_val), len(X_test), len(tickers),
    )

    # Accumulators across seeds
    platt_results: list[dict] = []
    entropy_results: list[dict] = []
    error_results: list[dict] = []

    for seed in range(n_seeds):
        logger.info("=== Seed %d/%d ===", seed + 1, n_seeds)
        np.random.seed(seed * 42 + 7)
        torch.manual_seed(seed * 42 + 7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 42 + 7)

        # Train CE baseline
        model = WaveletGPT(
            vocab_size=1, context_length=CONTEXT_LENGTH,
            task="return_quantile", n_output_classes=N_CLASSES,
            class_weights=class_weights, input_mode="continuous",
            n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
        )
        t0 = time.time()
        model.fit(X_train, y_train.astype(np.float64))
        logger.info("  CE trained in %.1fs", time.time() - t0)

        # Get probabilities on val + test
        val_proba = model.predict_proba(X_val)
        test_proba = model.predict_proba(X_test)
        test_preds = np.argmax(test_proba, axis=1).astype(np.int64)
        val_preds = np.argmax(val_proba, axis=1).astype(np.int64)

        # =========================================================
        # DIAGNOSTIC 1: Platt scaling prediction distribution
        # =========================================================
        logger.info("  D1: Platt scaling prediction distribution...")

        # Fit Platt on val
        val_correct = (val_preds == y_val).astype(np.float64)
        val_max_conf = np.max(val_proba, axis=1)

        _platt_conf = val_max_conf[val_valid]
        _platt_correct = val_correct[val_valid]

        def platt_loss(
            params: np.ndarray,
            _conf: np.ndarray = _platt_conf,
            _corr: np.ndarray = _platt_correct,
        ) -> float:
            a, b = params
            p = 1.0 / (1.0 + np.exp(-(a * _conf + b)))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -float(
                np.mean(_corr * np.log(p) + (1 - _corr) * np.log(1 - p)),
            )

        res = minimize(platt_loss, x0=[1.0, 0.0], method="Nelder-Mead")
        a_platt, b_platt = float(res.x[0]), float(res.x[1])
        logger.info("    Platt: a=%.3f, b=%.3f", a_platt, b_platt)

        # Also fit isotonic
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_max_conf[val_valid], val_correct[val_valid])

        # Apply to test
        test_max_conf = np.max(test_proba, axis=1)
        platt_conf = 1.0 / (1.0 + np.exp(-(a_platt * test_max_conf + b_platt)))
        iso_conf = iso.predict(test_max_conf)

        # Evaluate at the magic threshold (0.3) — this is what produced 70.4%@37.7%
        platt_seed: dict = {}
        for method_name, conf in [("platt", platt_conf), ("isotonic", iso_conf),
                                   ("uncalibrated", test_max_conf)]:
            method_results: dict = {}
            for thresh in [0.0, 0.3, 0.5]:
                selected = te_valid & (conf >= thresh)
                coverage = float(selected.sum()) / max(te_valid.sum(), 1)
                metrics = econ_dir_on_subset(test_preds, te_rets, selected)
                metrics["coverage"] = coverage
                method_results[str(thresh)] = metrics
            platt_seed[method_name] = method_results

        platt_results.append(platt_seed)

        # =========================================================
        # DIAGNOSTIC 2: CE softmax entropy binning
        # =========================================================
        logger.info("  D2: Softmax entropy binning...")

        # Compute entropy of each prediction's softmax distribution
        entropy = -np.sum(
            test_proba * np.log(np.clip(test_proba, 1e-10, 1.0)), axis=1,
        )

        # Only on valid samples
        valid_idx = np.where(te_valid)[0]
        valid_entropy = entropy[valid_idx]

        # Bin into thirds
        t33 = np.percentile(valid_entropy, 33.33)
        t67 = np.percentile(valid_entropy, 66.67)

        bins = {
            "low": valid_idx[valid_entropy <= t33],
            "medium": valid_idx[(valid_entropy > t33) & (valid_entropy <= t67)],
            "high": valid_idx[valid_entropy > t67],
        }

        entropy_seed: dict = {
            "thresholds": {"t33": float(t33), "t67": float(t67)},
            "max_entropy": float(np.log(N_CLASSES)),
        }
        for bin_name, idx in bins.items():
            mask = np.zeros(len(test_preds), dtype=bool)
            mask[idx] = True
            metrics = econ_dir_on_subset(test_preds, te_rets, mask)
            metrics["avg_entropy"] = float(entropy[idx].mean())
            metrics["n_samples"] = len(idx)
            entropy_seed[bin_name] = metrics

        # Also do a finer 10-bin analysis
        decile_edges = np.percentile(valid_entropy, np.arange(0, 101, 10))
        fine_bins: list[dict] = []
        for d in range(10):
            lo = decile_edges[d]
            hi = decile_edges[d + 1] if d < 9 else valid_entropy.max() + 1
            if d == 0:
                idx = valid_idx[valid_entropy <= hi]
            elif d == 9:
                idx = valid_idx[valid_entropy > lo]
            else:
                idx = valid_idx[(valid_entropy > lo) & (valid_entropy <= hi)]
            mask = np.zeros(len(test_preds), dtype=bool)
            mask[idx] = True
            metrics = econ_dir_on_subset(test_preds, te_rets, mask)
            metrics["avg_entropy"] = float(entropy[idx].mean()) if len(idx) > 0 else 0
            metrics["n_samples"] = len(idx)
            metrics["decile"] = d
            fine_bins.append(metrics)
        entropy_seed["deciles"] = fine_bins

        entropy_results.append(entropy_seed)

        # =========================================================
        # DIAGNOSTIC 3: Error analysis — return magnitude
        # =========================================================
        logger.info("  D3: Error analysis...")

        pred_dir = np.where(test_preds > MID, 1, np.where(test_preds < MID, -1, 0))
        actual_dir = np.sign(te_rets)

        # Only valid, non-flat predictions, returns above threshold
        usable = (
            te_valid
            & (pred_dir != 0)
            & (actual_dir != 0)
            & (np.abs(te_rets) > MIN_RETURN_THRESHOLD)
        )
        correct_mask = usable & (pred_dir == actual_dir)
        wrong_mask = usable & (pred_dir != actual_dir)

        correct_returns = np.abs(te_rets[correct_mask])
        wrong_returns = np.abs(te_rets[wrong_mask])

        # Percentile analysis
        pctiles = [10, 25, 50, 75, 90, 95, 99]

        error_seed: dict = {
            "n_correct": int(correct_mask.sum()),
            "n_wrong": int(wrong_mask.sum()),
            "accuracy": float(correct_mask.sum() / usable.sum()) if usable.sum() > 0 else 0,
        }

        if len(correct_returns) > 0:
            error_seed["correct_returns"] = {
                "mean": float(correct_returns.mean()),
                "median": float(np.median(correct_returns)),
                "std": float(correct_returns.std()),
                "percentiles": {
                    str(p): float(np.percentile(correct_returns, p))
                    for p in pctiles
                },
            }
        if len(wrong_returns) > 0:
            error_seed["wrong_returns"] = {
                "mean": float(wrong_returns.mean()),
                "median": float(np.median(wrong_returns)),
                "std": float(wrong_returns.std()),
                "percentiles": {
                    str(p): float(np.percentile(wrong_returns, p))
                    for p in pctiles
                },
            }

        # Bin returns into small/medium/large moves and check accuracy in each
        all_abs_rets = np.abs(te_rets[usable])
        ret_t33 = np.percentile(all_abs_rets, 33.33)
        ret_t67 = np.percentile(all_abs_rets, 66.67)

        for bin_name, lo, hi in [
            ("small", 0, ret_t33),
            ("medium", ret_t33, ret_t67),
            ("large", ret_t67, np.inf),
        ]:
            if bin_name == "small":
                bin_mask = usable & (np.abs(te_rets) <= hi)
            elif bin_name == "large":
                bin_mask = usable & (np.abs(te_rets) > lo)
            else:
                bin_mask = usable & (np.abs(te_rets) > lo) & (np.abs(te_rets) <= hi)

            n_bin = int(bin_mask.sum())
            if n_bin > 0:
                bin_correct = int((bin_mask & (pred_dir == actual_dir)).sum())
                bin_acc = bin_correct / n_bin
                error_seed[f"accuracy_{bin_name}_moves"] = {
                    "accuracy": float(bin_acc),
                    "n_samples": n_bin,
                    "return_range": [float(lo), float(hi)],
                    "mean_abs_return": float(np.abs(te_rets[bin_mask]).mean()),
                }

        # Also check: when wrong on large moves, what's the PnL impact?
        large_wrong = wrong_mask & (np.abs(te_rets) > ret_t67)
        large_correct = correct_mask & (np.abs(te_rets) > ret_t67)
        if large_wrong.sum() > 0:
            error_seed["large_move_pnl"] = {
                "n_wrong_large": int(large_wrong.sum()),
                "n_correct_large": int(large_correct.sum()),
                "avg_loss_on_wrong_large": float(np.abs(te_rets[large_wrong]).mean()),
                "avg_gain_on_correct_large": float(
                    np.abs(te_rets[large_correct]).mean(),
                ) if large_correct.sum() > 0 else 0,
                "net_pnl_large": float(
                    np.sum(np.abs(te_rets[large_correct]))
                    - np.sum(np.abs(te_rets[large_wrong])),
                ),
            }

        error_results.append(error_seed)

    # =========================================================
    # AGGREGATE AND PRINT
    # =========================================================

    w = 110
    print()
    print("=" * w)
    print("DIAGNOSTIC 1: Platt Scaling Prediction Distribution")
    print("=" * w)

    for method in ["uncalibrated", "platt", "isotonic"]:
        print(f"\n  {method.upper()}")
        print(f"  {'Thresh':>8} {'Cov':>8} {'Econ':>8} {'Trans':>8}"
              f"   {'Up%':>8} {'Flat%':>8} {'Down%':>8} {'#Dir':>8}")
        print("  " + "-" * 90)
        for thresh in ["0.0", "0.3", "0.5"]:
            covs = [r[method][thresh]["coverage"] for r in platt_results]
            econs = [r[method][thresh]["econ_dir"] for r in platt_results]
            trans = [r[method][thresh]["transition"] for r in platt_results]
            ups = [r[method][thresh]["distribution"]["up_pct"] for r in platt_results]
            flats = [r[method][thresh]["distribution"]["flat_pct"]
                     for r in platt_results]
            downs = [r[method][thresh]["distribution"]["down_pct"]
                     for r in platt_results]
            n_dirs = [r[method][thresh].get("n_directional", 0)
                      for r in platt_results]

            print(
                f"  {thresh:>8} {np.mean(covs):>7.1%} {np.mean(econs):>7.1%}"
                f" {np.mean(trans):>7.1%}   {np.mean(ups):>7.1%}"
                f" {np.mean(flats):>7.1%} {np.mean(downs):>7.1%}"
                f" {np.mean(n_dirs):>7.0f}",
            )

    print()
    print("=" * w)
    print("DIAGNOSTIC 2: CE Softmax Entropy Binning")
    print("=" * w)

    print(f"\n  {'Bin':>10} {'AvgH':>8} {'Econ':>8} {'Trans':>8}"
          f"   {'Up%':>8} {'Flat%':>8} {'Down%':>8} {'#Dir':>8} {'N':>8}")
    print("  " + "-" * 95)

    for bin_name in ["low", "medium", "high"]:
        econs = [r[bin_name]["econ_dir"] for r in entropy_results]
        trans = [r[bin_name]["transition"] for r in entropy_results]
        ups = [r[bin_name]["distribution"]["up_pct"] for r in entropy_results]
        flats = [r[bin_name]["distribution"]["flat_pct"] for r in entropy_results]
        downs = [r[bin_name]["distribution"]["down_pct"] for r in entropy_results]
        n_dirs = [r[bin_name].get("n_directional", 0) for r in entropy_results]
        ns = [r[bin_name]["n_samples"] for r in entropy_results]
        avgh = [r[bin_name]["avg_entropy"] for r in entropy_results]

        print(
            f"  {bin_name:>10} {np.mean(avgh):>7.3f} {np.mean(econs):>7.1%}"
            f" {np.mean(trans):>7.1%}   {np.mean(ups):>7.1%}"
            f" {np.mean(flats):>7.1%} {np.mean(downs):>7.1%}"
            f" {np.mean(n_dirs):>7.0f} {np.mean(ns):>7.0f}",
        )

    print(f"\n  Max possible entropy: {np.log(N_CLASSES):.3f}")

    # Print decile view
    print(f"\n  {'Decile':>8} {'AvgH':>8} {'Econ':>8} {'Trans':>8}"
          f"   {'Flat%':>8} {'#Dir':>8}")
    print("  " + "-" * 60)
    for d in range(10):
        econs = [r["deciles"][d]["econ_dir"] for r in entropy_results]
        trans = [r["deciles"][d]["transition"] for r in entropy_results]
        flats = [r["deciles"][d]["distribution"]["flat_pct"]
                 for r in entropy_results]
        n_dirs = [r["deciles"][d].get("n_directional", 0) for r in entropy_results]
        avgh = [r["deciles"][d]["avg_entropy"] for r in entropy_results]
        print(
            f"  D{d:>7} {np.mean(avgh):>7.3f} {np.mean(econs):>7.1%}"
            f" {np.mean(trans):>7.1%}   {np.mean(flats):>7.1%}"
            f" {np.mean(n_dirs):>7.0f}",
        )

    print()
    print("=" * w)
    print("DIAGNOSTIC 3: Error Analysis — Return Magnitude")
    print("=" * w)

    avg_n_correct = np.mean([r["n_correct"] for r in error_results])
    avg_n_wrong = np.mean([r["n_wrong"] for r in error_results])
    avg_acc = np.mean([r["accuracy"] for r in error_results])
    print(f"\n  Overall: {avg_acc:.1%} accuracy on {avg_n_correct + avg_n_wrong:.0f}"
          f" directional predictions ({avg_n_correct:.0f} correct, {avg_n_wrong:.0f} wrong)")

    print(f"\n  {'':>12} {'Correct':>12} {'Wrong':>12} {'Delta':>12}")
    print("  " + "-" * 55)
    for stat in ["mean", "median"]:
        c_vals = [r["correct_returns"][stat] for r in error_results]
        w_vals = [r["wrong_returns"][stat] for r in error_results]
        cm = np.mean(c_vals)
        wm = np.mean(w_vals)
        print(f"  {stat:>12} {cm:>11.5f} {wm:>11.5f} {cm - wm:>+11.5f}")

    print("\n  Return percentiles (avg across seeds):")
    print(f"  {'Pctile':>8} {'Correct':>12} {'Wrong':>12}")
    print("  " + "-" * 35)
    for p in ["10", "25", "50", "75", "90", "95"]:
        c_vals = [r["correct_returns"]["percentiles"][p] for r in error_results]
        w_vals = [r["wrong_returns"]["percentiles"][p] for r in error_results]
        print(f"  P{p:>7} {np.mean(c_vals):>11.5f} {np.mean(w_vals):>11.5f}")

    print("\n  Accuracy by move size:")
    print(f"  {'Bin':>10} {'Acc':>8} {'N':>8} {'AvgRet':>10}")
    print("  " + "-" * 40)
    for bin_name in ["small", "medium", "large"]:
        key = f"accuracy_{bin_name}_moves"
        accs = [r[key]["accuracy"] for r in error_results]
        ns = [r[key]["n_samples"] for r in error_results]
        avgs = [r[key]["mean_abs_return"] for r in error_results]
        print(
            f"  {bin_name:>10} {np.mean(accs):>7.1%} {np.mean(ns):>7.0f}"
            f" {np.mean(avgs):>9.5f}",
        )

    # Large move PnL
    if "large_move_pnl" in error_results[0]:
        print("\n  Large move PnL analysis:")
        n_cL = np.mean([r["large_move_pnl"]["n_correct_large"]
                        for r in error_results])
        n_wL = np.mean([r["large_move_pnl"]["n_wrong_large"]
                        for r in error_results])
        avg_gain = np.mean([r["large_move_pnl"]["avg_gain_on_correct_large"]
                            for r in error_results])
        avg_loss = np.mean([r["large_move_pnl"]["avg_loss_on_wrong_large"]
                            for r in error_results])
        net = np.mean([r["large_move_pnl"]["net_pnl_large"]
                       for r in error_results])
        print(f"    Correct on {n_cL:.0f} large moves (avg gain: {avg_gain:.5f})")
        print(f"    Wrong on {n_wL:.0f} large moves (avg loss: {avg_loss:.5f})")
        print(f"    Net PnL on large moves: {net:+.4f}")

    print()
    print("=" * w)

    # Save all results
    output = {
        "n_seeds": n_seeds,
        "diagnostic_1_platt": platt_results,
        "diagnostic_2_entropy": entropy_results,
        "diagnostic_3_errors": error_results,
    }
    results_dir = Path.home() / ".wavecast" / "audit" / "feature_tests"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "three_diagnostics_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)

    return output


if __name__ == "__main__":
    run_diagnostics()
