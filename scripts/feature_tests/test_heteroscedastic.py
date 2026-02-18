#!/usr/bin/env python3
"""Experiment 6: Faithful Heteroscedastic Regression.

Source: Stirn et al. AISTATS 2023 "Faithful Heteroscedastic Regression"
Hypothesis: Dual-head model predicts class logits + per-sample variance.
Loss = exp(-log_var) * CE + log_var. Initialize variance head to high
variance to start uncertain. Abstain on high-variance predictions.

Pass criteria: Unconditional econ_dir ≥ baseline. Selective econ_dir > 67%
at coverage > 30%.

Usage:
    python -m scripts.feature_tests.test_heteroscedastic
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


def run_heteroscedastic_test(n_seeds: int = 3) -> dict:
    """Compare CE vs heteroscedastic loss with variance-based selective trading."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger.info("Heteroscedastic loss experiment, %d seeds", n_seeds)

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

    baseline_results = []
    hetero_results = []

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
        base_eval = _evaluate(
            pred_base, te_rets, y_test, te_valid,
            metrics["train_loss"], base_time,
        )
        baseline_results.append(base_eval)
        logger.info(
            "  Baseline: econ=%.1f%% sharpe=%.3f",
            base_eval.econ_dir_accuracy * 100, base_eval.sharpe_with_costs,
        )

        # Heteroscedastic
        np.random.seed(seed * 42 + 7)
        torch.manual_seed(seed * 42 + 7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 42 + 7)

        model_het = WaveletGPT(
            vocab_size=1, context_length=CONTEXT_LENGTH,
            task="return_quantile", n_output_classes=N_CLASSES,
            class_weights=class_weights, input_mode="continuous",
            n_aux_features=N_AUX_FEATURES,
            loss_type="heteroscedastic",
            **MODEL_KWARGS,
        )
        t0 = time.time()
        metrics = model_het.fit(X_train, y_tr)
        het_time = time.time() - t0
        pred_het = model_het.predict(X_test).astype(np.int64)
        het_eval = _evaluate(
            pred_het, te_rets, y_test, te_valid,
            metrics["train_loss"], het_time,
        )

        # Get variance for selective trading
        variance = model_het.predict_variance(X_test)
        valid_mask = te_valid.astype(bool)
        valid_preds = pred_het[valid_mask]
        valid_rets = te_rets[valid_mask]
        valid_var = variance[valid_mask]

        selective_results = {}
        for coverage_target in [0.3, 0.5, 0.7, 0.9]:
            threshold = np.quantile(valid_var, coverage_target)
            keep = valid_var <= threshold
            actual_coverage = float(keep.mean())
            if keep.sum() > 0:
                kept_preds = valid_preds[keep]
                kept_rets = valid_rets[keep]
                mid = N_CLASSES // 2
                pred_dir = np.where(
                    kept_preds > mid, 1, np.where(kept_preds < mid, -1, 0)
                )
                actual_dir = np.sign(kept_rets)
                directional = (pred_dir != 0) & (actual_dir != 0)
                if directional.sum() > 0:
                    sel_econ = float(
                        (pred_dir[directional] == actual_dir[directional]).mean()
                    )
                else:
                    sel_econ = 0.0
                selective_results[str(coverage_target)] = {
                    "actual_coverage": actual_coverage,
                    "selective_econ_dir": sel_econ,
                    "n_samples": int(keep.sum()),
                }

        hetero_results.append({
            "eval": het_eval,
            "variance_mean": float(valid_var.mean()),
            "variance_std": float(valid_var.std()),
            "selective": selective_results,
        })
        logger.info(
            "  Heteroscedastic: econ=%.1f%% sharpe=%.3f var_mean=%.3f",
            het_eval.econ_dir_accuracy * 100,
            het_eval.sharpe_with_costs, valid_var.mean(),
        )

    # Aggregate
    base_econ = np.mean([r.econ_dir_accuracy for r in baseline_results])
    base_sharpe = np.mean([r.sharpe_with_costs for r in baseline_results])
    base_trans = np.mean([r.transition_accuracy for r in baseline_results])

    het_evals = [r["eval"] for r in hetero_results]
    het_econ = np.mean([e.econ_dir_accuracy for e in het_evals])
    het_sharpe = np.mean([e.sharpe_with_costs for e in het_evals])
    het_trans = np.mean([e.transition_accuracy for e in het_evals])

    # Check selective performance
    best_sel_econ = 0.0
    best_coverage = 0.0
    sel_summary = {}
    for cov_key in ["0.3", "0.5", "0.7", "0.9"]:
        sel_econs = []
        sel_coverages = []
        for r in hetero_results:
            if cov_key in r["selective"]:
                sel_econs.append(r["selective"][cov_key]["selective_econ_dir"])
                sel_coverages.append(r["selective"][cov_key]["actual_coverage"])
        if sel_econs:
            avg_sel_econ = float(np.mean(sel_econs))
            avg_coverage = float(np.mean(sel_coverages))
            sel_summary[cov_key] = {
                "avg_selective_econ_dir": avg_sel_econ,
                "avg_coverage": avg_coverage,
            }
            if avg_coverage > 0.30 and avg_sel_econ > best_sel_econ:
                best_sel_econ = avg_sel_econ
                best_coverage = avg_coverage

    d_econ = het_econ - base_econ
    uncond_ok = d_econ >= -0.01
    sel_ok = best_sel_econ > 0.67
    verdict = "PASS" if uncond_ok and sel_ok else "FAIL"

    w = 100
    print()
    print("=" * w)
    print("EXPERIMENT: Faithful Heteroscedastic Loss")
    print("=" * w)
    print(f"\n{'Metric':<20} {'Baseline':>10} {'Heteroscedastic':>16} {'Delta':>10}")
    print("-" * w)
    print(f"{'Econ Dir Acc':<20} {base_econ:>9.1%} {het_econ:>15.1%} {d_econ:>+9.1%}")
    print(f"{'Sharpe (+costs)':<20} {base_sharpe:>+9.3f} {het_sharpe:>+15.3f} {het_sharpe - base_sharpe:>+9.3f}")
    print(f"{'Transition Acc':<20} {base_trans:>9.1%} {het_trans:>15.1%} {het_trans - base_trans:>+9.1%}")
    print("-" * w)
    if best_sel_econ > 0:
        print(f"\nBest selective: econ_dir={best_sel_econ:.1%} at coverage={best_coverage:.1%}")
    for cov_key, s in sel_summary.items():
        print(f"  Coverage {cov_key}: econ_dir={s['avg_selective_econ_dir']:.1%} (actual={s['avg_coverage']:.1%})")
    print(f"\nVERDICT: {verdict}")
    print("=" * w)

    results_dir = Path.home() / ".wavecast" / "audit" / "feature_tests"
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "name": "heteroscedastic",
        "experiment_type": "loss_function",
        "n_seeds": n_seeds,
        "verdict": verdict,
        "baseline_mean": {
            "econ_dir": float(base_econ),
            "sharpe_costs": float(base_sharpe),
            "transition": float(base_trans),
        },
        "heteroscedastic_mean": {
            "econ_dir": float(het_econ),
            "sharpe_costs": float(het_sharpe),
            "transition": float(het_trans),
        },
        "delta": {
            "econ_dir": float(d_econ),
            "sharpe_costs": float(het_sharpe - base_sharpe),
        },
        "selective_summary": sel_summary,
        "best_selective_econ": best_sel_econ,
        "best_coverage": best_coverage,
    }
    out_path = results_dir / "heteroscedastic_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)
    return output


if __name__ == "__main__":
    run_heteroscedastic_test()
