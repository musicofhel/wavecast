#!/usr/bin/env python3
"""Experiment 11: SelectiveNet.

Source: Geifman & El-Yaniv ICML 2019 "SelectiveNet"
Hypothesis: End-to-end learned rejection head jointly optimizes
classification and abstention, producing better risk-coverage tradeoff
than post-hoc thresholding.

Pass criteria: At 70% coverage: selective econ_dir > 67%.
At 50% coverage: selective econ_dir > 70%.

Usage:
    python -m scripts.feature_tests.test_selectivenet
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


def selective_evaluate(
    preds: np.ndarray,
    returns: np.ndarray,
    reject_scores: np.ndarray,
    valid_mask: np.ndarray,
    coverage_target: float,
) -> dict:
    """Evaluate selective predictions at given coverage."""
    valid = valid_mask.astype(bool)
    v_preds = preds[valid]
    v_rets = returns[valid]
    v_reject = reject_scores[valid]

    # Accept top coverage_target fraction by reject score
    threshold = np.quantile(v_reject, 1 - coverage_target)
    keep = v_reject >= threshold
    actual_coverage = float(keep.mean())

    if keep.sum() == 0:
        return {"coverage": actual_coverage, "econ_dir": 0.0, "n_samples": 0}

    mid = N_CLASSES // 2
    pd = np.where(v_preds[keep] > mid, 1, np.where(v_preds[keep] < mid, -1, 0))
    ad = np.sign(v_rets[keep])
    dm = (pd != 0) & (ad != 0)
    econ = float((pd[dm] == ad[dm]).mean()) if dm.sum() > 0 else 0.0

    return {
        "coverage": actual_coverage,
        "econ_dir": econ,
        "n_samples": int(keep.sum()),
    }


def run_selectivenet_test(n_seeds: int = 3) -> dict:
    """Compare CE vs SelectiveNet with different target coverages."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger.info("SelectiveNet experiment, %d seeds", n_seeds)

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

    target_coverages = [0.5, 0.7, 0.9]
    baseline_results = []
    sn_results = {c: [] for c in target_coverages}

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

        for target_cov in target_coverages:
            np.random.seed(seed * 42 + 7)
            torch.manual_seed(seed * 42 + 7)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed * 42 + 7)

            model_sn = WaveletGPT(
                vocab_size=1, context_length=CONTEXT_LENGTH,
                task="return_quantile", n_output_classes=N_CLASSES,
                class_weights=class_weights, input_mode="continuous",
                n_aux_features=N_AUX_FEATURES,
                loss_type="selectivenet",
                loss_kwargs={"target_coverage": target_cov},
                **MODEL_KWARGS,
            )
            t0 = time.time()
            metrics = model_sn.fit(X_train, y_tr)
            sn_time = time.time() - t0
            pred_sn = model_sn.predict(X_test).astype(np.int64)
            reject_scores = model_sn.predict_rejection(X_test)

            sn_eval = _evaluate(
                pred_sn, te_rets, y_test, te_valid,
                metrics["train_loss"], sn_time,
            )

            # Evaluate at different coverage levels
            sel_results = {}
            for eval_cov in [0.3, 0.5, 0.7, 0.9]:
                sel = selective_evaluate(
                    pred_sn, te_rets, reject_scores, te_valid, eval_cov
                )
                sel_results[str(eval_cov)] = sel

            sn_results[target_cov].append({
                "eval": sn_eval,
                "selective": sel_results,
                "mean_reject": float(reject_scores.mean()),
            })
            logger.info(
                "  SN(cov=%.1f): econ=%.1f%% sharpe=%.3f reject_mean=%.3f",
                target_cov, sn_eval.econ_dir_accuracy * 100,
                sn_eval.sharpe_with_costs, reject_scores.mean(),
            )

    # Aggregate
    base_econ = np.mean([r.econ_dir_accuracy for r in baseline_results])
    base_sharpe = np.mean([r.sharpe_with_costs for r in baseline_results])

    cov_summaries = {}
    pass_70 = False
    pass_50 = False

    for tc in target_coverages:
        evals = [r["eval"] for r in sn_results[tc]]
        mean_econ = float(np.mean([e.econ_dir_accuracy for e in evals]))
        mean_sharpe = float(np.mean([e.sharpe_with_costs for e in evals]))

        sel_at_coverages = {}
        for eval_cov in ["0.3", "0.5", "0.7", "0.9"]:
            sel_econs = [r["selective"][eval_cov]["econ_dir"] for r in sn_results[tc]]
            sel_covs = [r["selective"][eval_cov]["coverage"] for r in sn_results[tc]]
            sel_at_coverages[eval_cov] = {
                "econ_dir": float(np.mean(sel_econs)),
                "coverage": float(np.mean(sel_covs)),
            }

        cov_summaries[str(tc)] = {
            "uncond_econ": mean_econ,
            "uncond_sharpe": mean_sharpe,
            "selective": sel_at_coverages,
        }

        # Check pass criteria
        if sel_at_coverages["0.7"]["econ_dir"] > 0.67:
            pass_70 = True
        if sel_at_coverages["0.5"]["econ_dir"] > 0.70:
            pass_50 = True

    verdict = "PASS" if pass_70 and pass_50 else "FAIL"

    w = 100
    print()
    print("=" * w)
    print("EXPERIMENT: SelectiveNet (End-to-End Rejection)")
    print("=" * w)
    print(f"\n{'Target Cov':<12} {'Uncond Econ':>12} {'Sel@50%':>10} {'Sel@70%':>10} {'Sel@90%':>10}")
    print("-" * w)
    print(f"{'baseline':<12} {base_econ:>11.1%}")
    for tc in target_coverages:
        s = cov_summaries[str(tc)]
        print(
            f"{tc:<12.1f} {s['uncond_econ']:>11.1%} "
            f"{s['selective']['0.5']['econ_dir']:>9.1%} "
            f"{s['selective']['0.7']['econ_dir']:>9.1%} "
            f"{s['selective']['0.9']['econ_dir']:>9.1%}"
        )
    print("-" * w)
    print(f"\nVERDICT: {verdict}")
    print("=" * w)

    results_dir = Path.home() / ".wavecast" / "audit" / "feature_tests"
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "name": "selectivenet",
        "experiment_type": "selective_prediction",
        "n_seeds": n_seeds,
        "verdict": verdict,
        "baseline_mean": {
            "econ_dir": float(base_econ),
            "sharpe_costs": float(base_sharpe),
        },
        "coverage_summaries": cov_summaries,
        "pass_70_pct": pass_70,
        "pass_50_pct": pass_50,
    }
    out_path = results_dir / "selectivenet_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)
    return output


if __name__ == "__main__":
    run_selectivenet_test()
