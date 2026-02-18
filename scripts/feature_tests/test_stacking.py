#!/usr/bin/env python3
"""Stacking experiment: compare 4 configurations to determine if winners compound.

Configurations:
  A: Heteroscedastic only
  B: Heteroscedastic + label smoothing (eps=0.1)
  C: Heteroscedastic + label smoothing + error regularization (lam=0.1)
  D: Heteroscedastic + SelectiveNet (orthogonal mechanisms)

Key question: does the unconditional econ_dir (100% coverage) improve when
stacking A→B→C? If gains don't compound at 100%, they won't at 50% either.
Config D tests whether the architecturally independent rejection head adds
value on top of the heteroscedastic loss.

Usage:
    python -m scripts.feature_tests.test_stacking
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

COVERAGE_TARGETS = [0.3, 0.5, 0.7, 0.9]


def selective_evaluate(
    preds: np.ndarray,
    returns: np.ndarray,
    scores: np.ndarray,
    valid_mask: np.ndarray,
    coverage_target: float,
    higher_is_keep: bool = True,
) -> dict:
    """Evaluate selective predictions at given coverage.

    Args:
        scores: Per-sample score. For variance, lower = more confident (keep).
            For rejection, higher = more likely to accept (keep).
        higher_is_keep: If True, keep top-scoring samples. If False, keep
            bottom-scoring samples (for variance).
    """
    valid = valid_mask.astype(bool)
    v_preds = preds[valid]
    v_rets = returns[valid]
    v_scores = scores[valid]

    if higher_is_keep:
        threshold = np.quantile(v_scores, 1 - coverage_target)
        keep = v_scores >= threshold
    else:
        threshold = np.quantile(v_scores, coverage_target)
        keep = v_scores <= threshold

    actual_coverage = float(keep.mean())

    if keep.sum() == 0:
        return {"coverage": actual_coverage, "econ_dir": 0.0, "n_samples": 0}

    mid = N_CLASSES // 2
    pd_arr = np.where(
        v_preds[keep] > mid, 1, np.where(v_preds[keep] < mid, -1, 0),
    )
    ad = np.sign(v_rets[keep])
    dm = (pd_arr != 0) & (ad != 0)
    econ = float((pd_arr[dm] == ad[dm]).mean()) if dm.sum() > 0 else 0.0

    return {
        "coverage": actual_coverage,
        "econ_dir": econ,
        "n_samples": int(keep.sum()),
    }


def run_stacking_test(n_seeds: int = 3) -> dict:
    """Compare 4 stacking configurations across seeds."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info("Stacking experiment: 4 configs, %d seeds", n_seeds)

    # Load and prepare data
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

    boundaries = compute_quantile_boundaries(
        tr_rets, tr_lvls, tr_valid, PERCENTILES, per_level=True,
    )
    y_train = assign_quantile_labels(tr_rets, tr_lvls, boundaries)
    y_test = assign_quantile_labels(te_rets, te_lvls, boundaries)

    valid_labels = y_train[tr_valid]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    y_tr = y_train.astype(np.float64)

    logger.info(
        "Train: %d windows, Test: %d windows, Tickers: %d",
        len(X_train), len(X_test), len(tickers),
    )

    # Define configurations
    configs = {
        "A_hetero": {
            "loss_type": "heteroscedastic",
            "loss_kwargs": {},
            "label": "Heteroscedastic only",
        },
        "B_hetero_ls": {
            "loss_type": "heteroscedastic",
            "loss_kwargs": {"label_smoothing": 0.1},
            "label": "Heteroscedastic + label smoothing (eps=0.1)",
        },
        "C_hetero_ls_ereg": {
            "loss_type": "heteroscedastic_error_reg",
            "loss_kwargs": {"label_smoothing": 0.1, "lam": 0.1},
            "label": "Heteroscedastic + LS + error reg (lam=0.1)",
        },
        "D_hetero_sn": {
            "loss_type": "heteroscedastic_selectivenet",
            "loss_kwargs": {"target_coverage": 0.7, "lam": 32.0},
            "label": "Heteroscedastic + SelectiveNet",
        },
    }

    # Also run a clean baseline (standard CE)
    all_results: dict[str, list[dict]] = {
        "baseline": [],
    }
    for key in configs:
        all_results[key] = []

    for seed in range(n_seeds):
        logger.info("=== Seed %d/%d ===", seed + 1, n_seeds)

        # Set deterministic seeds
        np.random.seed(seed * 42 + 7)
        torch.manual_seed(seed * 42 + 7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 42 + 7)

        # Baseline (standard CE)
        logger.info("  Training baseline (CE)...")
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
        all_results["baseline"].append({
            "econ_dir": base_eval.econ_dir_accuracy,
            "sharpe_costs": base_eval.sharpe_with_costs,
            "transition": base_eval.transition_accuracy,
        })
        logger.info(
            "  Baseline: econ=%.1f%% sharpe=%.3f trans=%.1f%%",
            base_eval.econ_dir_accuracy * 100,
            base_eval.sharpe_with_costs,
            base_eval.transition_accuracy * 100,
        )

        # Each configuration
        for cfg_key, cfg in configs.items():
            # Reset seeds for fair comparison
            np.random.seed(seed * 42 + 7)
            torch.manual_seed(seed * 42 + 7)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed * 42 + 7)

            logger.info("  Training %s...", cfg["label"])
            model = WaveletGPT(
                vocab_size=1, context_length=CONTEXT_LENGTH,
                task="return_quantile", n_output_classes=N_CLASSES,
                class_weights=class_weights, input_mode="continuous",
                n_aux_features=N_AUX_FEATURES,
                loss_type=cfg["loss_type"],
                loss_kwargs=cfg["loss_kwargs"],
                **MODEL_KWARGS,
            )
            t0 = time.time()
            metrics = model.fit(X_train, y_tr)
            train_time = time.time() - t0
            pred = model.predict(X_test).astype(np.int64)
            eval_result = _evaluate(
                pred, te_rets, y_test, te_valid,
                metrics["train_loss"], train_time,
            )

            result_entry: dict = {
                "econ_dir": eval_result.econ_dir_accuracy,
                "sharpe_costs": eval_result.sharpe_with_costs,
                "transition": eval_result.transition_accuracy,
            }

            # Selective metrics via variance (configs A, B, C have variance head)
            has_variance = cfg["loss_type"] in (
                "heteroscedastic", "heteroscedastic_error_reg",
                "heteroscedastic_selectivenet",
            )
            has_reject = cfg["loss_type"] in (
                "selectivenet", "heteroscedastic_selectivenet",
            )

            if has_variance:
                variance = model.predict_variance(X_test)
                sel_var = {}
                for cov in COVERAGE_TARGETS:
                    sel_var[str(cov)] = selective_evaluate(
                        pred, te_rets, variance, te_valid, cov,
                        higher_is_keep=False,
                    )
                result_entry["selective_variance"] = sel_var

            if has_reject:
                reject_scores = model.predict_rejection(X_test)
                sel_rej = {}
                for cov in COVERAGE_TARGETS:
                    sel_rej[str(cov)] = selective_evaluate(
                        pred, te_rets, reject_scores, te_valid, cov,
                        higher_is_keep=True,
                    )
                result_entry["selective_reject"] = sel_rej

            all_results[cfg_key].append(result_entry)
            logger.info(
                "  %s: econ=%.1f%% sharpe=%.3f trans=%.1f%%",
                cfg_key,
                eval_result.econ_dir_accuracy * 100,
                eval_result.sharpe_with_costs,
                eval_result.transition_accuracy * 100,
            )

    # Aggregate means
    summary: dict = {"n_seeds": n_seeds, "configs": {}}

    for key in ["baseline", *configs.keys()]:
        results = all_results[key]
        mean_econ = float(np.mean([r["econ_dir"] for r in results]))
        mean_sharpe = float(np.mean([r["sharpe_costs"] for r in results]))
        mean_trans = float(np.mean([r["transition"] for r in results]))

        entry: dict = {
            "label": configs[key]["label"] if key != "baseline" else "CE baseline",
            "mean_econ_dir": mean_econ,
            "mean_sharpe_costs": mean_sharpe,
            "mean_transition": mean_trans,
            "per_seed": results,
        }

        # Aggregate selective metrics
        if key != "baseline":
            for sel_key in ("selective_variance", "selective_reject"):
                if sel_key in results[0]:
                    agg_sel = {}
                    for cov in COVERAGE_TARGETS:
                        cov_s = str(cov)
                        econ_vals = [
                            r[sel_key][cov_s]["econ_dir"] for r in results
                        ]
                        cov_vals = [
                            r[sel_key][cov_s]["coverage"] for r in results
                        ]
                        agg_sel[cov_s] = {
                            "avg_econ_dir": float(np.mean(econ_vals)),
                            "avg_coverage": float(np.mean(cov_vals)),
                        }
                    entry[f"mean_{sel_key}"] = agg_sel

        summary["configs"][key] = entry

    # Print results
    w = 110
    print()
    print("=" * w)
    print("STACKING EXPERIMENT: Do the winners compound?")
    print("=" * w)

    print(f"\n{'Config':<35} {'Econ Dir':>10} {'Delta':>8} "
          f"{'Sharpe':>8} {'Transition':>12}")
    print("-" * w)

    base_econ = summary["configs"]["baseline"]["mean_econ_dir"]
    for key in ["baseline", *configs.keys()]:
        cfg = summary["configs"][key]
        delta = cfg["mean_econ_dir"] - base_econ
        print(
            f"{cfg['label']:<35} {cfg['mean_econ_dir']:>9.1%} "
            f"{delta:>+7.1%} {cfg['mean_sharpe_costs']:>+7.3f} "
            f"{cfg['mean_transition']:>11.1%}",
        )

    print("\n--- Selective metrics (variance-based, lower variance = keep) ---")
    print(f"{'Config':<35} {'@30%':>10} {'@50%':>10} {'@70%':>10} {'@90%':>10}")
    print("-" * w)
    for key in configs:
        cfg = summary["configs"][key]
        sel = cfg.get("mean_selective_variance")
        if sel:
            vals = [f"{sel[str(c)]['avg_econ_dir']:>9.1%}" for c in COVERAGE_TARGETS]
            print(f"{cfg['label']:<35} {'  '.join(vals)}")

    has_reject_configs = [
        k for k in configs if "mean_selective_reject" in summary["configs"][k]
    ]
    if has_reject_configs:
        print(
            "\n--- Selective metrics (rejection-head, higher score = keep) ---",
        )
        print(
            f"{'Config':<35} {'@30%':>10} {'@50%':>10} "
            f"{'@70%':>10} {'@90%':>10}",
        )
        print("-" * w)
        for key in has_reject_configs:
            cfg = summary["configs"][key]
            sel = cfg["mean_selective_reject"]
            vals = [
                f"{sel[str(c)]['avg_econ_dir']:>9.1%}"
                for c in COVERAGE_TARGETS
            ]
            print(f"{cfg['label']:<35} {'  '.join(vals)}")

    print("\n" + "=" * w)

    # Compounding analysis
    a_econ = summary["configs"]["A_hetero"]["mean_econ_dir"]
    b_econ = summary["configs"]["B_hetero_ls"]["mean_econ_dir"]
    c_econ = summary["configs"]["C_hetero_ls_ereg"]["mean_econ_dir"]
    d_econ = summary["configs"]["D_hetero_sn"]["mean_econ_dir"]

    print("\nCOMPOUNDING ANALYSIS (100% coverage / unconditional):")
    print(f"  CE baseline:              {base_econ:.1%}")
    print(f"  A (hetero):               {a_econ:.1%} ({a_econ - base_econ:+.1%})")
    print(f"  B (hetero + LS):          {b_econ:.1%} ({b_econ - a_econ:+.1%} vs A)")
    print(f"  C (hetero + LS + ereg):   {c_econ:.1%} ({c_econ - b_econ:+.1%} vs B)")
    print(f"  D (hetero + SelectiveNet):{d_econ:.1%} ({d_econ - a_econ:+.1%} vs A)")

    compounds = b_econ > a_econ + 0.005 and c_econ > b_econ + 0.005
    orthogonal = d_econ >= a_econ - 0.005

    if compounds:
        print("\n  VERDICT: Gains COMPOUND. Stack all three.")
    elif orthogonal and not compounds:
        print(
            "\n  VERDICT: LS/ereg gains DON'T compound with heteroscedastic.",
        )
        print("  Heteroscedastic already handles overconfidence.")
        if d_econ >= a_econ:
            print("  D (hetero + SelectiveNet) is the best combination.")
    else:
        print("\n  VERDICT: Heteroscedastic alone may be optimal.")

    print("=" * w)

    # Save results
    results_dir = Path.home() / ".wavecast" / "audit" / "feature_tests"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "stacking_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results saved to %s", out_path)

    return summary


if __name__ == "__main__":
    run_stacking_test()
