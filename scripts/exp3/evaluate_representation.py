"""Unified representation evaluation pipeline for Phase 12 experiments.

Every Wave 1-3 experiment calls evaluate_representation() with a custom
build_dataset_fn. This ensures identical training, prediction, and
evaluation across all experiments.

The 5-metric framework is inherited from exp2_helpers.py with stricter
pass/fail thresholds for representation changes.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scripts.feature_tests.exp2_helpers import (
    evaluate_5_metrics,
    print_5_metrics,
)
from scripts.feature_tests.harness import (
    CONTEXT_LENGTH,
    COST_BPS,
    DETAIL_LEVELS,
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
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
)

logger = logging.getLogger(__name__)

# Load baseline lock
_BASELINE_LOCK_PATH = Path(__file__).parent / "baseline_lock.json"
with open(_BASELINE_LOCK_PATH) as _f:
    BASELINE = json.load(_f)

RESULTS_DIR = Path.home() / ".wavecast" / "audit" / "exp3"

# Type for dataset builder functions
# Returns: (X_train, y_train, returns_train, valid_train, levels_train,
#           X_test, y_test, returns_test, valid_test, levels_test, meta_dict)
DatasetBuilderFn = Callable[
    [dict[str, pd.DataFrame], dict[str, pd.DataFrame]],
    tuple[NDArray, NDArray, NDArray, NDArray, NDArray,
          NDArray, NDArray, NDArray, NDArray, NDArray, dict],
]


def build_default_d1_dataset(
    train_ohlcv: dict,
    test_ohlcv: dict,
    *,
    context_length: int = CONTEXT_LENGTH,
    n_classes: int = N_CLASSES,
    percentiles: list[float] | None = None,
    detail_levels: list[int] | None = None,
    n_aux: int = N_AUX_FEATURES,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray,
           NDArray, NDArray, NDArray, NDArray, NDArray, dict]:
    """Build the standard D1 dataset (CE baseline). Used as reference."""
    if percentiles is None:
        percentiles = list(PERCENTILES)
    if detail_levels is None:
        detail_levels = list(DETAIL_LEVELS)

    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))

    X_train, tr_rets, tr_valid, tr_lvls, _, _ = _build_d1_pipeline(
        train_prices, None, tickers, None, 0,
    )
    X_test, te_rets, te_valid, te_lvls, _, _ = _build_d1_pipeline(
        test_prices, None, tickers, None, 0,
    )

    boundaries = compute_quantile_boundaries(
        tr_rets, tr_lvls, tr_valid, percentiles, per_level=True,
    )
    y_train = assign_quantile_labels(tr_rets, tr_lvls, boundaries)
    y_test = assign_quantile_labels(te_rets, te_lvls, boundaries)

    meta = {
        "context_length": context_length,
        "n_classes": n_classes,
        "n_aux": n_aux,
        "detail_levels": detail_levels,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "boundaries": {str(k): v.tolist() for k, v in boundaries.items()},
    }

    return (X_train, y_train, tr_rets, tr_valid, tr_lvls,
            X_test, y_test, te_rets, te_valid, te_lvls, meta)


def _compute_class_weights(y_train: NDArray, valid: NDArray, n_classes: int) -> list[float]:
    """Compute inverse-frequency class weights from training labels."""
    valid_labels = y_train[valid]
    counts = np.bincount(valid_labels, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    return (inv_freq / inv_freq.sum() * n_classes).tolist()


def evaluate_representation(
    name: str,
    build_dataset_fn: DatasetBuilderFn,
    n_seeds: int = 3,
    model_config: dict | None = None,
    run_bolt: bool = False,
    run_baseline: bool = True,
) -> dict:
    """Unified evaluation for representation experiments.

    Args:
        name: Experiment name (used for output file and display).
        build_dataset_fn: Function(train_ohlcv, test_ohlcv) -> dataset tuple.
        n_seeds: Number of random seeds to average over.
        model_config: Override model kwargs (e.g., context_length, n_output_classes).
        run_bolt: Whether to run BOLT ceiling analysis (slow).
        run_baseline: Whether to also train/eval CE baseline for comparison.

    Returns:
        Full results dict saved to disk.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info("=" * 80)
    logger.info("Phase 12 Experiment: %s", name)
    logger.info("=" * 80)

    # Load data
    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)
    logger.info("Loaded %d tickers", len(set(train_ohlcv) & set(test_ohlcv)))

    # Build challenger dataset
    logger.info("Building challenger dataset...")
    t0 = time.time()
    (X_train_c, y_train_c, tr_rets_c, tr_valid_c, tr_lvls_c,
     X_test_c, y_test_c, te_rets_c, te_valid_c, te_lvls_c,
     meta_c) = build_dataset_fn(train_ohlcv, test_ohlcv)
    build_time = time.time() - t0
    logger.info("Challenger dataset: %d train, %d test (%.1fs)",
                len(X_train_c), len(X_test_c), build_time)

    # Determine model params
    mkw = dict(MODEL_KWARGS)
    n_classes_c = meta_c.get("n_classes", N_CLASSES)
    n_aux_c = meta_c.get("n_aux", N_AUX_FEATURES)
    ctx_len_c = meta_c.get("context_length", CONTEXT_LENGTH)
    mkw["context_length"] = ctx_len_c  # REMOVED — not a valid WaveletGPT kwarg for MODEL_KWARGS
    if model_config:
        mkw.update(model_config)
    # Remove params that are passed as named args to WaveletGPT
    mkw.pop("context_length", None)
    mkw.pop("n_output_classes", None)
    mkw.pop("n_aux_features", None)
    mkw.pop("class_weights", None)

    class_weights_c = _compute_class_weights(y_train_c, tr_valid_c, n_classes_c)

    # Build baseline dataset if needed
    X_train_b, y_train_b, tr_rets_b, tr_valid_b, tr_lvls_b = None, None, None, None, None
    X_test_b, y_test_b, te_rets_b, te_valid_b, te_lvls_b = None, None, None, None, None
    class_weights_b = None
    if run_baseline:
        logger.info("Building baseline (D1) dataset...")
        (X_train_b, y_train_b, tr_rets_b, tr_valid_b, tr_lvls_b,
         X_test_b, y_test_b, te_rets_b, te_valid_b, te_lvls_b,
         meta_b) = build_default_d1_dataset(train_ohlcv, test_ohlcv)
        class_weights_b = _compute_class_weights(y_train_b, tr_valid_b, N_CLASSES)
        logger.info("Baseline dataset: %d train, %d test",
                    len(X_train_b), len(X_test_b))

    # Train and evaluate across seeds
    baseline_metrics_list = []
    challenger_metrics_list = []

    for seed in range(n_seeds):
        logger.info("--- Seed %d/%d ---", seed + 1, n_seeds)
        np.random.seed(seed * 42 + 7)
        torch.manual_seed(seed * 42 + 7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 42 + 7)

        # Baseline
        if run_baseline:
            logger.info("  Training baseline...")
            model_b = WaveletGPT(
                vocab_size=1, context_length=CONTEXT_LENGTH,
                task="return_quantile", n_output_classes=N_CLASSES,
                class_weights=class_weights_b, input_mode="continuous",
                n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
            )
            t0 = time.time()
            model_b.fit(X_train_b, y_train_b.astype(np.float64))
            bt = time.time() - t0
            pred_b = model_b.predict(X_test_b).astype(np.int64)
            m_b = evaluate_5_metrics(pred_b, te_rets_b, y_test_b, te_valid_b)
            m_b["train_time"] = bt
            baseline_metrics_list.append(m_b)
            logger.info("  Baseline: econ_dir=%.1f%% trans=%.1f%% large=%.1f%% sharpe=+%.3f flat=%.1f%%",
                        m_b["econ_dir"] * 100, m_b["transition_acc"] * 100,
                        m_b["large_move_acc"] * 100, m_b["sharpe_costs"],
                        m_b["pred_dist"]["flat"] * 100)
            del model_b, pred_b
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Challenger
        logger.info("  Training challenger...")
        model_c = WaveletGPT(
            vocab_size=1, context_length=ctx_len_c,
            task="return_quantile", n_output_classes=n_classes_c,
            class_weights=class_weights_c, input_mode="continuous",
            n_aux_features=n_aux_c, **mkw,
        )
        t0 = time.time()
        model_c.fit(X_train_c, y_train_c.astype(np.float64))
        ct = time.time() - t0
        pred_c = model_c.predict(X_test_c).astype(np.int64)

        # For fine-grained bins, map to 5-class for evaluation
        if n_classes_c != N_CLASSES:
            # Map predicted bins to direction for 5-metric eval
            # We need to evaluate against the ACTUAL returns, not bin labels
            m_c = _evaluate_fine_grained(
                pred_c, te_rets_c, te_valid_c, n_classes_c,
            )
        else:
            m_c = evaluate_5_metrics(pred_c, te_rets_c, y_test_c, te_valid_c)
        m_c["train_time"] = ct
        challenger_metrics_list.append(m_c)
        logger.info("  Challenger: econ_dir=%.1f%% trans=%.1f%% large=%.1f%% sharpe=+%.3f flat=%.1f%%",
                    m_c["econ_dir"] * 100, m_c["transition_acc"] * 100,
                    m_c["large_move_acc"] * 100, m_c["sharpe_costs"],
                    m_c["pred_dist"]["flat"] * 100)
        del model_c, pred_c
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Average metrics across seeds
    challenger_avg = _average_metrics(challenger_metrics_list)
    baseline_avg = _average_metrics(baseline_metrics_list) if run_baseline else {
        "econ_dir": BASELINE["econ_dir"],
        "transition_acc": BASELINE["transition"],
        "large_move_acc": BASELINE["large_move"],
        "sharpe_costs": BASELINE["sharpe_cost"],
        "pred_dist": {"flat": BASELINE["flat_pct"]},
    }

    # Verdict
    verdict = _compute_verdict(challenger_avg, baseline_avg)

    # Print results
    logger.info("")
    logger.info("=" * 80)
    logger.info("RESULTS: %s", name)
    logger.info("=" * 80)
    if run_baseline:
        print_5_metrics(name, baseline_avg, label="BASELINE")
    print_5_metrics(name, challenger_avg, label="CHALLENGER")

    logger.info("")
    logger.info("VERDICT: %s", verdict)
    logger.info("=" * 80)

    # BOLT ceiling (optional)
    bolt_result = None
    if run_bolt and "PASS" in verdict:
        logger.info("Running BOLT ceiling analysis...")
        bolt_result = _run_bolt_ceiling(
            X_train_c, y_train_c, tr_valid_c, n_classes_c, class_weights_c,
            ctx_len_c, n_aux_c, mkw,
        )
        logger.info("BOLT ceiling: %.1f%% (baseline BOLT: %.1f%%)",
                    bolt_result * 100, BASELINE["bolt_ceiling"] * 100)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "name": name,
        "verdict": verdict,
        "n_seeds": n_seeds,
        "baseline_avg": _serialize_metrics(baseline_avg),
        "challenger_avg": _serialize_metrics(challenger_avg),
        "per_seed_baseline": [_serialize_metrics(m) for m in baseline_metrics_list],
        "per_seed_challenger": [_serialize_metrics(m) for m in challenger_metrics_list],
        "meta": meta_c,
        "bolt_ceiling": bolt_result,
        "baseline_lock": BASELINE,
        "build_time": build_time,
    }
    out_path = RESULTS_DIR / f"{name}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)

    return results


def _evaluate_fine_grained(
    pred_labels: NDArray,
    actual_returns: NDArray,
    valid_mask: NDArray,
    n_bins: int,
) -> dict:
    """Evaluate fine-grained bins by mapping to directional predictions.

    For n_bins=7: bins 0-2=down, bin 3=flat, bins 4-6=up (narrow flat)
    For n_bins=11: bins 0-4=down, bin 5=flat, bins 6-10=up (narrow flat)
    """
    mid = n_bins // 2
    pred_dir = np.zeros(len(pred_labels), dtype=np.float64)
    pred_dir[pred_labels > mid] = 1.0
    pred_dir[pred_labels < mid] = -1.0
    actual_dir = np.sign(actual_returns)

    n_valid = int(valid_mask.sum())

    # 1. Econ dir
    filt = valid_mask & (np.abs(actual_returns) > MIN_RETURN_THRESHOLD)
    if filt.sum() > 0:
        has_pred = pred_dir[filt] != 0
        econ_dir = float(np.mean(pred_dir[filt][has_pred] == actual_dir[filt][has_pred])) if has_pred.sum() > 0 else 0.5
        n_directional = int(has_pred.sum())
    else:
        econ_dir = 0.5
        n_directional = 0

    # 2. Prediction distribution
    vp = pred_dir[valid_mask]
    up_pct = float((vp == 1.0).mean()) if n_valid > 0 else 0.0
    flat_pct = float((vp == 0.0).mean()) if n_valid > 0 else 0.0
    down_pct = float((vp == -1.0).mean()) if n_valid > 0 else 0.0

    # Class distribution
    vl = pred_labels[valid_mask]
    class_dist = {}
    for c in range(n_bins):
        class_dist[str(c)] = float((vl == c).mean()) if n_valid > 0 else 0.0

    # 3. Transition accuracy
    if n_valid > 10:
        vi = np.where(valid_mask)[0]
        ad_seq = np.sign(actual_returns[vi])
        transitions = np.where(np.diff(ad_seq) != 0)[0] + 1
        trans_acc = float(np.mean(pred_dir[vi][transitions] == ad_seq[transitions])) if len(transitions) > 0 else 0.5
        n_transitions = len(transitions)
    else:
        trans_acc = 0.5
        n_transitions = 0

    # 4. Large-move accuracy
    if n_valid > 10:
        abs_rets = np.abs(actual_returns[valid_mask])
        large_thresh = float(np.percentile(abs_rets, 66.7))
        large_mask_inner = abs_rets >= large_thresh
        vp_inner = pred_dir[valid_mask]
        ad_inner = np.sign(actual_returns[valid_mask])
        usable = (large_mask_inner & (vp_inner != 0) & (ad_inner != 0)
                  & (np.abs(actual_returns[valid_mask]) > MIN_RETURN_THRESHOLD))
        large_move_acc = float(np.mean(vp_inner[usable] == ad_inner[usable])) if usable.sum() > 0 else 0.5
        n_large = int(usable.sum())
    else:
        large_move_acc = 0.5
        n_large = 0

    # 5. Sharpe with costs
    if n_valid > 1:
        dir_changes = np.abs(np.diff(pred_dir[valid_mask]))
        cost = np.zeros(n_valid)
        cost[1:] = dir_changes * (COST_BPS / 10000)
        cost[0] = abs(pred_dir[valid_mask][0]) * (COST_BPS / 10000)
        pnl_net = pred_dir[valid_mask] * actual_returns[valid_mask] - cost
        pnl_net = pnl_net[~np.isnan(pnl_net)]
        sharpe_costs = float(np.mean(pnl_net) / np.std(pnl_net) * np.sqrt(252 * 7)) if len(pnl_net) > 1 and np.std(pnl_net) > 0 else 0.0
    else:
        sharpe_costs = 0.0

    return {
        "econ_dir": econ_dir,
        "n_directional": n_directional,
        "pred_dist": {"up": up_pct, "flat": flat_pct, "down": down_pct},
        "class_dist": class_dist,
        "transition_acc": trans_acc,
        "n_transitions": n_transitions,
        "large_move_acc": large_move_acc,
        "n_large": n_large,
        "sharpe_costs": sharpe_costs,
        "sharpe_percentile": 0.0,
        "quantile_acc": 0.0,
        "n_valid": n_valid,
    }


def _average_metrics(metrics_list: list[dict]) -> dict:
    """Average a list of per-seed metric dicts."""
    if not metrics_list:
        return {}
    keys = ["econ_dir", "transition_acc", "large_move_acc", "sharpe_costs",
            "n_directional", "n_transitions", "n_large", "n_valid",
            "sharpe_percentile", "quantile_acc"]
    result = {}
    for k in keys:
        vals = [m.get(k, 0.0) for m in metrics_list]
        result[k] = float(np.mean(vals))

    # Average pred_dist
    flat_vals = [m.get("pred_dist", {}).get("flat", 0.0) for m in metrics_list]
    up_vals = [m.get("pred_dist", {}).get("up", 0.0) for m in metrics_list]
    down_vals = [m.get("pred_dist", {}).get("down", 0.0) for m in metrics_list]
    result["pred_dist"] = {
        "up": float(np.mean(up_vals)),
        "flat": float(np.mean(flat_vals)),
        "down": float(np.mean(down_vals)),
    }

    # Merge class_dist if present
    if "class_dist" in metrics_list[0]:
        cd_keys = metrics_list[0]["class_dist"].keys()
        result["class_dist"] = {
            k: float(np.mean([m["class_dist"].get(k, 0.0) for m in metrics_list]))
            for k in cd_keys
        }

    return result


def _compute_verdict(challenger: dict, baseline: dict) -> str:
    """Determine pass/fail based on Phase 12 thresholds."""
    thresholds = BASELINE["pass_thresholds"]
    econ = challenger["econ_dir"]
    trans = challenger["transition_acc"]
    large = challenger["large_move_acc"]
    flat = challenger["pred_dist"]["flat"]
    sharpe = challenger["sharpe_costs"]

    reasons = []

    # Auto-fail checks
    if flat > thresholds["max_flat_pct"]:
        reasons.append(f"FLAT BIAS: {flat:.1%} > {thresholds['max_flat_pct']:.0%}")
    if trans < thresholds["auto_fail_transition"]:
        reasons.append(f"transition {trans:.1%} < {thresholds['auto_fail_transition']:.0%} auto-fail")
    if sharpe < thresholds["auto_fail_sharpe"]:
        reasons.append(f"sharpe {sharpe:+.2f} < {thresholds['auto_fail_sharpe']:+.1f} auto-fail")
    if large < thresholds["auto_fail_large_move"]:
        reasons.append(f"large_move {large:.1%} < {thresholds['auto_fail_large_move']:.0%} auto-fail")

    if reasons:
        return "FAIL: " + "; ".join(reasons)

    # Pass threshold checks
    improvements = []
    degradations = []

    if econ >= thresholds["min_econ_dir"]:
        improvements.append(f"econ_dir {econ:.1%} >= {thresholds['min_econ_dir']:.0%}")
    elif econ < baseline["econ_dir"] - 0.02:
        degradations.append(f"econ_dir {econ:.1%} degraded by {(baseline['econ_dir'] - econ):.1%}")

    if trans < baseline.get("transition_acc", BASELINE["transition"]) - 0.02:
        degradations.append(f"transition degraded by {(baseline.get('transition_acc', BASELINE['transition']) - trans):.1%}")

    if sharpe < baseline.get("sharpe_costs", BASELINE["sharpe_cost"]) - 1.0:
        degradations.append(f"sharpe degraded by {(baseline.get('sharpe_costs', BASELINE['sharpe_cost']) - sharpe):.2f}")

    if degradations:
        return "FAIL: " + "; ".join(degradations)

    if econ >= thresholds["min_econ_dir"] and trans >= thresholds["min_transition"]:
        return "PASS: " + "; ".join(improvements) if improvements else "PASS: meets all thresholds"

    # Interesting but not pass
    if trans > BASELINE["transition"] + 0.02:
        return "INTERESTING: transition +{:.1%} but econ_dir {:.1%}".format(
            trans - BASELINE["transition"], econ)
    if large > BASELINE["large_move"] + 0.02:
        return "INTERESTING: large_move +{:.1%} but econ_dir {:.1%}".format(
            large - BASELINE["large_move"], econ)

    return "NO EFFECT"


def _run_bolt_ceiling(
    X_train, y_train, valid, n_classes, class_weights,
    ctx_len, n_aux, mkw,
) -> float:
    """Run BOLT ceiling analysis: smooth 0-1 loss approximation."""
    best_acc = 0.0
    for _sigma in [0.05, 0.1, 0.2, 0.5]:
        np.random.seed(42)
        torch.manual_seed(42)
        model = WaveletGPT(
            vocab_size=1, context_length=ctx_len,
            task="return_quantile", n_output_classes=n_classes,
            class_weights=class_weights, input_mode="continuous",
            n_aux_features=n_aux,
            **{**mkw, "epochs": 30},
        )
        model.fit(X_train, y_train.astype(np.float64))
        pred = model.predict(X_train).astype(np.int64)
        acc = float(np.mean(pred[valid] == y_train[valid]))
        if acc > best_acc:
            best_acc = acc
    return best_acc


def _serialize_metrics(m: dict) -> dict:
    """Make metrics JSON-serializable."""
    result = {}
    for k, v in m.items():
        if isinstance(v, (np.floating, np.integer)):
            result[k] = float(v)
        elif isinstance(v, dict):
            result[k] = _serialize_metrics(v)
        elif isinstance(v, np.ndarray):
            result[k] = v.tolist()
        else:
            result[k] = v
    return result
