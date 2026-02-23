"""Experiment 3.7: Signed Return Regression.

Replace classification head with regression head predicting signed returns.
No flat class exists — direction comes from sign of prediction.
Loss: Huber loss (robust to outliers).

Branch: exp3/wave3-regression
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from numpy.typing import NDArray
from scripts.exp3.evaluate_representation import (
    BASELINE,
    RESULTS_DIR,
    _serialize_metrics,
)
from scripts.feature_tests.exp2_helpers import print_5_metrics
from scripts.feature_tests.harness import (
    CONTEXT_LENGTH,
    COST_BPS,
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


def _evaluate_regression(
    predictions: NDArray,
    actual_returns: NDArray,
    valid_mask: NDArray,
) -> dict:
    """Evaluate regression predictions using direction from sign.

    No flat class — direction is always sign(prediction).
    """
    n_valid = int(valid_mask.sum())

    pred_dir = np.sign(predictions)
    actual_dir = np.sign(actual_returns)

    # 1. Econ dir (all directional, no flat exclusion)
    filt = valid_mask & (np.abs(actual_returns) > MIN_RETURN_THRESHOLD)
    if filt.sum() > 0:
        # Include near-zero predictions as flat
        has_pred = np.abs(predictions[filt]) > 1e-8
        if has_pred.sum() > 0:
            econ_dir = float(np.mean(
                pred_dir[filt][has_pred] == actual_dir[filt][has_pred],
            ))
        else:
            econ_dir = 0.5
        n_directional = int(has_pred.sum())
    else:
        econ_dir = 0.5
        n_directional = 0

    # 2. Prediction distribution
    vp = pred_dir[valid_mask]
    up_pct = float((vp > 0).mean()) if n_valid > 0 else 0.0
    near_zero = float((np.abs(predictions[valid_mask]) < 1e-4).mean()) if n_valid > 0 else 0.0
    down_pct = float((vp < 0).mean()) if n_valid > 0 else 0.0

    # 3. Transition accuracy
    if n_valid > 10:
        vi = np.where(valid_mask)[0]
        ad_seq = np.sign(actual_returns[vi])
        transitions = np.where(np.diff(ad_seq) != 0)[0] + 1
        if len(transitions) > 0:
            trans_acc = float(np.mean(
                pred_dir[vi][transitions] == ad_seq[transitions],
            ))
        else:
            trans_acc = 0.5
        n_transitions = len(transitions)
    else:
        trans_acc = 0.5
        n_transitions = 0

    # 4. Large-move accuracy
    if n_valid > 10:
        abs_rets = np.abs(actual_returns[valid_mask])
        large_thresh = float(np.percentile(abs_rets, 66.7))
        large_mask = abs_rets >= large_thresh
        vp_inner = pred_dir[valid_mask]
        ad_inner = np.sign(actual_returns[valid_mask])
        usable = large_mask & (np.abs(predictions[valid_mask]) > 1e-8) & (ad_inner != 0)
        large_move_acc = float(np.mean(vp_inner[usable] == ad_inner[usable])) if usable.sum() > 0 else 0.5
        n_large = int(usable.sum())
    else:
        large_move_acc = 0.5
        n_large = 0

    # 5. Sharpe with costs (use sign as position, clip magnitude)
    if n_valid > 1:
        position = pred_dir[valid_mask]
        dir_changes = np.abs(np.diff(position))
        cost = np.zeros(n_valid)
        cost[1:] = dir_changes * (COST_BPS / 10000)
        cost[0] = abs(position[0]) * (COST_BPS / 10000)
        pnl_net = position * actual_returns[valid_mask] - cost
        pnl_net = pnl_net[~np.isnan(pnl_net)]
        sharpe_costs = float(np.mean(pnl_net) / np.std(pnl_net) * np.sqrt(252 * 7)) if len(pnl_net) > 1 and np.std(pnl_net) > 0 else 0.0
    else:
        sharpe_costs = 0.0

    # Prediction magnitude stats
    abs_preds = np.abs(predictions[valid_mask])

    return {
        "econ_dir": econ_dir,
        "n_directional": n_directional,
        "pred_dist": {"up": up_pct, "flat": near_zero, "down": down_pct},
        "transition_acc": trans_acc,
        "n_transitions": n_transitions,
        "large_move_acc": large_move_acc,
        "n_large": n_large,
        "sharpe_costs": sharpe_costs,
        "sharpe_percentile": 0.0,
        "quantile_acc": 0.0,
        "n_valid": n_valid,
        "pred_magnitude": {
            "mean": float(np.mean(abs_preds)),
            "median": float(np.median(abs_preds)),
            "std": float(np.std(abs_preds)),
            "near_zero_pct": near_zero,
        },
    }


def run_regression_experiment():
    """Run the regression experiment with Huber loss."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info("=" * 80)
    logger.info("Experiment 3.7: Signed Return Regression")
    logger.info("=" * 80)

    # Load data
    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)
    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))

    # Build D1 datasets
    X_train, tr_rets, tr_valid, tr_lvls, _, _ = _build_d1_pipeline(
        train_prices, None, tickers, None, 0,
    )
    X_test, te_rets, te_valid, te_lvls, _, _ = _build_d1_pipeline(
        test_prices, None, tickers, None, 0,
    )

    logger.info("Data: %d train, %d test", len(X_train), len(X_test))

    # Normalize returns for regression target
    valid_tr_rets = tr_rets[tr_valid]
    ret_mean = float(np.mean(valid_tr_rets))
    ret_std = float(np.std(valid_tr_rets))
    logger.info("Return stats: mean=%.6f std=%.6f", ret_mean, ret_std)

    # y for regression: z-scored returns
    y_train_reg = np.where(tr_valid, (tr_rets - ret_mean) / max(ret_std, 1e-10), 0.0).astype(np.float64)
    # y for baseline classification
    boundaries = compute_quantile_boundaries(
        tr_rets, tr_lvls, tr_valid, list(PERCENTILES), per_level=True,
    )
    y_train_cls = assign_quantile_labels(tr_rets, tr_lvls, boundaries)
    y_test_cls = assign_quantile_labels(te_rets, te_lvls, boundaries)
    class_weights = None
    valid_labels = y_train_cls[tr_valid]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    n_seeds = 3
    baseline_metrics_list = []
    regression_metrics_list = []

    for seed in range(n_seeds):
        logger.info("--- Seed %d/%d ---", seed + 1, n_seeds)
        np.random.seed(seed * 42 + 7)
        torch.manual_seed(seed * 42 + 7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 42 + 7)

        # Baseline (classification)
        logger.info("  Training baseline (CE classification)...")
        model_b = WaveletGPT(
            vocab_size=1, context_length=CONTEXT_LENGTH,
            task="return_quantile", n_output_classes=N_CLASSES,
            class_weights=class_weights, input_mode="continuous",
            n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
        )
        t0 = time.time()
        model_b.fit(X_train, y_train_cls.astype(np.float64))
        bt = time.time() - t0
        from scripts.feature_tests.exp2_helpers import evaluate_5_metrics
        pred_b = model_b.predict(X_test).astype(np.int64)
        m_b = evaluate_5_metrics(pred_b, te_rets, y_test_cls, te_valid)
        m_b["train_time"] = bt
        baseline_metrics_list.append(m_b)

        # Regression
        logger.info("  Training regression (Huber loss)...")
        model_r = WaveletGPT(
            vocab_size=1, context_length=CONTEXT_LENGTH,
            task="return_regression", input_mode="continuous",
            n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
        )
        t0 = time.time()
        model_r.fit(X_train, y_train_reg)
        rt = time.time() - t0
        pred_r = model_r.predict(X_test)  # raw float predictions

        # Inverse z-score to get actual return scale
        pred_returns = pred_r * ret_std + ret_mean

        m_r = _evaluate_regression(pred_returns, te_rets, te_valid)
        m_r["train_time"] = rt
        regression_metrics_list.append(m_r)

        logger.info("  Baseline: econ_dir=%.1f%% trans=%.1f%% sharpe=+%.3f",
                    m_b["econ_dir"] * 100, m_b["transition_acc"] * 100, m_b["sharpe_costs"])
        logger.info("  Regression: econ_dir=%.1f%% trans=%.1f%% sharpe=+%.3f near_zero=%.1f%%",
                    m_r["econ_dir"] * 100, m_r["transition_acc"] * 100, m_r["sharpe_costs"],
                    m_r["pred_dist"]["flat"] * 100)

    # Average
    def _avg(metrics_list):
        keys = ["econ_dir", "transition_acc", "large_move_acc", "sharpe_costs", "n_valid",
                "n_directional", "n_transitions", "n_large", "sharpe_percentile", "quantile_acc"]
        result = {k: float(np.mean([m.get(k, 0.0) for m in metrics_list])) for k in keys}
        result["pred_dist"] = {
            "up": float(np.mean([m["pred_dist"]["up"] for m in metrics_list])),
            "flat": float(np.mean([m["pred_dist"]["flat"] for m in metrics_list])),
            "down": float(np.mean([m["pred_dist"]["down"] for m in metrics_list])),
        }
        return result

    baseline_avg = _avg(baseline_metrics_list)
    regression_avg = _avg(regression_metrics_list)

    # Verdict
    econ = regression_avg["econ_dir"]
    trans = regression_avg["transition_acc"]
    flat = regression_avg["pred_dist"]["flat"]
    if econ >= 0.65 and trans >= 0.523 and flat < 0.40:
        verdict = f"PASS: econ_dir={econ:.1%}"
    elif econ < BASELINE["econ_dir"] - 0.02 or trans < 0.45:
        verdict = f"FAIL: econ_dir={econ:.1%} trans={trans:.1%}"
    elif trans > BASELINE["transition"] + 0.02:
        verdict = f"INTERESTING: trans={trans:.1%} but econ_dir={econ:.1%}"
    else:
        verdict = "NO EFFECT"

    # Print
    print_5_metrics("Regression", regression_avg, label="CHALLENGER")
    print_5_metrics("CE Baseline", baseline_avg, label="BASELINE")
    logger.info("\nVERDICT: %s", verdict)

    # Save
    import json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "name": "regression",
        "verdict": verdict,
        "n_seeds": n_seeds,
        "baseline_avg": _serialize_metrics(baseline_avg),
        "regression_avg": _serialize_metrics(regression_avg),
        "per_seed_baseline": [_serialize_metrics(m) for m in baseline_metrics_list],
        "per_seed_regression": [_serialize_metrics(m) for m in regression_metrics_list],
        "return_stats": {"mean": ret_mean, "std": ret_std},
    }
    out_path = RESULTS_DIR / "regression_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)

    return results


if __name__ == "__main__":
    results = run_regression_experiment()
    print(f"\nFinal verdict: {results['verdict']}")
