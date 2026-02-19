"""Shared helpers for Round 2 ephemeral branch experiments.

Every Round 2 experiment MUST report 5 metrics:
1. Econ dir accuracy
2. Prediction distribution (up% / flat% / down%)
3. Transition accuracy
4. Large-move accuracy (top 1/3 by |return|)
5. Sharpe with costs

This module provides the evaluation function, data loading, and common constants.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scripts.feature_tests.harness import (
    COST_BPS,
    MIN_RETURN_THRESHOLD,
    N_CLASSES,
    N_RANDOM_TRIALS,
)

logger = logging.getLogger(__name__)

MID = N_CLASSES // 2


def evaluate_5_metrics(
    pred_labels: NDArray,
    actual_returns: NDArray,
    actual_labels: NDArray,
    valid_mask: NDArray,
) -> dict:
    """Compute all 5 required metrics for Round 2 experiments.

    Args:
        pred_labels: Predicted class labels (0..N_CLASSES-1).
        actual_returns: Raw returns per sample.
        actual_labels: Ground-truth class labels.
        valid_mask: Boolean mask for valid samples.

    Returns:
        Dict with econ_dir, pred_dist, transition_acc, large_move_acc,
        sharpe_costs, plus auxiliary metrics.
    """
    n_valid = int(valid_mask.sum())

    # Direction from class labels
    pred_dir = np.zeros(len(pred_labels), dtype=np.float64)
    pred_dir[pred_labels > MID] = 1.0
    pred_dir[pred_labels < MID] = -1.0
    actual_dir = np.sign(actual_returns)

    # 1. Econ dir accuracy (only directional preds on non-trivial returns)
    filt = valid_mask & (np.abs(actual_returns) > MIN_RETURN_THRESHOLD)
    if filt.sum() > 0:
        has_pred = pred_dir[filt] != 0
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
    up_pct = float((vp == 1.0).mean()) if n_valid > 0 else 0.0
    flat_pct = float((vp == 0.0).mean()) if n_valid > 0 else 0.0
    down_pct = float((vp == -1.0).mean()) if n_valid > 0 else 0.0

    # Also check raw class distribution (all 5 classes)
    vl = pred_labels[valid_mask]
    class_dist = {}
    for c in range(N_CLASSES):
        class_dist[str(c)] = float((vl == c).mean()) if n_valid > 0 else 0.0

    # 3. Transition accuracy
    if n_valid > 10:
        vi = np.where(valid_mask)[0]
        ad_seq = np.sign(actual_returns[vi])
        transitions = np.where(np.diff(ad_seq) != 0)[0] + 1
        if len(transitions) > 0:
            trans_acc = float(np.mean(
                pred_dir[vi][transitions] == ad_seq[transitions],
            ))
            n_transitions = len(transitions)
        else:
            trans_acc = 0.5
            n_transitions = 0
    else:
        trans_acc = 0.5
        n_transitions = 0

    # 4. Large-move accuracy (top 1/3 by |return|)
    if n_valid > 10:
        abs_rets = np.abs(actual_returns[valid_mask])
        large_thresh = float(np.percentile(abs_rets, 66.7))
        large_mask_inner = abs_rets >= large_thresh
        # Also need non-zero predictions and returns above threshold
        vp_inner = pred_dir[valid_mask]
        ad_inner = np.sign(actual_returns[valid_mask])
        usable = (large_mask_inner
                  & (vp_inner != 0)
                  & (ad_inner != 0)
                  & (np.abs(actual_returns[valid_mask]) > MIN_RETURN_THRESHOLD))
        if usable.sum() > 0:
            large_move_acc = float(np.mean(
                vp_inner[usable] == ad_inner[usable],
            ))
        else:
            large_move_acc = 0.5
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
        if len(pnl_net) > 1 and np.std(pnl_net) > 0:
            sharpe_costs = float(
                np.mean(pnl_net) / np.std(pnl_net) * np.sqrt(252 * 7),
            )
        else:
            sharpe_costs = 0.0
    else:
        sharpe_costs = 0.0

    # Random baseline percentile
    rng = np.random.default_rng(42)
    random_sharpes = []
    if n_valid > 1:
        vr = actual_returns[valid_mask]
        vr = vr[~np.isnan(vr)]
        for _ in range(N_RANDOM_TRIALS):
            rd = rng.choice([-1.0, 0.0, 1.0], size=len(vr))
            rp = rd * vr
            if len(rp) > 1 and np.std(rp) > 0:
                random_sharpes.append(
                    float(np.mean(rp) / np.std(rp) * np.sqrt(252 * 7)),
                )
    sharpe_pctile = (
        float(np.mean(np.array(random_sharpes) < sharpe_costs) * 100)
        if random_sharpes else 50.0
    )

    # Quantile accuracy
    q_acc = (
        float(np.mean(pred_labels[valid_mask] == actual_labels[valid_mask]))
        if n_valid > 0 else 0.0
    )

    return {
        "econ_dir": econ_dir,
        "n_directional": n_directional,
        "pred_dist": {
            "up": up_pct, "flat": flat_pct, "down": down_pct,
        },
        "class_dist": class_dist,
        "transition_acc": trans_acc,
        "n_transitions": n_transitions,
        "large_move_acc": large_move_acc,
        "n_large": n_large,
        "sharpe_costs": sharpe_costs,
        "sharpe_percentile": sharpe_pctile,
        "quantile_acc": q_acc,
        "n_valid": n_valid,
    }


def print_5_metrics(name: str, metrics: dict, label: str = "") -> None:
    """Pretty-print the 5-metric summary."""
    w = 100
    prefix = f" ({label})" if label else ""
    print(f"\n{'=' * w}")
    print(f"  {name}{prefix}")
    print(f"{'=' * w}")
    print(f"  Econ Dir Accuracy:    {metrics['econ_dir']:.1%}"
          f"  ({metrics['n_directional']} directional preds)")
    print(f"  Prediction Dist:      up={metrics['pred_dist']['up']:.1%}"
          f"  flat={metrics['pred_dist']['flat']:.1%}"
          f"  down={metrics['pred_dist']['down']:.1%}")
    print(f"  Transition Accuracy:  {metrics['transition_acc']:.1%}"
          f"  ({metrics['n_transitions']} transitions)")
    print(f"  Large-Move Accuracy:  {metrics['large_move_acc']:.1%}"
          f"  ({metrics['n_large']} large moves)")
    print(f"  Sharpe (with costs):  {metrics['sharpe_costs']:+.3f}"
          f"  (P{metrics['sharpe_percentile']:.0f} vs random)")
    print(f"  Quantile Accuracy:    {metrics['quantile_acc']:.1%}")
    if "class_dist" in metrics:
        cd = metrics["class_dist"]
        print("  Class Dist:           "
              + "  ".join(f"c{k}={v:.1%}" for k, v in sorted(cd.items())))
    print(f"{'=' * w}")


def save_results(name: str, results: dict) -> Path:
    """Save experiment results to standard location."""
    results_dir = Path.home() / ".wavecast" / "audit" / "feature_tests"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{name}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)
    return out_path


def verdict_from_metrics(
    baseline: dict,
    challenger: dict,
    *,
    min_econ_dir: float = 0.63,
    min_transition: float = 0.52,
    min_large_move: float = 0.68,
    max_flat_pct: float = 0.40,
    min_sharpe: float = 4.0,
) -> str:
    """Determine PASS/FAIL based on absolute thresholds.

    Unlike Round 1's delta-based verdicts, Round 2 uses absolute thresholds
    plus a flat-bias check.
    """
    econ = challenger["econ_dir"]
    trans = challenger["transition_acc"]
    large = challenger["large_move_acc"]
    flat = challenger["pred_dist"]["flat"]
    sharpe = challenger["sharpe_costs"]

    reasons = []

    # Flat-bias gaming check (CRITICAL)
    if flat > max_flat_pct:
        reasons.append(f"FLAT BIAS: {flat:.1%} flat (>{max_flat_pct:.0%} threshold)")

    # Absolute thresholds
    if econ < min_econ_dir:
        reasons.append(f"econ_dir {econ:.1%} < {min_econ_dir:.0%}")
    if trans < min_transition:
        reasons.append(f"transition {trans:.1%} < {min_transition:.0%}")
    if large < min_large_move:
        reasons.append(f"large_move {large:.1%} < {min_large_move:.0%}")

    if reasons:
        return "FAIL: " + "; ".join(reasons)

    # Improvements over baseline
    improvements = []
    if econ > baseline["econ_dir"] + 0.01:
        improvements.append(f"econ_dir +{(econ - baseline['econ_dir']):.1%}")
    if trans > baseline["transition_acc"] + 0.02:
        improvements.append(f"transition +{(trans - baseline['transition_acc']):.1%}")
    if large > baseline["large_move_acc"] + 0.01:
        improvements.append(f"large_move +{(large - baseline['large_move_acc']):.1%}")
    if sharpe > baseline["sharpe_costs"] + 0.2:
        improvements.append(f"sharpe +{(sharpe - baseline['sharpe_costs']):.2f}")

    if improvements:
        return "PASS: " + "; ".join(improvements)

    return "NO EFFECT"
