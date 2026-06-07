"""Cross-tabulation: large-move × transition accuracy for CE, finegrain_11, regression.

The expensive error is being wrong on a large move that's also a transition (reversal).
This script trains each model variant once and computes the 2×2 cross-tab:
  {large, not-large} × {transition, not-transition}

Usage:
    cd ~/wavecast && source .venv/bin/activate
    python -u -m scripts.exp3.crosstab_analysis
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from scripts.exp3.evaluate_representation import (
    BASELINE,
    RESULTS_DIR,
    build_default_d1_dataset,
    _compute_class_weights,
)
from scripts.exp3.test_finegrain import _make_finegrain_builder
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

SEED = 7  # seed * 42 + 7 with seed=0, matching existing convention


def _compute_crosstab(
    pred_dir: NDArray,
    actual_returns: NDArray,
    valid_mask: NDArray,
) -> dict:
    """Compute 2×2 cross-tab: {large, medium, small} × {transition, non-transition}.

    Returns accuracy, n_samples, avg_win, avg_loss for each cell.
    """
    n_valid = int(valid_mask.sum())
    if n_valid < 20:
        return {"error": "too few valid samples"}

    vi = np.where(valid_mask)[0]
    actual_dir = np.sign(actual_returns)

    # Transition mask (on valid indices)
    ad_seq = actual_dir[vi]
    is_transition = np.zeros(len(vi), dtype=bool)
    is_transition[1:] = np.diff(ad_seq) != 0

    # Large-move mask (on valid indices, top 1/3 by |return|)
    abs_rets = np.abs(actual_returns[vi])
    large_thresh = float(np.percentile(abs_rets, 66.7))
    small_thresh = float(np.percentile(abs_rets, 33.3))
    is_large = abs_rets >= large_thresh
    is_small = abs_rets < small_thresh
    is_medium = ~is_large & ~is_small

    # Predictions on valid indices
    pv = pred_dir[vi]
    av = actual_dir[vi]

    # Filter: only directional predictions on non-trivial returns
    usable = (pv != 0) & (av != 0) & (np.abs(actual_returns[vi]) > MIN_RETURN_THRESHOLD)

    results = {}
    for size_name, size_mask in [("large", is_large), ("medium", is_medium), ("small", is_small)]:
        for trans_name, trans_mask in [("transition", is_transition), ("non_transition", ~is_transition)]:
            cell_mask = usable & size_mask & trans_mask
            n = int(cell_mask.sum())
            if n == 0:
                results[f"{size_name}_{trans_name}"] = {
                    "accuracy": None, "n_samples": 0,
                    "avg_win": None, "avg_loss": None,
                }
                continue

            correct = pv[cell_mask] == av[cell_mask]
            accuracy = float(correct.mean())
            rets = actual_returns[vi][cell_mask]

            # PnL: correct predictions earn |return|, wrong predictions lose |return|
            correct_rets = np.abs(rets[correct])
            wrong_rets = np.abs(rets[~correct])

            avg_win = float(correct_rets.mean()) if len(correct_rets) > 0 else 0.0
            avg_loss = float(wrong_rets.mean()) if len(wrong_rets) > 0 else 0.0
            n_correct = int(correct.sum())
            n_wrong = n - n_correct
            net_pnl = n_correct * avg_win - n_wrong * avg_loss
            expectancy = avg_win * accuracy - avg_loss * (1 - accuracy)

            results[f"{size_name}_{trans_name}"] = {
                "accuracy": accuracy,
                "n_samples": n,
                "n_correct": n_correct,
                "n_wrong": n_wrong,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "net_pnl": net_pnl,
                "expectancy_per_trade": expectancy,
            }

    # Summary: large-move transitions specifically
    lt = results.get("large_transition", {})
    lnt = results.get("large_non_transition", {})
    results["summary"] = {
        "large_transition_acc": lt.get("accuracy"),
        "large_non_transition_acc": lnt.get("accuracy"),
        "large_transition_expectancy": lt.get("expectancy_per_trade"),
        "large_non_transition_expectancy": lnt.get("expectancy_per_trade"),
        "large_transition_n": lt.get("n_samples", 0),
        "large_non_transition_n": lnt.get("n_samples", 0),
    }

    return results


def _train_and_predict_ce(X_train, y_train, X_test, valid_train):
    """Train CE baseline, return (pred_labels, pred_proba)."""
    class_weights = _compute_class_weights(y_train, valid_train, N_CLASSES)
    model = WaveletGPT(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=N_CLASSES,
        class_weights=class_weights, input_mode="continuous",
        n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
    )
    t0 = time.time()
    model.fit(X_train, y_train.astype(np.float64))
    logger.info("  CE baseline trained in %.1fs", time.time() - t0)
    pred_labels = model.predict(X_test).astype(np.int64)
    pred_proba = model.predict_proba(X_test)  # (N, 5) softmax
    return pred_labels, pred_proba, model


def _train_and_predict_finegrain11(train_ohlcv, test_ohlcv):
    """Train finegrain_11, return (pred_dir, pred_labels)."""
    builder = _make_finegrain_builder(11)
    (X_train, y_train, tr_rets, tr_valid, tr_lvls,
     X_test, y_test, te_rets, te_valid, te_lvls, meta) = builder(train_ohlcv, test_ohlcv)

    class_weights = _compute_class_weights(y_train, tr_valid, 11)
    mkw = dict(MODEL_KWARGS)
    model = WaveletGPT(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=11,
        class_weights=class_weights, input_mode="continuous",
        n_aux_features=N_AUX_FEATURES, **mkw,
    )
    t0 = time.time()
    model.fit(X_train, y_train.astype(np.float64))
    logger.info("  finegrain_11 trained in %.1fs", time.time() - t0)
    pred_labels = model.predict(X_test).astype(np.int64)

    # Map to direction: bins 0-4 = down, bin 5 = flat, bins 6-10 = up
    mid = 11 // 2  # 5
    pred_dir = np.zeros(len(pred_labels), dtype=np.float64)
    pred_dir[pred_labels > mid] = 1.0
    pred_dir[pred_labels < mid] = -1.0

    return pred_dir, pred_labels, te_rets, te_valid, model


def _train_and_predict_regression(X_train, tr_rets, tr_valid, X_test):
    """Train regression, return (pred_dir, pred_magnitudes)."""
    valid_tr_rets = tr_rets[tr_valid]
    ret_mean = float(np.mean(valid_tr_rets))
    ret_std = float(np.std(valid_tr_rets))

    y_train_reg = np.where(tr_valid, (tr_rets - ret_mean) / max(ret_std, 1e-10), 0.0).astype(np.float64)

    model = WaveletGPT(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_regression", input_mode="continuous",
        n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
    )
    t0 = time.time()
    model.fit(X_train, y_train_reg)
    logger.info("  regression trained in %.1fs", time.time() - t0)
    pred_raw = model.predict(X_test)
    pred_returns = pred_raw * ret_std + ret_mean
    pred_dir = np.sign(pred_returns)

    return pred_dir, pred_returns, model, ret_mean, ret_std


def run_crosstab():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info("=" * 80)
    logger.info("CROSS-TABULATION: large-move × transition accuracy")
    logger.info("=" * 80)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Load data
    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)
    logger.info("Loaded %d tickers", len(set(train_ohlcv) & set(test_ohlcv)))

    # Build baseline dataset
    logger.info("Building D1 dataset...")
    (X_train, y_train, tr_rets, tr_valid, tr_lvls,
     X_test, y_test, te_rets, te_valid, te_lvls, meta) = build_default_d1_dataset(
        train_ohlcv, test_ohlcv,
    )
    logger.info("Dataset: %d train, %d test, %d valid test",
                len(X_train), len(X_test), int(te_valid.sum()))

    all_results = {}

    # --- 1. CE Baseline ---
    logger.info("\n--- CE Baseline (5-class) ---")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    pred_labels_ce, pred_proba_ce, model_ce = _train_and_predict_ce(
        X_train, y_train, X_test, tr_valid,
    )
    mid_ce = N_CLASSES // 2
    pred_dir_ce = np.zeros(len(pred_labels_ce), dtype=np.float64)
    pred_dir_ce[pred_labels_ce > mid_ce] = 1.0
    pred_dir_ce[pred_labels_ce < mid_ce] = -1.0

    ct_ce = _compute_crosstab(pred_dir_ce, te_rets, te_valid)
    all_results["ce_baseline"] = ct_ce

    del model_ce
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 2. Finegrain_11 ---
    logger.info("\n--- Finegrain_11 (11-class) ---")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    pred_dir_fg, pred_labels_fg, te_rets_fg, te_valid_fg, model_fg = _train_and_predict_finegrain11(
        train_ohlcv, test_ohlcv,
    )
    ct_fg = _compute_crosstab(pred_dir_fg, te_rets_fg, te_valid_fg)
    all_results["finegrain_11"] = ct_fg

    del model_fg
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 3. Regression ---
    logger.info("\n--- Regression (Huber) ---")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    pred_dir_reg, pred_returns_reg, model_reg, ret_mean, ret_std = _train_and_predict_regression(
        X_train, tr_rets, tr_valid, X_test,
    )
    ct_reg = _compute_crosstab(pred_dir_reg, te_rets, te_valid)
    all_results["regression"] = ct_reg

    del model_reg
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Print comparison ---
    print("\n" + "=" * 100)
    print("CROSS-TABULATION RESULTS: accuracy on {size} × {transition}")
    print("=" * 100)

    header = f"{'Cell':<28} {'CE Baseline':>14} {'Finegrain_11':>14} {'Regression':>14}"
    print(header)
    print("-" * 70)

    cells = [
        "large_transition", "large_non_transition",
        "medium_transition", "medium_non_transition",
        "small_transition", "small_non_transition",
    ]
    for cell in cells:
        ce_val = all_results["ce_baseline"].get(cell, {})
        fg_val = all_results["finegrain_11"].get(cell, {})
        reg_val = all_results["regression"].get(cell, {})

        def _fmt(v):
            acc = v.get("accuracy")
            n = v.get("n_samples", 0)
            if acc is None:
                return "  ---"
            return f"{acc:.1%} (n={n:,})"

        print(f"  {cell:<26} {_fmt(ce_val):>14} {_fmt(fg_val):>14} {_fmt(reg_val):>14}")

    # The key comparison
    print("\n" + "=" * 100)
    print("KEY METRIC: Large-Move Transitions (the expensive error)")
    print("=" * 100)
    for name, res in all_results.items():
        lt = res.get("large_transition", {})
        lnt = res.get("large_non_transition", {})
        acc_lt = lt.get("accuracy")
        acc_lnt = lnt.get("accuracy")
        exp_lt = lt.get("expectancy_per_trade")
        n_lt = lt.get("n_samples", 0)
        n_lnt = lnt.get("n_samples", 0)

        print(f"\n  {name}:")
        if acc_lt is not None:
            print(f"    Large + Transition:     {acc_lt:.1%}  (n={n_lt:,})  "
                  f"expectancy={exp_lt:.4%}/trade")
        if acc_lnt is not None:
            print(f"    Large + Non-transition: {acc_lnt:.1%}  (n={n_lnt:,})  "
                  f"expectancy={lnt.get('expectancy_per_trade', 0):.4%}/trade")

        # PnL comparison
        if lt.get("net_pnl") is not None and lnt.get("net_pnl") is not None:
            print(f"    Net PnL (large trans):  {lt['net_pnl']:.2f}")
            print(f"    Net PnL (large !trans): {lnt['net_pnl']:.2f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "crosstab_results.json"

    # Make serializable
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(_clean(all_results), f, indent=2)
    logger.info("\nResults saved to %s", out_path)

    return all_results


if __name__ == "__main__":
    run_crosstab()
