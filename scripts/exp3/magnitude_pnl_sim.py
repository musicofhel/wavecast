"""Magnitude-weighted PnL simulation on CE baseline.

Tests whether softmax extreme-bin probability correlates with actual move
magnitude, then runs PnL simulations comparing:
  1. Flat sizing (position = ±1)
  2. Confidence-weighted (position = max_softmax × direction)
  3. Extreme-bin weighted (position = P(strong_up) + P(strong_down))
  4. Hybrid: CE direction × regression |prediction| for magnitude

Also tests the correlation prerequisite: does the model's implied magnitude
signal predict actual magnitude?

Usage:
    cd ~/wavecast && source .venv/bin/activate
    python -u -m scripts.exp3.magnitude_pnl_sim
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

SEED = 7


def _compute_sharpe(pnl: NDArray, annual_factor: float = np.sqrt(252 * 7)) -> float:
    """Annualized Sharpe from per-period PnL."""
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) < 2 or np.std(pnl) == 0:
        return 0.0
    return float(np.mean(pnl) / np.std(pnl) * annual_factor)


def _compute_max_drawdown(equity: NDArray) -> float:
    """Max drawdown from equity curve."""
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    return float(np.min(dd))


def _run_pnl_sim(
    positions: NDArray,
    actual_returns: NDArray,
    valid_mask: NDArray,
    cost_bps: float = COST_BPS,
) -> dict:
    """Run PnL simulation with position array and actual returns.

    positions: signed position sizes (e.g., +0.5, -1.0, 0.0)
    """
    n_valid = int(valid_mask.sum())
    vi = np.where(valid_mask)[0]

    pos = positions[vi]
    rets = actual_returns[vi]

    # Transaction costs on position changes
    cost = np.zeros(n_valid)
    pos_changes = np.abs(np.diff(pos))
    cost[1:] = pos_changes * (cost_bps / 10000)
    cost[0] = abs(pos[0]) * (cost_bps / 10000)

    pnl_gross = pos * rets
    pnl_net = pnl_gross - cost
    pnl_net = pnl_net[~np.isnan(pnl_net)]

    equity = np.cumprod(1 + pnl_net)
    total_return = float(equity[-1] - 1) if len(equity) > 0 else 0.0

    # Directional accuracy (where we have a position)
    has_pos = np.abs(pos) > 1e-8
    actual_dir = np.sign(rets)
    pred_dir = np.sign(pos)
    if has_pos.sum() > 0:
        correct = pred_dir[has_pos] == actual_dir[has_pos]
        dir_acc = float(correct.mean())
    else:
        dir_acc = 0.0

    # Win/loss stats
    winning = pnl_net > 0
    losing = pnl_net < 0
    win_rate = float(winning.mean()) if len(pnl_net) > 0 else 0.0
    avg_win = float(pnl_net[winning].mean()) if winning.sum() > 0 else 0.0
    avg_loss = float(np.abs(pnl_net[losing]).mean()) if losing.sum() > 0 else 0.0

    return {
        "sharpe": _compute_sharpe(pnl_net),
        "total_return": total_return,
        "max_drawdown": _compute_max_drawdown(equity),
        "dir_accuracy": dir_acc,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_win_loss_ratio": avg_win / avg_loss if avg_loss > 0 else float("inf"),
        "n_trades": int(has_pos.sum()),
        "avg_position_size": float(np.abs(pos[has_pos]).mean()) if has_pos.sum() > 0 else 0.0,
        "total_costs": float(cost.sum()),
        "cum_pnl_gross": float(pnl_gross[~np.isnan(pnl_gross)].sum()),
        "cum_pnl_net": float(pnl_net.sum()),
    }


def run_magnitude_sim():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info("=" * 80)
    logger.info("MAGNITUDE-WEIGHTED PnL SIMULATION")
    logger.info("=" * 80)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Load data
    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)

    logger.info("Building D1 dataset...")
    (X_train, y_train, tr_rets, tr_valid, tr_lvls,
     X_test, y_test, te_rets, te_valid, te_lvls, meta) = build_default_d1_dataset(
        train_ohlcv, test_ohlcv,
    )
    logger.info("Dataset: %d train, %d test", len(X_train), len(X_test))

    # --- Train CE Baseline ---
    logger.info("\nTraining CE baseline...")
    class_weights = _compute_class_weights(y_train, tr_valid, N_CLASSES)
    model_ce = WaveletGPT(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=N_CLASSES,
        class_weights=class_weights, input_mode="continuous",
        n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
    )
    t0 = time.time()
    model_ce.fit(X_train, y_train.astype(np.float64))
    logger.info("  CE trained in %.1fs", time.time() - t0)

    pred_labels = model_ce.predict(X_test).astype(np.int64)
    pred_proba = model_ce.predict_proba(X_test)  # (N, 5)

    del model_ce
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Train Regression (for hybrid) ---
    logger.info("\nTraining regression (for hybrid magnitude signal)...")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    valid_tr_rets = tr_rets[tr_valid]
    ret_mean = float(np.mean(valid_tr_rets))
    ret_std = float(np.std(valid_tr_rets))
    y_train_reg = np.where(tr_valid, (tr_rets - ret_mean) / max(ret_std, 1e-10), 0.0).astype(np.float64)

    model_reg = WaveletGPT(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_regression", input_mode="continuous",
        n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
    )
    t0 = time.time()
    model_reg.fit(X_train, y_train_reg)
    logger.info("  Regression trained in %.1fs", time.time() - t0)
    pred_raw_reg = model_reg.predict(X_test)
    pred_returns_reg = pred_raw_reg * ret_std + ret_mean

    del model_reg
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === CORRELATION ANALYSIS ===
    logger.info("\n" + "=" * 80)
    logger.info("CORRELATION: model signals vs actual magnitude")
    logger.info("=" * 80)

    vi = np.where(te_valid)[0]
    actual_mag = np.abs(te_rets[vi])

    mid = N_CLASSES // 2
    pred_dir_ce = np.zeros(len(pred_labels), dtype=np.float64)
    pred_dir_ce[pred_labels > mid] = 1.0
    pred_dir_ce[pred_labels < mid] = -1.0

    # Magnitude signal 1: max softmax confidence
    max_conf = np.max(pred_proba, axis=1)
    conf_on_valid = max_conf[vi]
    r_conf = float(np.corrcoef(conf_on_valid, actual_mag)[0, 1])
    logger.info("  Correlation(max_softmax_conf, |actual_return|): r = %.4f", r_conf)

    # Magnitude signal 2: extreme-bin probability = P(class 0) + P(class 4)
    extreme_prob = pred_proba[:, 0] + pred_proba[:, -1]
    extreme_on_valid = extreme_prob[vi]
    r_extreme = float(np.corrcoef(extreme_on_valid, actual_mag)[0, 1])
    logger.info("  Correlation(P(extreme_bins), |actual_return|):  r = %.4f", r_extreme)

    # Magnitude signal 3: entropy-based (lower entropy = more confident)
    entropy = -np.sum(pred_proba * np.log(pred_proba + 1e-10), axis=1)
    inv_entropy = 1.0 / (entropy + 1e-10)  # higher = more confident
    inv_entropy_valid = inv_entropy[vi]
    r_entropy = float(np.corrcoef(inv_entropy_valid, actual_mag)[0, 1])
    logger.info("  Correlation(1/entropy, |actual_return|):        r = %.4f", r_entropy)

    # Magnitude signal 4: regression |prediction|
    reg_mag = np.abs(pred_returns_reg)
    reg_mag_valid = reg_mag[vi]
    r_reg = float(np.corrcoef(reg_mag_valid, actual_mag)[0, 1])
    logger.info("  Correlation(|regression_pred|, |actual_return|): r = %.4f", r_reg)

    # Magnitude signal 5: expected magnitude from softmax
    # E[|return|] ≈ sum over classes of P(class) × typical_magnitude(class)
    # Use class centers as magnitude proxies: {0: -2, 1: -1, 2: 0, 3: 1, 4: 2} (ordinal)
    class_magnitudes = np.array([2.0, 1.0, 0.0, 1.0, 2.0])
    expected_mag = pred_proba @ class_magnitudes  # weighted sum
    expected_mag_valid = expected_mag[vi]
    r_expected = float(np.corrcoef(expected_mag_valid, actual_mag)[0, 1])
    logger.info("  Correlation(E[ordinal_mag], |actual_return|):   r = %.4f", r_expected)

    correlations = {
        "max_softmax_conf": r_conf,
        "extreme_bin_prob": r_extreme,
        "inv_entropy": r_entropy,
        "regression_abs_pred": r_reg,
        "expected_ordinal_mag": r_expected,
    }

    # === Rank correlation (Spearman) ===
    from scipy.stats import spearmanr
    logger.info("\n  Spearman rank correlations:")
    spearman_results = {}
    for name, signal in [
        ("max_softmax_conf", conf_on_valid),
        ("extreme_bin_prob", extreme_on_valid),
        ("regression_abs_pred", reg_mag_valid),
        ("expected_ordinal_mag", expected_mag_valid),
    ]:
        rho, pval = spearmanr(signal, actual_mag)
        logger.info("    %s: rho = %.4f (p = %.2e)", name, rho, pval)
        spearman_results[name] = {"rho": float(rho), "p_value": float(pval)}

    # === PnL SIMULATIONS ===
    logger.info("\n" + "=" * 80)
    logger.info("PnL SIMULATIONS")
    logger.info("=" * 80)

    strategies = {}

    # Strategy 1: Flat sizing (±1 or 0)
    pos_flat = pred_dir_ce.copy()
    strategies["flat"] = _run_pnl_sim(pos_flat, te_rets, te_valid)

    # Strategy 2: Confidence-weighted (max softmax × direction)
    pos_conf = pred_dir_ce * max_conf
    strategies["confidence_weighted"] = _run_pnl_sim(pos_conf, te_rets, te_valid)

    # Strategy 3: Extreme-bin weighted
    pos_extreme = pred_dir_ce * extreme_prob
    strategies["extreme_bin_weighted"] = _run_pnl_sim(pos_extreme, te_rets, te_valid)

    # Strategy 4: Expected magnitude weighted
    pos_expected = pred_dir_ce * (expected_mag / 2.0)  # normalize to [0,1]
    strategies["expected_mag_weighted"] = _run_pnl_sim(pos_expected, te_rets, te_valid)

    # Strategy 5: Regression magnitude (hybrid — CE direction × |regression pred|)
    # Normalize regression magnitude to [0, 1] range
    reg_mag_norm = np.clip(reg_mag / np.percentile(reg_mag[vi], 95), 0, 1)
    pos_hybrid = pred_dir_ce * reg_mag_norm
    strategies["hybrid_ce_dir_reg_mag"] = _run_pnl_sim(pos_hybrid, te_rets, te_valid)

    # Strategy 6: Regression as standalone (direction + magnitude)
    pred_dir_reg = np.sign(pred_returns_reg)
    pos_reg_standalone = pred_dir_reg * reg_mag_norm
    strategies["regression_standalone"] = _run_pnl_sim(pos_reg_standalone, te_rets, te_valid)

    # Strategy 7: Flat sizing but ONLY on large predicted moves (selective)
    # Use extreme_prob > median as "large predicted move" filter
    extreme_median = float(np.median(extreme_prob[vi]))
    pos_selective = pred_dir_ce.copy()
    pos_selective[extreme_prob < extreme_median] = 0.0
    strategies["selective_large_pred_only"] = _run_pnl_sim(pos_selective, te_rets, te_valid)

    # Strategy 8: Tiered sizing (0.5× small, 1× medium, 2× large predicted)
    extreme_p33 = float(np.percentile(extreme_prob[vi], 33.3))
    extreme_p67 = float(np.percentile(extreme_prob[vi], 66.7))
    pos_tiered = pred_dir_ce.copy()
    small_pred = extreme_prob < extreme_p33
    med_pred = (extreme_prob >= extreme_p33) & (extreme_prob < extreme_p67)
    large_pred = extreme_prob >= extreme_p67
    pos_tiered[small_pred] *= 0.5
    pos_tiered[med_pred] *= 1.0
    pos_tiered[large_pred] *= 2.0
    strategies["tiered_sizing"] = _run_pnl_sim(pos_tiered, te_rets, te_valid)

    # Print comparison
    print("\n" + "=" * 110)
    print("PnL SIMULATION RESULTS")
    print("=" * 110)
    header = (f"{'Strategy':<30} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8} "
              f"{'DirAcc':>8} {'WinRate':>8} {'W/L':>6} {'AvgPos':>8} {'Trades':>8}")
    print(header)
    print("-" * 110)

    for name, s in strategies.items():
        print(f"  {name:<28} {s['sharpe']:>+7.3f} {s['total_return']:>+9.2%} "
              f"{s['max_drawdown']:>+7.2%} {s['dir_accuracy']:>7.1%} "
              f"{s['win_rate']:>7.1%} {s['avg_win_loss_ratio']:>5.2f} "
              f"{s['avg_position_size']:>7.3f} {s['n_trades']:>7,}")

    # === MAGNITUDE DECILE ANALYSIS ===
    # How does accuracy vary by predicted magnitude?
    logger.info("\n" + "=" * 80)
    logger.info("ACCURACY BY PREDICTED MAGNITUDE DECILE")
    logger.info("=" * 80)

    extreme_valid = extreme_prob[vi]
    actual_dir_valid = np.sign(te_rets[vi])
    pred_dir_valid = pred_dir_ce[vi]
    usable_valid = (pred_dir_valid != 0) & (actual_dir_valid != 0) & (np.abs(te_rets[vi]) > MIN_RETURN_THRESHOLD)

    decile_results = []
    for d in range(10):
        lo = np.percentile(extreme_valid, d * 10)
        hi = np.percentile(extreme_valid, (d + 1) * 10)
        if d == 9:
            mask = (extreme_valid >= lo) & usable_valid
        else:
            mask = (extreme_valid >= lo) & (extreme_valid < hi) & usable_valid

        if mask.sum() == 0:
            continue

        acc = float((pred_dir_valid[mask] == actual_dir_valid[mask]).mean())
        avg_mag = float(np.abs(te_rets[vi][mask]).mean())
        avg_sig = float(extreme_valid[mask].mean())
        n = int(mask.sum())

        decile_results.append({
            "decile": d,
            "accuracy": acc,
            "n_samples": n,
            "avg_actual_magnitude": avg_mag,
            "avg_extreme_prob": avg_sig,
        })

        logger.info("  D%d: acc=%.1f%% avg|ret|=%.3f%% avg_extreme_prob=%.3f (n=%d)",
                    d, acc * 100, avg_mag * 100, avg_sig, n)

    # Save all results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if obj == float("inf"):
            return "inf"
        return obj

    output = {
        "correlations_pearson": _clean(correlations),
        "correlations_spearman": _clean(spearman_results),
        "strategies": _clean(strategies),
        "magnitude_deciles": _clean(decile_results),
        "meta": {
            "n_test": int(te_valid.sum()),
            "cost_bps": COST_BPS,
            "seed": SEED,
            "extreme_prob_thresholds": {
                "median": float(extreme_median),
                "p33": float(extreme_p33),
                "p67": float(extreme_p67),
            },
        },
    }

    out_path = RESULTS_DIR / "magnitude_pnl_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("\nResults saved to %s", out_path)

    return output


if __name__ == "__main__":
    run_magnitude_sim()
