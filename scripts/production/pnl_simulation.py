"""Production PnL simulation: reversal-filtered magnitude-weighted trading.

Trains a CE baseline for 1 seed, saves full prediction arrays, then
simulates PnL across 8 trading configurations:

  Axis 1 (trade filter): ALL vs REVERSAL-ONLY
  Axis 2 (magnitude filter): ALL vs LARGE-ONLY (top tercile by predicted magnitude)
  Axis 3 (position sizing): FLAT vs MAGNITUDE-WEIGHTED

Also trains a regression model for Signal D (|predicted_return| as magnitude).

Usage:
    cd ~/wavecast && source .venv/bin/activate
    python -u -m scripts.production.pnl_simulation
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
from scipy.stats import spearmanr
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

SEED = 42
MID = N_CLASSES // 2
RESULTS_DIR = Path.home() / ".wavecast" / "audit" / "production"


# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------


def _generate_predictions(
    X_train: NDArray, y_train: NDArray, tr_rets: NDArray, tr_valid: NDArray,
    X_test: NDArray, te_rets: NDArray, te_valid: NDArray,
    windows: list,
) -> dict[str, NDArray]:
    """Train CE baseline + regression, return full prediction arrays."""

    class_weights = _compute_class_weights(y_train, tr_valid)

    # --- CE baseline ---
    logger.info("Training CE baseline (seed=%d)...", SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

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

    # --- Regression (for Signal D) ---
    logger.info("Training regression (seed=%d)...", SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    valid_tr_rets = tr_rets[tr_valid]
    ret_mean = float(np.mean(valid_tr_rets))
    ret_std = float(np.std(valid_tr_rets))
    y_train_reg = np.where(
        tr_valid, (tr_rets - ret_mean) / max(ret_std, 1e-10), 0.0,
    ).astype(np.float64)

    model_reg = WaveletGPT(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_regression", input_mode="continuous",
        n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
    )
    t0 = time.time()
    model_reg.fit(X_train, y_train_reg)
    logger.info("  Regression trained in %.1fs", time.time() - t0)
    reg_raw = model_reg.predict(X_test)
    reg_returns = reg_raw * ret_std + ret_mean

    del model_reg
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Compute bin midpoints from training data ---
    boundaries_lvl1 = compute_quantile_boundaries(
        tr_rets, np.ones(len(tr_rets), dtype=np.int64),
        tr_valid, list(PERCENTILES), per_level=True,
    )
    bounds = boundaries_lvl1.get(1, np.array(PERCENTILES))
    bin_midpoints = _compute_bin_midpoints(tr_rets[tr_valid], bounds)
    logger.info("  Bin midpoints: %s", [f"{m:.5f}" for m in bin_midpoints])

    # --- Build per-sample recent_returns ---
    recent_returns = _build_recent_returns(windows, te_rets, max_lookback=5)

    # --- Predicted direction from CE ---
    pred_dir = np.zeros(len(pred_labels), dtype=np.float64)
    pred_dir[pred_labels > MID] = 1.0
    pred_dir[pred_labels < MID] = -1.0

    return {
        "pred_labels": pred_labels,
        "pred_dir": pred_dir,
        "pred_proba": pred_proba,
        "actual_returns": te_rets,
        "actual_dir": np.sign(te_rets),
        "valid": te_valid,
        "recent_returns": recent_returns,
        "reg_returns": reg_returns,
        "bin_midpoints": bin_midpoints,
        "ret_mean": ret_mean,
        "ret_std": ret_std,
    }


def _compute_class_weights(y_train: NDArray, valid: NDArray) -> list[float]:
    valid_labels = y_train[valid]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    return (inv_freq / inv_freq.sum() * N_CLASSES).tolist()


def _compute_bin_midpoints(returns: NDArray, boundaries: NDArray) -> NDArray:
    """Compute mean return within each quantile bin."""
    n_bins = len(boundaries) + 1
    midpoints = np.zeros(n_bins)
    for b in range(n_bins):
        if b == 0:
            mask = returns < boundaries[0]
        elif b == n_bins - 1:
            mask = returns >= boundaries[-1]
        else:
            mask = (returns >= boundaries[b - 1]) & (returns < boundaries[b])
        midpoints[b] = float(np.mean(returns[mask])) if mask.sum() > 0 else 0.0
    return midpoints


def _build_recent_returns(
    windows: list, returns: NDArray, max_lookback: int = 5,
) -> NDArray:
    """Build (N, max_lookback) array of recent returns per sample.

    Windows within a (ticker, level) series are contiguous and sequential.
    At series boundaries, pad with NaN.
    """
    n = len(windows)
    recent = np.full((n, max_lookback), np.nan, dtype=np.float64)

    # Identify series boundaries: where (ticker, level) changes
    series_start = np.zeros(n, dtype=np.int64)
    prev_key = None
    for i, w in enumerate(windows):
        key = (w.ticker, w.level)
        if key != prev_key:
            series_start[i] = i
            prev_key = key
        else:
            series_start[i] = series_start[i - 1]

    for i in range(n):
        start = int(series_start[i])
        for lb in range(max_lookback):
            j = i - lb - 1
            if j >= start:
                recent[i, lb] = returns[j]

    return recent


# ---------------------------------------------------------------------------
# Magnitude signals
# ---------------------------------------------------------------------------


def compute_magnitude_signals(
    pred_proba: NDArray, bin_midpoints: NDArray, reg_returns: NDArray,
) -> dict[str, NDArray]:
    """Compute all candidate magnitude signals."""
    n = len(pred_proba)

    # Signal A: extreme-bin probability
    sig_a = pred_proba[:, 0] + pred_proba[:, -1]

    # Signal B: expected absolute return
    abs_midpoints = np.abs(bin_midpoints)
    sig_b = pred_proba @ abs_midpoints

    # Signal C: negative entropy (higher = more confident)
    entropy = -np.sum(pred_proba * np.log(pred_proba + 1e-10), axis=1)
    sig_c = -entropy

    # Signal D: regression |predicted return|
    sig_d = np.abs(reg_returns)

    return {"A_extreme_bin": sig_a, "B_expected_abs": sig_b,
            "C_neg_entropy": sig_c, "D_reg_abs": sig_d}


# ---------------------------------------------------------------------------
# Trading filter helpers
# ---------------------------------------------------------------------------


def is_reversal(pred_dir: float, recent_rets: NDArray, lookback: int) -> bool | None:
    """True if predicted direction disagrees with recent direction."""
    lb_rets = recent_rets[:lookback]
    valid_rets = lb_rets[~np.isnan(lb_rets)]
    if len(valid_rets) == 0:
        return None
    recent_dir = np.sign(np.mean(valid_rets))
    if recent_dir == 0:
        return None
    return pred_dir != recent_dir


# ---------------------------------------------------------------------------
# PnL simulation engine
# ---------------------------------------------------------------------------


def simulate_pnl(
    pred_dir: NDArray,
    actual_returns: NDArray,
    valid: NDArray,
    magnitude_signal: NDArray,
    recent_returns: NDArray,
    *,
    trade_filter: str = "all",           # "all" or "reversal_only"
    magnitude_filter: str = "all",       # "all" or "large_only"
    sizing_method: str = "flat",         # "flat" or "magnitude_weighted"
    lookback: int = 3,
    cost_bps: float = COST_BPS,
    mag_cap: float = 3.0,
) -> dict:
    """Run PnL simulation for one configuration."""
    n = len(pred_dir)
    vi = np.where(valid)[0]

    # Compute magnitude tercile thresholds on valid samples
    valid_mag = magnitude_signal[vi]
    mag_p67 = float(np.percentile(valid_mag, 66.7))
    mag_median = float(np.median(valid_mag))

    # Per-sample decisions
    take_trade = np.zeros(n, dtype=bool)
    position_size = np.zeros(n, dtype=np.float64)

    for idx in vi:
        pd_i = pred_dir[idx]
        if pd_i == 0:
            continue  # flat prediction → no trade

        # Trade filter
        if trade_filter == "reversal_only":
            rev = is_reversal(pd_i, recent_returns[idx], lookback)
            if rev is None or not rev:
                continue

        # Magnitude filter
        if magnitude_filter == "large_only":
            if magnitude_signal[idx] < mag_p67:
                continue

        take_trade[idx] = True

        # Position sizing
        if sizing_method == "flat":
            position_size[idx] = pd_i  # ±1
        else:
            # Linear: size = signal / median, capped
            raw = magnitude_signal[idx] / max(mag_median, 1e-10)
            size = float(np.clip(raw, 0.5, mag_cap))
            position_size[idx] = pd_i * size

    # Compute PnL on traded samples
    traded_idx = np.where(take_trade)[0]
    n_trades = len(traded_idx)

    if n_trades == 0:
        return _empty_result()

    pos = position_size[traded_idx]
    rets = actual_returns[traded_idx]
    a_dir = np.sign(rets)
    p_dir = np.sign(pos)

    # Transaction costs: on absolute position change
    cost = np.zeros(n_trades)
    cost[0] = abs(pos[0]) * (cost_bps / 10000)
    if n_trades > 1:
        cost[1:] = np.abs(np.diff(pos)) * (cost_bps / 10000)

    pnl_gross = pos * rets
    pnl_net = pnl_gross - cost

    # Filter NaN
    clean = ~np.isnan(pnl_net)
    pnl_net = pnl_net[clean]
    pnl_gross = pnl_gross[clean]
    pos_clean = pos[clean]
    rets_clean = rets[clean]
    a_dir_clean = a_dir[clean]
    p_dir_clean = p_dir[clean]

    if len(pnl_net) == 0:
        return _empty_result()

    # Equity curve
    equity = np.cumprod(1 + pnl_net)

    # Accuracy on trades taken
    directional = (p_dir_clean != 0) & (a_dir_clean != 0)
    if directional.sum() > 0:
        accuracy = float((p_dir_clean[directional] == a_dir_clean[directional]).mean())
    else:
        accuracy = 0.5

    # Win/loss
    wins = pnl_net > 0
    losses = pnl_net < 0
    win_rate = float(wins.mean())
    avg_win = float(pnl_net[wins].mean()) if wins.sum() > 0 else 0.0
    avg_loss = float(np.abs(pnl_net[losses]).mean()) if losses.sum() > 0 else 0.0

    # Sharpe (annualized, 7 trades/day × 252 days)
    sharpe = _sharpe(pnl_net)

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(np.min(dd))

    # Per-trade expectancy
    expectancy = float(np.mean(pnl_net))

    return {
        "n_trades": int(directional.sum()),
        "accuracy": accuracy,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "expectancy_per_trade": expectancy,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "wl_ratio": avg_win / avg_loss if avg_loss > 0 else float("inf"),
        "cum_pnl_net": float(pnl_net.sum()),
        "cum_pnl_gross": float(pnl_gross.sum()),
        "total_cost": float(cost[clean].sum()),
        "avg_position_size": float(np.abs(pos_clean).mean()),
        "equity_final": float(equity[-1]) if len(equity) > 0 else 1.0,
    }


def _sharpe(pnl: NDArray, annual_factor: float = np.sqrt(252 * 7)) -> float:
    if len(pnl) < 2 or np.std(pnl) == 0:
        return 0.0
    return float(np.mean(pnl) / np.std(pnl) * annual_factor)


def _empty_result() -> dict:
    return {
        "n_trades": 0, "accuracy": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
        "expectancy_per_trade": 0.0, "win_rate": 0.0, "avg_win": 0.0,
        "avg_loss": 0.0, "wl_ratio": 0.0, "cum_pnl_net": 0.0,
        "cum_pnl_gross": 0.0, "total_cost": 0.0, "avg_position_size": 0.0,
        "equity_final": 1.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info("=" * 90)
    logger.info("PRODUCTION PnL SIMULATION")
    logger.info("=" * 90)

    # --- Load data ---
    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)
    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))
    logger.info("Loaded %d tickers", len(tickers))

    # --- Build D1 dataset ---
    from wavecast.data.continuous_dataset import build_continuous_dataset
    from wavecast.data.auxiliary_features import compute_detail_auxiliary_features
    from wavecast.experiments.runner import SECTOR_ID_MAP
    from wavecast.core.universe import DEFAULT_UNIVERSE
    from wavecast.wavelets.dwt import decompose
    from scripts.feature_tests.harness import DETAIL_LEVELS, _build_d1_pipeline

    X_train, tr_rets, tr_valid, tr_lvls, _, _ = _build_d1_pipeline(
        train_prices, None, tickers, None, 0,
    )
    X_test, te_rets, te_valid, te_lvls, _, _ = _build_d1_pipeline(
        test_prices, None, tickers, None, 0,
    )

    # We also need the windows list for series boundary tracking
    ac_map = {
        a.ticker: SECTOR_ID_MAP.get(a.sector.value, 0)
        for a in DEFAULT_UNIVERSE.assets if a.sector
    }
    coeff_series: dict[tuple[str, int], NDArray] = {}
    for ticker in tickers:
        ts = test_prices[ticker]
        dec = decompose(ts, level=5)
        for lvl in DETAIL_LEVELS:
            detail = dec.detail_at_level(lvl)
            deltas = np.diff(detail) if len(detail) > 1 else detail
            if len(deltas) > CONTEXT_LENGTH:
                coeff_series[(ticker, lvl)] = deltas
    test_ds = build_continuous_dataset(coeff_series, CONTEXT_LENGTH, ac_map, normalize=True)
    windows = test_ds.windows
    logger.info("Dataset: %d train, %d test (%d valid)",
                len(X_train), len(X_test), int(te_valid.sum()))

    # --- Quantile labels for training ---
    boundaries = compute_quantile_boundaries(
        tr_rets, tr_lvls, tr_valid, list(PERCENTILES), per_level=True,
    )
    y_train = assign_quantile_labels(tr_rets, tr_lvls, boundaries)

    # --- Generate predictions ---
    preds = _generate_predictions(
        X_train, y_train, tr_rets, tr_valid, X_test, te_rets, te_valid, windows,
    )

    # --- Save raw predictions ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = RESULTS_DIR / "ce_baseline_predictions.npz"
    np.savez_compressed(
        pred_path,
        pred_labels=preds["pred_labels"],
        pred_dir=preds["pred_dir"],
        pred_proba=preds["pred_proba"],
        actual_returns=preds["actual_returns"],
        actual_dir=preds["actual_dir"],
        valid=preds["valid"],
        recent_returns=preds["recent_returns"],
        reg_returns=preds["reg_returns"],
        bin_midpoints=preds["bin_midpoints"],
    )
    logger.info("Saved raw predictions to %s", pred_path)

    # --- Magnitude signal correlation ---
    signals = compute_magnitude_signals(
        preds["pred_proba"], preds["bin_midpoints"], preds["reg_returns"],
    )

    vi = np.where(preds["valid"])[0]
    actual_mag = np.abs(preds["actual_returns"][vi])

    print("\n" + "=" * 90)
    print("MAGNITUDE SIGNAL CORRELATION WITH ACTUAL |RETURN|")
    print("=" * 90)

    correlations = {}
    best_signal_name = None
    best_rho = -1.0
    for name, sig in signals.items():
        sig_valid = sig[vi]
        r_pearson = float(np.corrcoef(sig_valid, actual_mag)[0, 1])
        rho, pval = spearmanr(sig_valid, actual_mag)
        correlations[name] = {
            "pearson": r_pearson, "spearman_rho": float(rho), "p_value": float(pval),
        }
        print(f"  Signal {name:<20s}: Pearson r={r_pearson:+.4f}  "
              f"Spearman rho={rho:+.4f}  (p={pval:.2e})")
        if float(rho) > best_rho:
            best_rho = float(rho)
            best_signal_name = name

    print(f"\n  --> Best signal: {best_signal_name} (rho={best_rho:+.4f})")
    best_signal = signals[best_signal_name]

    # --- 8-config PnL simulation ---
    configs = [
        ("A1i",  "all",           "all",        "flat"),
        ("A1ii", "all",           "all",        "magnitude_weighted"),
        ("A2i",  "all",           "large_only", "flat"),
        ("A2ii", "all",           "large_only", "magnitude_weighted"),
        ("B1i",  "reversal_only", "all",        "flat"),
        ("B1ii", "reversal_only", "all",        "magnitude_weighted"),
        ("B2i",  "reversal_only", "large_only", "flat"),
        ("B2ii", "reversal_only", "large_only", "magnitude_weighted"),
    ]

    print("\n" + "=" * 90)
    print("TRADING SYSTEM COMPARISON (lookback=3)")
    print("=" * 90)
    header = (f"  {'Config':<7} {'Filter':<10} {'MagFilt':<12} {'Sizing':<10} "
              f"{'Trades':>7} {'Acc%':>7} {'Sharpe':>8} {'MaxDD':>8} "
              f"{'Expect':>9} {'WinR':>6} {'W/L':>6} {'PnL':>9}")
    print(header)
    print("  " + "-" * 106)

    all_results = {}
    for label, trade_f, mag_f, sizing in configs:
        r = simulate_pnl(
            preds["pred_dir"], preds["actual_returns"], preds["valid"],
            best_signal, preds["recent_returns"],
            trade_filter=trade_f, magnitude_filter=mag_f,
            sizing_method=sizing, lookback=3,
        )
        all_results[label] = r
        print(f"  {label:<7} {trade_f:<10} {mag_f:<12} {sizing:<10} "
              f"{r['n_trades']:>7,} {r['accuracy']:>6.1%} {r['sharpe']:>+7.3f} "
              f"{r['max_drawdown']:>+7.2%} {r['expectancy_per_trade']:>+8.4%} "
              f"{r['win_rate']:>5.1%} {r['wl_ratio']:>5.2f} "
              f"{r['cum_pnl_net']:>+8.2f}")

    # --- Lookback sensitivity on reversal configs ---
    print("\n" + "=" * 90)
    print("REVERSAL LOOKBACK SENSITIVITY")
    print("=" * 90)

    # Test on each reversal config
    lookback_results = {}
    for base_label, trade_f, mag_f, sizing in configs:
        if trade_f != "reversal_only":
            continue
        print(f"\n  {base_label} ({mag_f}, {sizing}):")
        lb_results = {}
        for lb in [2, 3, 5]:
            r = simulate_pnl(
                preds["pred_dir"], preds["actual_returns"], preds["valid"],
                best_signal, preds["recent_returns"],
                trade_filter="reversal_only", magnitude_filter=mag_f,
                sizing_method=sizing, lookback=lb,
            )
            lb_results[lb] = r
            print(f"    lookback={lb}: Trades={r['n_trades']:>6,}  "
                  f"Acc={r['accuracy']:>6.1%}  Sharpe={r['sharpe']:>+7.3f}  "
                  f"Expect={r['expectancy_per_trade']:>+.4%}  PnL={r['cum_pnl_net']:>+.2f}")
        lookback_results[base_label] = lb_results

    # --- Find best config ---
    # Rank by Sharpe, require >100 trades
    viable = {k: v for k, v in all_results.items() if v["n_trades"] > 100}
    if viable:
        best_config = max(viable, key=lambda k: viable[k]["sharpe"])
    else:
        best_config = max(all_results, key=lambda k: all_results[k]["sharpe"])
    best = all_results[best_config]

    # Check if any lookback improves the best reversal config
    best_lb = 3
    if best_config in lookback_results:
        for lb, r in lookback_results[best_config].items():
            if r["sharpe"] > best["sharpe"] and r["n_trades"] > 100:
                best = r
                best_lb = lb

    print("\n" + "=" * 90)
    print(f"BEST CONFIGURATION: {best_config} (lookback={best_lb})")
    print("=" * 90)
    print(f"  Trades:              {best['n_trades']:,}")
    print(f"  Accuracy:            {best['accuracy']:.1%}")
    print(f"  Sharpe (+{COST_BPS:.0f}bps):     {best['sharpe']:+.3f}")
    print(f"  Max drawdown:        {best['max_drawdown']:+.2%}")
    print(f"  Per-trade expectancy:{best['expectancy_per_trade']:+.4%}")
    print(f"  Win rate:            {best['win_rate']:.1%}")
    print(f"  Avg win/loss ratio:  {best['wl_ratio']:.2f}")
    print(f"  Cumulative PnL:      {best['cum_pnl_net']:+.2f}")
    print(f"  Total costs:         {best['total_cost']:.4f}")
    print("=" * 90)

    # --- Save ---
    def _clean(obj):
        if isinstance(obj, dict):
            return {str(k): _clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if obj == float("inf"):
            return "inf"
        return obj

    output = {
        "best_config": best_config,
        "best_lookback": best_lb,
        "best_magnitude_signal": best_signal_name,
        "correlations": _clean(correlations),
        "configs": _clean(all_results),
        "lookback_sensitivity": _clean(lookback_results),
        "best_result": _clean(best),
        "meta": {
            "seed": SEED,
            "cost_bps": COST_BPS,
            "n_test": int(preds["valid"].sum()),
            "n_classes": N_CLASSES,
            "bin_midpoints": preds["bin_midpoints"].tolist(),
        },
    }

    out_path = RESULTS_DIR / "pnl_simulation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)

    return output


if __name__ == "__main__":
    main()
