#!/usr/bin/env python3
"""Phase 10 Step 1: Per-ticker/sector/level breakdown of D1 (Delta+Aux) results.

Trains D1 on all tickers combined (cross-ticker, quick mode), then evaluates
per-ticker, per-sector, and per-level to identify where the signal lives.

Usage:
    python scripts/d1_ticker_breakdown.py
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wavecast.core.universe import DEFAULT_UNIVERSE
from wavecast.data.auxiliary_features import (
    N_AUX_FEATURES,
    compute_detail_auxiliary_features,
)
from wavecast.data.cache import ParquetCache
from wavecast.data.continuous_dataset import build_continuous_dataset
from wavecast.experiments.runner import SECTOR_ID_MAP
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
)
from wavecast.wavelets.dwt import decompose

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Config ---
TICKERS = [a.ticker for a in DEFAULT_UNIVERSE.assets]
TICKER_SECTOR = {a.ticker: a.sector.value for a in DEFAULT_UNIVERSE.assets if a.sector}
SECTOR_ORDER = ["tech", "finance", "energy", "healthcare", "broad_etf", "commodity_etf"]
INTERVAL = "1h"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
DETAIL_LEVELS = [1, 2, 5]
CONTEXT_LENGTH = 16
N_CLASSES = 5
PERCENTILES = [10.0, 30.0, 70.0, 90.0]
MIN_RETURN_THRESHOLD = 0.001
COST_BPS = 7.0
N_RANDOM_TRIALS = 500  # higher than aggregate (100) because per-ticker N is smaller
MIN_RELIABLE_N = 1000  # flag results below this as unreliable


@dataclass
class GroupResult:
    name: str
    n_test: int
    n_valid: int
    econ_dir_accuracy: float
    econ_dir_inverse: float  # accuracy if we flip predictions
    sharpe_raw: float
    sharpe_with_costs: float
    sharpe_percentile: float
    transition_accuracy: float
    quantile_accuracy: float
    reliable: bool  # True if n_valid >= MIN_RELIABLE_N


# --- Data Loading ---


def load_and_split_prices(cache):
    from wavecast.core.types import TimeSeries

    train_prices = {}
    test_prices = {}
    for ticker in TICKERS:
        ts = cache.get(ticker, INTERVAL)
        if ts is None:
            continue
        train_mask = ts.timestamps <= np.datetime64(TRAIN_END)
        test_mask = ts.timestamps >= np.datetime64(TEST_START)
        if train_mask.sum() > 100 and test_mask.sum() > 50:
            train_prices[ticker] = TimeSeries(
                values=ts.values[train_mask], timestamps=ts.timestamps[train_mask],
                ticker=ticker, interval=INTERVAL,
            )
            test_prices[ticker] = TimeSeries(
                values=ts.values[test_mask], timestamps=ts.timestamps[test_mask],
                ticker=ticker, interval=INTERVAL,
            )
    common = sorted(set(train_prices) & set(test_prices))
    return train_prices, test_prices, common


def build_asset_class_map():
    ac_map = {}
    for a in DEFAULT_UNIVERSE.assets:
        if a.sector is not None:
            ac_map[a.ticker] = SECTOR_ID_MAP.get(a.sector.value, 0)
    return ac_map


# --- Coefficient + Aux Extractors ---


def detail_delta_coeffs(decomp, lvl):
    c = decomp.detail_at_level(lvl)
    return np.diff(c) if len(c) > 1 else c


def detail_delta_aux_fn(decomp, lvl):
    detail = decomp.detail_at_level(lvl)
    approx = decomp.approximation
    return compute_detail_auxiliary_features(detail, approx)


# --- Evaluation ---


def evaluate_group(
    name: str,
    pred_labels: NDArray,
    actual_returns: NDArray,
    actual_labels: NDArray,
    valid_mask: NDArray,
) -> GroupResult:
    """Full evaluation on a group (ticker, sector, or level)."""
    n_test = len(pred_labels)
    n_valid = int(valid_mask.sum())
    mid = N_CLASSES // 2

    pred_dir = np.zeros(n_test, dtype=np.float64)
    pred_dir[pred_labels > mid] = 1.0
    pred_dir[pred_labels < mid] = -1.0

    actual_dir = np.sign(actual_returns)

    # --- Economic directional accuracy (filtered) ---
    filt = valid_mask & (np.abs(actual_returns) > MIN_RETURN_THRESHOLD)
    if filt.sum() > 0:
        has_pred = pred_dir[filt] != 0
        if has_pred.sum() > 0:
            econ_dir_acc = float(np.mean(pred_dir[filt][has_pred] == actual_dir[filt][has_pred]))
            # Inverse accuracy — flip all predictions
            econ_dir_inv = float(np.mean(-pred_dir[filt][has_pred] == actual_dir[filt][has_pred]))
        else:
            econ_dir_acc = 0.5
            econ_dir_inv = 0.5
    else:
        econ_dir_acc = 0.5
        econ_dir_inv = 0.5

    # --- Quantile accuracy ---
    q_acc = float(np.mean(pred_labels[valid_mask] == actual_labels[valid_mask])) if n_valid > 0 else 0.0

    # --- Sharpe ---
    if n_valid > 0:
        pnl_raw = pred_dir[valid_mask] * actual_returns[valid_mask]
        pnl_raw = pnl_raw[~np.isnan(pnl_raw)]
        dir_changes = np.abs(np.diff(pred_dir[valid_mask]))
        cost_per_step = np.zeros(n_valid)
        cost_per_step[1:] = dir_changes * (COST_BPS / 10000)
        cost_per_step[0] = abs(pred_dir[valid_mask][0]) * (COST_BPS / 10000)
        pnl_net = pred_dir[valid_mask] * actual_returns[valid_mask] - cost_per_step
        pnl_net = pnl_net[~np.isnan(pnl_net)]

        sharpe_raw = (
            float(np.mean(pnl_raw) / np.std(pnl_raw) * np.sqrt(252 * 7))
            if len(pnl_raw) > 1 and np.std(pnl_raw) > 0 else 0.0
        )
        sharpe_costs = (
            float(np.mean(pnl_net) / np.std(pnl_net) * np.sqrt(252 * 7))
            if len(pnl_net) > 1 and np.std(pnl_net) > 0 else 0.0
        )
    else:
        sharpe_raw = 0.0
        sharpe_costs = 0.0

    # --- Sharpe vs random ---
    rng = np.random.default_rng(42)
    random_sharpes = []
    if n_valid > 0:
        valid_rets = actual_returns[valid_mask]
        valid_rets_clean = valid_rets[~np.isnan(valid_rets)]
        for _ in range(N_RANDOM_TRIALS):
            rand_dir = rng.choice([-1.0, 0.0, 1.0], size=len(valid_rets_clean))
            rand_pnl = rand_dir * valid_rets_clean
            if len(rand_pnl) > 1 and np.std(rand_pnl) > 0:
                random_sharpes.append(
                    float(np.mean(rand_pnl) / np.std(rand_pnl) * np.sqrt(252 * 7))
                )
    sharpe_pctile = (
        float(np.mean(np.array(random_sharpes) < sharpe_costs) * 100)
        if random_sharpes else 50.0
    )

    # --- Transition accuracy ---
    if n_valid > 10:
        valid_idx = np.where(valid_mask)[0]
        actual_dir_valid = actual_dir[valid_idx]
        pred_dir_valid = pred_dir[valid_idx]
        transitions = np.where(np.diff(np.sign(actual_dir_valid)) != 0)[0] + 1
        if len(transitions) > 0:
            trans_acc = float(np.mean(pred_dir_valid[transitions] == actual_dir_valid[transitions]))
        else:
            trans_acc = 0.5
    else:
        trans_acc = 0.5

    return GroupResult(
        name=name,
        n_test=n_test,
        n_valid=n_valid,
        econ_dir_accuracy=econ_dir_acc,
        econ_dir_inverse=econ_dir_inv,
        sharpe_raw=sharpe_raw,
        sharpe_with_costs=sharpe_costs,
        sharpe_percentile=sharpe_pctile,
        transition_accuracy=trans_acc,
        quantile_accuracy=q_acc,
        reliable=n_valid >= MIN_RELIABLE_N,
    )


def verdict(r: GroupResult) -> str:
    """Assign PASS/MARG/NOISE/HARMFUL/INVERSE verdict."""
    if not r.reliable:
        return "LOW-N"
    if r.econ_dir_accuracy > 0.55 and r.sharpe_with_costs > 0.5:
        return "PASS"
    if r.econ_dir_accuracy > 0.52:
        return "MARG"
    if r.econ_dir_accuracy >= 0.48:
        return "NOISE"
    # HARMFUL — check if inverse passes
    if r.econ_dir_inverse > 0.55:
        return "INVERSE"
    return "HARMFUL"


# --- Aux Window Builder ---


def build_aux_windows(aux_series, context_length, n_aux_features):
    """Build sliding windows from auxiliary feature series (sorted order)."""
    windows = []
    for (_ticker, _level), aux in sorted(aux_series.items()):
        if len(aux) <= context_length:
            continue
        for i in range(len(aux) - context_length):
            windows.append(aux[i:i + context_length])
    if not windows:
        return np.empty((0, context_length, n_aux_features), dtype=np.float64)
    return np.array(windows, dtype=np.float64)


# --- Main ---


def main() -> None:
    cache = ParquetCache(Path.home() / ".wavecast" / "cache")
    train_prices, test_prices, tickers = load_and_split_prices(cache)
    logger.info("Loaded %d tickers", len(tickers))
    ac_map = build_asset_class_map()

    # --- Build D1 datasets ---
    train_series: dict[tuple[str, int], NDArray] = {}
    test_series: dict[tuple[str, int], NDArray] = {}
    train_aux_series: dict[tuple[str, int], NDArray] = {}
    test_aux_series: dict[tuple[str, int], NDArray] = {}

    for ticker in tickers:
        train_decomp = decompose(train_prices[ticker], level=5)
        test_decomp = decompose(test_prices[ticker], level=5)
        for lvl in DETAIL_LEVELS:
            tc = detail_delta_coeffs(train_decomp, lvl)
            if len(tc) > CONTEXT_LENGTH:
                train_series[(ticker, lvl)] = tc
                train_aux_series[(ticker, lvl)] = detail_delta_aux_fn(train_decomp, lvl)
            ec = detail_delta_coeffs(test_decomp, lvl)
            if len(ec) > CONTEXT_LENGTH:
                test_series[(ticker, lvl)] = ec
                test_aux_series[(ticker, lvl)] = detail_delta_aux_fn(test_decomp, lvl)

    train_ds = build_continuous_dataset(train_series, CONTEXT_LENGTH, ac_map, normalize=True)
    test_ds = build_continuous_dataset(test_series, CONTEXT_LENGTH, ac_map, normalize=True)
    logger.info("Train: %d windows, Test: %d windows", len(train_ds.windows), len(test_ds.windows))

    # --- Compute returns ---
    def compute_window_returns(ds, prices_dict, series_dict):
        rets = np.full(len(ds.windows), np.nan)
        valid = np.zeros(len(ds.windows), dtype=np.bool_)
        for i, w in enumerate(ds.windows):
            key = (w.ticker, w.level)
            if key not in series_dict or w.ticker not in prices_dict:
                continue
            coeffs_len = len(series_dict[key])
            target_pos = w.token_position
            if target_pos >= coeffs_len:
                continue
            span = 2 ** w.level
            bar_start = target_pos * span
            bar_end = bar_start + span
            pv = prices_dict[w.ticker].values
            if bar_end < len(pv) and pv[bar_start] > 0:
                rets[i] = (pv[bar_end] - pv[bar_start]) / pv[bar_start]
                valid[i] = True
        return rets, valid

    train_rets, train_valid = compute_window_returns(train_ds, train_prices, train_series)
    test_rets, test_valid = compute_window_returns(test_ds, test_prices, test_series)
    train_lvls = np.array([w.level for w in train_ds.windows], dtype=np.int64)
    test_lvls = np.array([w.level for w in test_ds.windows], dtype=np.int64)

    # --- Quantile boundaries + labels ---
    boundaries = compute_quantile_boundaries(train_rets, train_lvls, train_valid, PERCENTILES, per_level=True)
    y_train = assign_quantile_labels(train_rets, train_lvls, boundaries)
    y_test = assign_quantile_labels(test_rets, test_lvls, boundaries)

    # --- Class weights ---
    valid_train_labels = y_train[train_valid]
    counts = np.bincount(valid_train_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    # --- Build X arrays ---
    ctx_arr, lvl_arr, ac_arr = train_ds.to_arrays()
    ctx_arr_t, lvl_arr_t, ac_arr_t = test_ds.to_arrays()

    train_aux_win = build_aux_windows(train_aux_series, CONTEXT_LENGTH, N_AUX_FEATURES)
    test_aux_win = build_aux_windows(test_aux_series, CONTEXT_LENGTH, N_AUX_FEATURES)
    train_aux_flat = train_aux_win.reshape(len(train_aux_win), -1)
    test_aux_flat = test_aux_win.reshape(len(test_aux_win), -1)
    X_train = np.column_stack([ctx_arr, train_aux_flat, lvl_arr, ac_arr])
    X_test = np.column_stack([ctx_arr_t, test_aux_flat, lvl_arr_t, ac_arr_t])

    # --- Train D1 (quick mode) ---
    model_kwargs = {
        "embed_dim": 64, "num_heads": 4, "num_layers": 3,
        "dropout": 0.1, "epochs": 20, "batch_size": 64,
        "learning_rate": 0.0005, "patience": 10,
    }
    model = WaveletGPT(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=N_CLASSES,
        class_weights=class_weights, input_mode="continuous",
        n_aux_features=N_AUX_FEATURES, **model_kwargs,
    )
    logger.info("Training D1 (quick mode)...")
    t0 = time.time()
    metrics = model.fit(
        X_train, y_train.astype(np.float64),
        X_val=X_test, y_val=y_test.astype(np.float64),
    )
    train_time = time.time() - t0
    logger.info("Training done in %.1fs (loss=%.4f)", train_time, metrics["train_loss"])

    # --- Predict ---
    predicted = model.predict(X_test).astype(np.int64)

    # --- Build per-sample metadata ---
    test_tickers = np.array([w.ticker for w in test_ds.windows])
    test_levels = np.array([w.level for w in test_ds.windows], dtype=np.int64)
    test_sectors = np.array([TICKER_SECTOR.get(w.ticker, "unknown") for w in test_ds.windows])

    # --- Aggregate evaluation ---
    agg = evaluate_group("AGGREGATE", predicted, test_rets, y_test, test_valid)

    # --- Per-ticker evaluation ---
    ticker_results: list[GroupResult] = []
    for ticker in sorted(set(test_tickers)):
        mask = test_tickers == ticker
        idx = np.where(mask)[0]
        r = evaluate_group(
            ticker, predicted[idx], test_rets[idx], y_test[idx], test_valid[idx],
        )
        ticker_results.append(r)

    # --- Per-sector evaluation ---
    sector_results: list[GroupResult] = []
    for sector in SECTOR_ORDER:
        mask = test_sectors == sector
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        r = evaluate_group(
            sector, predicted[idx], test_rets[idx], y_test[idx], test_valid[idx],
        )
        sector_results.append(r)

    # --- Per-level evaluation ---
    level_results: list[GroupResult] = []
    for lvl in sorted(DETAIL_LEVELS):
        mask = test_levels == lvl
        idx = np.where(mask)[0]
        r = evaluate_group(
            f"level_{lvl}", predicted[idx], test_rets[idx], y_test[idx], test_valid[idx],
        )
        level_results.append(r)

    # --- Print tables ---
    w = 120
    print("\n" + "=" * w)
    print("D1 (DELTA+AUX) PER-TICKER BREAKDOWN")
    print("=" * w)
    print(
        f"{'Ticker':<7} {'Sector':<14} {'N_valid':>7} {'Econ Dir':>8} "
        f"{'Inverse':>8} {'S+Cost':>7} {'vs Rnd':>7} {'Trans':>6} {'Q.Acc':>6}  Verdict"
    )
    print("-" * w)

    for r in sorted(ticker_results, key=lambda x: x.econ_dir_accuracy, reverse=True):
        v = verdict(r)
        rel = "" if r.reliable else " *"
        print(
            f"{r.name:<7} {TICKER_SECTOR.get(r.name, '?'):<14} {r.n_valid:>7d} "
            f"{r.econ_dir_accuracy:>7.1%} {r.econ_dir_inverse:>8.1%} "
            f"{r.sharpe_with_costs:>+7.3f} {r.sharpe_percentile:>6.1f}% "
            f"{r.transition_accuracy:>5.1%} {r.quantile_accuracy:>5.1%}  {v}{rel}"
        )

    print("-" * w)
    print(
        f"{'AGGREGATE':<7} {'all':<14} {agg.n_valid:>7d} "
        f"{agg.econ_dir_accuracy:>7.1%} {agg.econ_dir_inverse:>8.1%} "
        f"{agg.sharpe_with_costs:>+7.3f} {agg.sharpe_percentile:>6.1f}% "
        f"{agg.transition_accuracy:>5.1%} {agg.quantile_accuracy:>5.1%}  {verdict(agg)}"
    )

    print(f"\n{'* N_valid < ' + str(MIN_RELIABLE_N) + ' — result may be unreliable':>60}")

    print("\n" + "=" * w)
    print("PER-SECTOR BREAKDOWN")
    print("=" * w)
    print(
        f"{'Sector':<14} {'N_valid':>7} {'Econ Dir':>8} {'Inverse':>8} "
        f"{'S+Cost':>7} {'vs Rnd':>7} {'Trans':>6} {'Q.Acc':>6}  Verdict"
    )
    print("-" * w)
    for r in sector_results:
        v = verdict(r)
        rel = "" if r.reliable else " *"
        print(
            f"{r.name:<14} {r.n_valid:>7d} {r.econ_dir_accuracy:>7.1%} "
            f"{r.econ_dir_inverse:>8.1%} {r.sharpe_with_costs:>+7.3f} "
            f"{r.sharpe_percentile:>6.1f}% {r.transition_accuracy:>5.1%} "
            f"{r.quantile_accuracy:>5.1%}  {v}{rel}"
        )

    print("\n" + "=" * w)
    print("PER-LEVEL BREAKDOWN")
    print("=" * w)
    print(
        f"{'Level':<10} {'N_valid':>7} {'Econ Dir':>8} {'Inverse':>8} "
        f"{'S+Cost':>7} {'vs Rnd':>7} {'Trans':>6} {'Q.Acc':>6}  Verdict"
    )
    print("-" * w)
    for r in level_results:
        v = verdict(r)
        rel = "" if r.reliable else " *"
        print(
            f"{r.name:<10} {r.n_valid:>7d} {r.econ_dir_accuracy:>7.1%} "
            f"{r.econ_dir_inverse:>8.1%} {r.sharpe_with_costs:>+7.3f} "
            f"{r.sharpe_percentile:>6.1f}% {r.transition_accuracy:>5.1%} "
            f"{r.quantile_accuracy:>5.1%}  {v}{rel}"
        )

    print("=" * w)

    # --- Decision summary ---
    passing = [r for r in ticker_results if verdict(r) == "PASS"]
    marginal = [r for r in ticker_results if verdict(r) == "MARG"]
    inverse = [r for r in ticker_results if verdict(r) == "INVERSE"]
    noise = [r for r in ticker_results if verdict(r) == "NOISE"]
    harmful = [r for r in ticker_results if verdict(r) == "HARMFUL"]
    low_n = [r for r in ticker_results if verdict(r) == "LOW-N"]

    print("\nDECISION SUMMARY")
    print(f"  PASS ({len(passing)}): {', '.join(r.name for r in passing) or 'none'}")
    print(f"  MARGINAL ({len(marginal)}): {', '.join(r.name for r in marginal) or 'none'}")
    print(f"  INVERSE ({len(inverse)}): {', '.join(r.name for r in inverse) or 'none'}")
    print(f"  NOISE ({len(noise)}): {', '.join(r.name for r in noise) or 'none'}")
    print(f"  HARMFUL ({len(harmful)}): {', '.join(r.name for r in harmful) or 'none'}")
    if low_n:
        print(f"  LOW-N ({len(low_n)}): {', '.join(r.name for r in low_n) or 'none'}")

    forward_tickers = [r.name for r in passing]
    if inverse:
        print(f"\n  INVERSE tickers (flip signal for forward test): {', '.join(r.name for r in inverse)}")
    if len(passing) < 5:
        print(f"\n  WARNING: Only {len(passing)} tickers pass — signal may be too concentrated")
    print(f"\n  FORWARD TEST CANDIDATES: {', '.join(forward_tickers) or 'none'}")

    # --- Save JSON ---
    results_dir = Path.home() / ".wavecast" / "audit"
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "aggregate": asdict(agg),
        "per_ticker": [asdict(r) for r in ticker_results],
        "per_sector": [asdict(r) for r in sector_results],
        "per_level": [asdict(r) for r in level_results],
        "verdicts": {r.name: verdict(r) for r in ticker_results},
        "forward_test_candidates": forward_tickers,
        "inverse_candidates": [r.name for r in inverse],
        "config": {
            "model_kwargs": model_kwargs,
            "train_time": train_time,
            "train_loss": metrics["train_loss"],
            "n_random_trials": N_RANDOM_TRIALS,
            "min_reliable_n": MIN_RELIABLE_N,
            "cost_bps": COST_BPS,
            "min_return_threshold": MIN_RETURN_THRESHOLD,
        },
    }
    out_path = results_dir / "d1_ticker_breakdown.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
