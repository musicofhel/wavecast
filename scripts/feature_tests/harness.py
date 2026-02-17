"""Feature test harness: trains baseline vs challenger D1 across multiple seeds.

Feature tests select on 2025 data. Winners get validated on 2026 as a final gate.
DO NOT test features on 2026 — that's the holdout.

Usage from a feature test script:

    from scripts.feature_tests.harness import run_feature_test

    def my_feature_fn(ohlcv_df, detail_coeffs, approx_coeffs, level):
        # Return shape (len(detail_coeffs) - 1, n_new_features)
        ...

    run_feature_test("my_feature", my_feature_fn, n_new_features=3)
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from wavecast.core.types import TimeSeries
from wavecast.core.universe import DEFAULT_UNIVERSE
from wavecast.data.auxiliary_features import (
    N_AUX_FEATURES,
    compute_detail_auxiliary_features,
)
from wavecast.data.continuous_dataset import build_continuous_dataset
from wavecast.experiments.runner import SECTOR_ID_MAP
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
)
from wavecast.wavelets.dwt import decompose

logger = logging.getLogger(__name__)

# --- Config ---
TICKERS = [a.ticker for a in DEFAULT_UNIVERSE.assets]
INTERVAL = "1h"
TRAIN_END = "2024-06-30"
VAL_START = "2024-07-01"
VAL_END = "2024-12-31"
TEST_START = "2025-01-01"
DETAIL_LEVELS = [1]
CONTEXT_LENGTH = 16
N_CLASSES = 5
PERCENTILES = [10.0, 30.0, 70.0, 90.0]
MIN_RETURN_THRESHOLD = 0.001
COST_BPS = 7.0
N_RANDOM_TRIALS = 200

MODEL_KWARGS = {
    "embed_dim": 64, "num_heads": 4, "num_layers": 3,
    "dropout": 0.1, "epochs": 20, "batch_size": 64,
    "learning_rate": 0.0005, "patience": 10,
}

# Feature function type
FeatureFn = Callable[[pd.DataFrame, NDArray, NDArray, int], NDArray]


@dataclass
class EvalResult:
    econ_dir_accuracy: float
    sharpe_with_costs: float
    sharpe_percentile: float
    transition_accuracy: float
    quantile_accuracy: float
    n_valid: int
    train_loss: float
    train_time: float


# --- Data Loading ---


def _load_ohlcv() -> dict[str, pd.DataFrame]:
    """Load OHLCV parquets from cache (2021 to ~2026-01-01)."""
    cache_dir = Path.home() / ".wavecast" / "cache"
    ohlcv: dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        path = cache_dir / f"{ticker}_1h_ohlcv.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            # Ensure tz-naive timestamps for comparison
            if df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(None)
            ohlcv[ticker] = df
    return ohlcv


def _split_ohlcv(
    ohlcv: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Split OHLCV into train (<= TRAIN_END) and test (>= TEST_START)."""
    train_end = pd.Timestamp(TRAIN_END)
    test_start = pd.Timestamp(TEST_START)

    train: dict[str, pd.DataFrame] = {}
    test: dict[str, pd.DataFrame] = {}

    for ticker, df in ohlcv.items():
        tr = df[df["timestamp"] <= train_end]
        te = df[df["timestamp"] >= test_start]
        if len(tr) > 200 and len(te) > 50:
            train[ticker] = tr.reset_index(drop=True)
            test[ticker] = te.reset_index(drop=True)

    return train, test


def _ohlcv_to_timeseries(ohlcv: dict[str, pd.DataFrame]) -> dict[str, TimeSeries]:
    """Extract close prices as TimeSeries from OHLCV DataFrames."""
    prices: dict[str, TimeSeries] = {}
    for ticker, df in ohlcv.items():
        prices[ticker] = TimeSeries(
            values=df["close"].to_numpy(dtype=np.float64),
            timestamps=df["timestamp"].to_numpy(dtype="datetime64[ns]"),
            ticker=ticker, interval=INTERVAL,
        )
    return prices


# --- Pipeline ---


def _build_d1_pipeline(
    prices: dict[str, TimeSeries],
    ohlcv: dict[str, pd.DataFrame] | None,
    tickers: list[str],
    feature_fn: FeatureFn | None,
    n_new_features: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, dict[tuple[str, int], NDArray]]:
    """Build D1 dataset. Returns (X, returns, valid, levels, labels, series_dict)."""
    ac_map = {
        a.ticker: SECTOR_ID_MAP.get(a.sector.value, 0)
        for a in DEFAULT_UNIVERSE.assets if a.sector
    }

    coeff_series: dict[tuple[str, int], NDArray] = {}
    aux_series: dict[tuple[str, int], NDArray] = {}

    for ticker in tickers:
        ts = prices[ticker]
        decomp = decompose(ts, level=5)
        for lvl in DETAIL_LEVELS:
            detail = decomp.detail_at_level(lvl)
            deltas = np.diff(detail) if len(detail) > 1 else detail
            if len(deltas) <= CONTEXT_LENGTH:
                continue

            coeff_series[(ticker, lvl)] = deltas

            # Existing 4 aux features
            existing_aux = compute_detail_auxiliary_features(detail, decomp.approximation)

            if feature_fn is not None and ohlcv is not None and ticker in ohlcv:
                # Compute new features and concatenate
                new_feats = feature_fn(ohlcv[ticker], detail, decomp.approximation, lvl)
                n_deltas = len(deltas)
                # Ensure correct length
                if len(new_feats) != n_deltas:
                    # Resample via linear interpolation
                    resampled = np.zeros((n_deltas, new_feats.shape[1]), dtype=np.float64)
                    indices = np.linspace(0, len(new_feats) - 1, n_deltas)
                    for j in range(new_feats.shape[1]):
                        resampled[:, j] = np.interp(
                            indices, np.arange(len(new_feats)), new_feats[:, j]
                        )
                    new_feats = resampled
                aux_series[(ticker, lvl)] = np.concatenate([existing_aux, new_feats], axis=1)
            else:
                aux_series[(ticker, lvl)] = existing_aux

    if not coeff_series:
        return (np.empty((0,)), np.empty((0,)), np.empty((0,), dtype=np.bool_),
                np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), {})

    ds = build_continuous_dataset(coeff_series, CONTEXT_LENGTH, ac_map, normalize=True)

    # Returns
    rets = np.full(len(ds.windows), np.nan)
    valid = np.zeros(len(ds.windows), dtype=np.bool_)
    for i, w in enumerate(ds.windows):
        key = (w.ticker, w.level)
        if key not in coeff_series or w.ticker not in prices:
            continue
        target_pos = w.token_position
        if target_pos >= len(coeff_series[key]):
            continue
        span = 2 ** w.level
        bar_start = target_pos * span
        bar_end = bar_start + span
        pv = prices[w.ticker].values
        if bar_end < len(pv) and pv[bar_start] > 0:
            rets[i] = (pv[bar_end] - pv[bar_start]) / pv[bar_start]
            valid[i] = True

    lvls = np.array([w.level for w in ds.windows], dtype=np.int64)

    # Build X
    ctx_arr, lvl_arr, ac_arr = ds.to_arrays()
    total_aux = N_AUX_FEATURES + n_new_features if feature_fn else N_AUX_FEATURES
    aux_win = _build_aux_windows(aux_series, CONTEXT_LENGTH, total_aux)
    aux_flat = aux_win.reshape(len(aux_win), -1)
    X = np.column_stack([ctx_arr, aux_flat, lvl_arr, ac_arr])

    return X, rets, valid, lvls, np.empty(0), coeff_series


def _build_aux_windows(aux_series, context_length, n_features):
    windows = []
    for (_t, _l), aux in sorted(aux_series.items()):
        if len(aux) <= context_length:
            continue
        for i in range(len(aux) - context_length):
            windows.append(aux[i:i + context_length])
    if not windows:
        return np.empty((0, context_length, n_features), dtype=np.float64)
    return np.array(windows, dtype=np.float64)


# --- Evaluation ---


def _evaluate(
    pred_labels: NDArray, actual_returns: NDArray,
    actual_labels: NDArray, valid_mask: NDArray,
    train_loss: float, train_time: float,
) -> EvalResult:
    n_valid = int(valid_mask.sum())
    mid = N_CLASSES // 2

    pred_dir = np.zeros(len(pred_labels), dtype=np.float64)
    pred_dir[pred_labels > mid] = 1.0
    pred_dir[pred_labels < mid] = -1.0
    actual_dir = np.sign(actual_returns)

    filt = valid_mask & (np.abs(actual_returns) > MIN_RETURN_THRESHOLD)
    if filt.sum() > 0:
        has_pred = pred_dir[filt] != 0
        econ_dir = float(np.mean(pred_dir[filt][has_pred] == actual_dir[filt][has_pred])) if has_pred.sum() > 0 else 0.5
    else:
        econ_dir = 0.5

    q_acc = float(np.mean(pred_labels[valid_mask] == actual_labels[valid_mask])) if n_valid > 0 else 0.0

    if n_valid > 0:
        pnl_raw = pred_dir[valid_mask] * actual_returns[valid_mask]
        pnl_raw = pnl_raw[~np.isnan(pnl_raw)]
        dir_changes = np.abs(np.diff(pred_dir[valid_mask]))
        cost = np.zeros(n_valid)
        cost[1:] = dir_changes * (COST_BPS / 10000)
        cost[0] = abs(pred_dir[valid_mask][0]) * (COST_BPS / 10000)
        pnl_net = pred_dir[valid_mask] * actual_returns[valid_mask] - cost
        pnl_net = pnl_net[~np.isnan(pnl_net)]
        sharpe_costs = float(np.mean(pnl_net) / np.std(pnl_net) * np.sqrt(252 * 7)) if len(pnl_net) > 1 and np.std(pnl_net) > 0 else 0.0
    else:
        sharpe_costs = 0.0

    rng = np.random.default_rng(42)
    random_sharpes = []
    if n_valid > 0:
        vr = actual_returns[valid_mask][~np.isnan(actual_returns[valid_mask])]
        for _ in range(N_RANDOM_TRIALS):
            rd = rng.choice([-1.0, 0.0, 1.0], size=len(vr))
            rp = rd * vr
            if len(rp) > 1 and np.std(rp) > 0:
                random_sharpes.append(float(np.mean(rp) / np.std(rp) * np.sqrt(252 * 7)))
    sharpe_pctile = float(np.mean(np.array(random_sharpes) < sharpe_costs) * 100) if random_sharpes else 50.0

    if n_valid > 10:
        vi = np.where(valid_mask)[0]
        transitions = np.where(np.diff(np.sign(actual_dir[vi])) != 0)[0] + 1
        trans = float(np.mean(pred_dir[vi][transitions] == actual_dir[vi][transitions])) if len(transitions) > 0 else 0.5
    else:
        trans = 0.5

    return EvalResult(
        econ_dir_accuracy=econ_dir, sharpe_with_costs=sharpe_costs,
        sharpe_percentile=sharpe_pctile, transition_accuracy=trans,
        quantile_accuracy=q_acc, n_valid=n_valid,
        train_loss=train_loss, train_time=train_time,
    )


# --- Main Entry ---


def run_feature_test(
    name: str,
    feature_fn: FeatureFn,
    n_new_features: int,
    n_seeds: int = 3,
) -> dict:
    """Run a feature test: baseline D1 vs D1 + new features, across seeds.

    Args:
        name: Feature test name (used for output file).
        feature_fn: Function(ohlcv_df, detail_coeffs, approx_coeffs, level)
            returning (n_deltas, n_new_features) array.
        n_new_features: Number of features returned by feature_fn.
        n_seeds: Number of random seeds for comparison.

    Returns:
        Dict with baseline and challenger results per seed.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Feature test: %s (%d new features, %d seeds)", name, n_new_features, n_seeds)

    # Load data
    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))
    logger.info("Loaded %d tickers", len(tickers))

    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)

    # Build datasets (deterministic — only need to do once)
    logger.info("Building baseline dataset (4 aux features)...")
    X_train_base, tr_rets, tr_valid, tr_lvls, _, _ = _build_d1_pipeline(
        train_prices, None, tickers, None, 0,
    )
    X_test_base, te_rets, te_valid, te_lvls, _, _ = _build_d1_pipeline(
        test_prices, None, tickers, None, 0,
    )

    logger.info("Building challenger dataset (4 + %d aux features)...", n_new_features)
    X_train_chal, _, _, _, _, _ = _build_d1_pipeline(
        train_prices, train_ohlcv, tickers, feature_fn, n_new_features,
    )
    X_test_chal, _, _, _, _, _ = _build_d1_pipeline(
        test_prices, test_ohlcv, tickers, feature_fn, n_new_features,
    )

    logger.info("Train: %d windows, Test: %d windows", len(X_train_base), len(X_test_base))

    # Quantile boundaries from training
    boundaries = compute_quantile_boundaries(tr_rets, tr_lvls, tr_valid, PERCENTILES, per_level=True)
    y_train = assign_quantile_labels(tr_rets, tr_lvls, boundaries)
    y_test = assign_quantile_labels(te_rets, te_lvls, boundaries)

    # Class weights
    valid_labels = y_train[tr_valid]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    y_tr = y_train.astype(np.float64)
    y_te_labels = y_test

    baseline_results: list[EvalResult] = []
    challenger_results: list[EvalResult] = []

    for seed in range(n_seeds):
        logger.info("--- Seed %d/%d ---", seed + 1, n_seeds)

        # Set seeds
        np.random.seed(seed * 42 + 7)
        torch.manual_seed(seed * 42 + 7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 42 + 7)

        # Baseline
        logger.info("  Training baseline...")
        model_base = WaveletGPT(
            vocab_size=1, context_length=CONTEXT_LENGTH,
            task="return_quantile", n_output_classes=N_CLASSES,
            class_weights=class_weights, input_mode="continuous",
            n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
        )
        t0 = time.time()
        metrics = model_base.fit(X_train_base, y_tr)
        base_time = time.time() - t0
        pred_base = model_base.predict(X_test_base).astype(np.int64)
        base_eval = _evaluate(pred_base, te_rets, y_te_labels, te_valid, metrics["train_loss"], base_time)
        baseline_results.append(base_eval)
        logger.info("  Baseline: econ_dir=%.1f%% sharpe=%.3f",
                     base_eval.econ_dir_accuracy * 100, base_eval.sharpe_with_costs)

        # Challenger
        logger.info("  Training challenger...")
        model_chal = WaveletGPT(
            vocab_size=1, context_length=CONTEXT_LENGTH,
            task="return_quantile", n_output_classes=N_CLASSES,
            class_weights=class_weights, input_mode="continuous",
            n_aux_features=N_AUX_FEATURES + n_new_features, **MODEL_KWARGS,
        )
        t0 = time.time()
        metrics = model_chal.fit(X_train_chal, y_tr)
        chal_time = time.time() - t0
        pred_chal = model_chal.predict(X_test_chal).astype(np.int64)
        chal_eval = _evaluate(pred_chal, te_rets, y_te_labels, te_valid, metrics["train_loss"], chal_time)
        challenger_results.append(chal_eval)
        logger.info("  Challenger: econ_dir=%.1f%% sharpe=%.3f",
                     chal_eval.econ_dir_accuracy * 100, chal_eval.sharpe_with_costs)

    # --- Results ---
    base_econ = np.mean([r.econ_dir_accuracy for r in baseline_results])
    chal_econ = np.mean([r.econ_dir_accuracy for r in challenger_results])
    base_sharpe = np.mean([r.sharpe_with_costs for r in baseline_results])
    chal_sharpe = np.mean([r.sharpe_with_costs for r in challenger_results])
    base_trans = np.mean([r.transition_accuracy for r in baseline_results])
    chal_trans = np.mean([r.transition_accuracy for r in challenger_results])

    delta_econ = chal_econ - base_econ
    delta_sharpe = chal_sharpe - base_sharpe
    delta_trans = chal_trans - base_trans

    # Verdict
    improved_econ = delta_econ > 0.01
    improved_sharpe = delta_sharpe > 0.2
    improved_trans = delta_trans > 0.02
    degraded_econ = delta_econ < -0.01
    degraded_sharpe = delta_sharpe < -0.3
    degraded_trans = delta_trans < -0.01

    any_improved = improved_econ or improved_sharpe or improved_trans
    any_degraded = degraded_econ or degraded_sharpe or degraded_trans

    if any_improved and not any_degraded:
        verdict = "PASS"
    elif any_degraded:
        verdict = "FAIL"
    else:
        verdict = "NO EFFECT"

    # Consistency check: did challenger beat baseline in all seeds?
    all_seeds_better_econ = all(
        c.econ_dir_accuracy > b.econ_dir_accuracy
        for b, c in zip(baseline_results, challenger_results, strict=True)
    )

    w = 100
    print()
    print("=" * w)
    print(f"FEATURE TEST: {name}")
    print(f"New features: {n_new_features}, Seeds: {n_seeds}")
    print("=" * w)
    print(f"\n{'Metric':<20} {'Baseline':>10} {'Challenger':>12} {'Delta':>10} {'Threshold':>12}")
    print("-" * w)
    print(f"{'Econ Dir Acc':<20} {base_econ:>9.1%} {chal_econ:>11.1%} {delta_econ:>+9.1%} "
          f"{'> +1.0pp':>12}  {'IMPROVED' if improved_econ else 'DEGRADED' if degraded_econ else ''}")
    print(f"{'Sharpe (+costs)':<20} {base_sharpe:>+9.3f} {chal_sharpe:>+11.3f} {delta_sharpe:>+9.3f} "
          f"{'> +0.2':>12}  {'IMPROVED' if improved_sharpe else 'DEGRADED' if degraded_sharpe else ''}")
    print(f"{'Transition Acc':<20} {base_trans:>9.1%} {chal_trans:>11.1%} {delta_trans:>+9.1%} "
          f"{'> +2.0pp':>12}  {'IMPROVED' if improved_trans else 'DEGRADED' if degraded_trans else ''}")
    print("-" * w)

    print(f"\nPer-seed Econ Dir: baseline={[f'{r.econ_dir_accuracy:.1%}' for r in baseline_results]}"
          f"  challenger={[f'{r.econ_dir_accuracy:.1%}' for r in challenger_results]}")
    print(f"All seeds better econ dir: {all_seeds_better_econ}")

    print(f"\nVERDICT: {verdict}")
    if verdict == "PASS":
        print("  -> Feature improves D1. Validate on 2026 before production.")
    elif verdict == "FAIL":
        print("  -> Feature degrades at least one metric. Discard.")
    else:
        print("  -> No meaningful effect. Not worth the complexity.")

    print("=" * w)

    # Save
    results_dir = Path.home() / ".wavecast" / "audit" / "feature_tests"
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "name": name,
        "n_new_features": n_new_features,
        "n_seeds": n_seeds,
        "verdict": verdict,
        "baseline_mean": {"econ_dir": base_econ, "sharpe_costs": base_sharpe, "transition": base_trans},
        "challenger_mean": {"econ_dir": chal_econ, "sharpe_costs": chal_sharpe, "transition": chal_trans},
        "delta": {"econ_dir": delta_econ, "sharpe_costs": delta_sharpe, "transition": delta_trans},
        "all_seeds_better_econ": all_seeds_better_econ,
        "per_seed": {
            "baseline": [
                {"econ_dir": r.econ_dir_accuracy, "sharpe_costs": r.sharpe_with_costs,
                 "transition": r.transition_accuracy, "train_loss": r.train_loss}
                for r in baseline_results
            ],
            "challenger": [
                {"econ_dir": r.econ_dir_accuracy, "sharpe_costs": r.sharpe_with_costs,
                 "transition": r.transition_accuracy, "train_loss": r.train_loss}
                for r in challenger_results
            ],
        },
    }
    out_path = results_dir / f"{name}_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)

    return output
