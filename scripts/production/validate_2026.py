"""2026 out-of-sample validation of the winning production trading rule.

Loads the frozen CE model (trained 2021-2025), runs inference on 2026 data,
applies the winning config from pnl_simulation.py, and compares 2025 vs 2026.

Usage:
    cd ~/wavecast && source .venv/bin/activate
    python -u -m scripts.production.validate_2026

    # With explicit config override:
    python -u -m scripts.production.validate_2026 --config B2i --lookback 3

    # Skip API fetch (use cached data only):
    python -u -m scripts.production.validate_2026 --skip-fetch
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wavecast.core.types import TimeSeries
from wavecast.core.universe import DEFAULT_UNIVERSE
from wavecast.data.auxiliary_features import (
    N_AUX_FEATURES,
    compute_detail_auxiliary_features,
)
from wavecast.data.cache import ParquetCache
from wavecast.data.continuous_dataset import build_continuous_dataset
from wavecast.data.sources import fetch_massive
from wavecast.experiments.runner import SECTOR_ID_MAP
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.targets.returns import assign_quantile_labels
from wavecast.wavelets.dwt import decompose

from scripts.production.pnl_simulation import (
    COST_BPS,
    MIN_RETURN_THRESHOLD,
    compute_magnitude_signals,
    is_reversal,
    simulate_pnl,
    _build_recent_returns,
    _sharpe,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TICKERS = [a.ticker for a in DEFAULT_UNIVERSE.assets]
INTERVAL = "1h"
DETAIL_LEVELS = [1]  # Match the harness: level 1 only for the experiment pipeline
CONTEXT_LENGTH = 16
N_CLASSES = 5
MID = N_CLASSES // 2

MODEL_DIR = Path.home() / ".wavecast" / "models" / "d1_augmented_v1"
RESULTS_DIR = Path.home() / ".wavecast" / "audit" / "production"
SIM_RESULTS = RESULTS_DIR / "pnl_simulation.json"


def _load_2026_cached(start: str = "2026-01-01") -> dict[str, TimeSeries]:
    """Load 2026 data from parquet cache."""
    cache_dir = Path.home() / ".wavecast" / "cache"
    prices: dict[str, TimeSeries] = {}
    start_dt = np.datetime64(start)
    for ticker in TICKERS:
        path = cache_dir / f"{ticker}_1h_ohlcv.parquet"
        if not path.exists():
            continue
        import pandas as pd
        df = pd.read_parquet(path)
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        mask = df["timestamp"] >= pd.Timestamp(start)
        df_2026 = df[mask]
        if len(df_2026) < 50:
            continue
        prices[ticker] = TimeSeries(
            values=df_2026["close"].to_numpy(dtype=np.float64),
            timestamps=df_2026["timestamp"].to_numpy(dtype="datetime64[ns]"),
            ticker=ticker, interval=INTERVAL,
        )
    return prices


def _fetch_2026(start: str, end: str, rate_pause: float = 12.5) -> dict[str, TimeSeries]:
    """Fetch 2026 data from Massive API and save to cache."""
    prices: dict[str, TimeSeries] = {}
    for i, ticker in enumerate(TICKERS):
        logger.info("Fetching %s (%d/%d)...", ticker, i + 1, len(TICKERS))
        try:
            ts = fetch_massive(ticker, start=start, end=end, interval=INTERVAL)
            if len(ts.values) < 50:
                logger.warning("  %s: only %d bars, skipping", ticker, len(ts.values))
                continue
            prices[ticker] = ts
            logger.info("  %s: %d bars", ticker, len(ts.values))
        except Exception as e:
            logger.error("  %s: fetch failed: %s", ticker, e)
        if i < len(TICKERS) - 1:
            time.sleep(rate_pause)
    return prices


def _build_d1_test(
    prices: dict[str, TimeSeries],
) -> tuple[NDArray, NDArray, NDArray, NDArray, list]:
    """Build D1 dataset from prices. Returns (X, returns, valid, levels, windows)."""
    ac_map = {
        a.ticker: SECTOR_ID_MAP.get(a.sector.value, 0)
        for a in DEFAULT_UNIVERSE.assets if a.sector
    }

    coeff_series: dict[tuple[str, int], NDArray] = {}
    aux_series: dict[tuple[str, int], NDArray] = {}

    for ticker in sorted(prices.keys()):
        ts = prices[ticker]
        dec = decompose(ts, level=5)
        for lvl in DETAIL_LEVELS:
            detail = dec.detail_at_level(lvl)
            deltas = np.diff(detail) if len(detail) > 1 else detail
            if len(deltas) > CONTEXT_LENGTH:
                coeff_series[(ticker, lvl)] = deltas
                aux_series[(ticker, lvl)] = compute_detail_auxiliary_features(
                    detail, dec.approximation,
                )

    ds = build_continuous_dataset(coeff_series, CONTEXT_LENGTH, ac_map, normalize=True)
    windows = ds.windows

    if not windows:
        return np.empty((0,)), np.empty((0,)), np.zeros(0, dtype=bool), np.empty((0,), dtype=np.int64), []

    # Returns
    rets = np.full(len(windows), np.nan)
    valid = np.zeros(len(windows), dtype=bool)
    for i, w in enumerate(windows):
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

    lvls = np.array([w.level for w in windows], dtype=np.int64)

    # Build X
    ctx_arr, lvl_arr, ac_arr = ds.to_arrays()

    # Aux features
    aux_windows = []
    for (_t, _l), aux in sorted(aux_series.items()):
        if len(aux) <= CONTEXT_LENGTH:
            continue
        for j in range(len(aux) - CONTEXT_LENGTH):
            aux_windows.append(aux[j:j + CONTEXT_LENGTH])
    if aux_windows:
        aux_win = np.array(aux_windows, dtype=np.float64)
    else:
        aux_win = np.empty((0, CONTEXT_LENGTH, N_AUX_FEATURES), dtype=np.float64)
    aux_flat = aux_win.reshape(len(aux_win), -1)
    X = np.column_stack([ctx_arr, aux_flat, lvl_arr, ac_arr])

    return X, rets, valid, lvls, windows


def main():
    skip_fetch = "--skip-fetch" in sys.argv
    start_date = "2026-01-01"
    end_date = "2026-02-22"

    # Load winning config from PnL sim
    config_override = None
    lookback_override = None
    for arg in sys.argv[1:]:
        if arg.startswith("--config="):
            config_override = arg.split("=", 1)[1]
        elif arg.startswith("--lookback="):
            lookback_override = int(arg.split("=", 1)[1])
        elif arg.startswith("--start="):
            start_date = arg.split("=", 1)[1]
        elif arg.startswith("--end="):
            end_date = arg.split("=", 1)[1]

    # Read pnl_simulation results for winning config
    if SIM_RESULTS.exists():
        with open(SIM_RESULTS) as f:
            sim = json.load(f)
        best_config = config_override or sim["best_config"]
        best_lookback = lookback_override or sim["best_lookback"]
        best_signal_name = sim["best_magnitude_signal"]
        ref_2025 = sim["configs"].get(best_config, sim["best_result"])
        bin_midpoints = np.array(sim["meta"]["bin_midpoints"])
    else:
        logger.warning("No pnl_simulation.json found — using defaults (B2i, lookback=3)")
        best_config = config_override or "B2i"
        best_lookback = lookback_override or 3
        best_signal_name = "B_expected_abs"
        ref_2025 = None
        bin_midpoints = None

    # Parse config label
    trade_filter = "reversal_only" if best_config.startswith("B") else "all"
    magnitude_filter = "large_only" if "2" in best_config[1:3] else "all"
    sizing_method = "magnitude_weighted" if best_config.endswith("ii") else "flat"

    logger.info("=" * 90)
    logger.info("2026 OUT-OF-SAMPLE VALIDATION")
    logger.info("Config: %s (filter=%s, mag=%s, sizing=%s, lookback=%d)",
                best_config, trade_filter, magnitude_filter, sizing_method, best_lookback)
    logger.info("=" * 90)

    # --- Load production model ---
    logger.info("Loading model from %s", MODEL_DIR)
    model = WaveletGPT.load(MODEL_DIR)

    boundaries_path = MODEL_DIR / "quantile_boundaries.json"
    with open(boundaries_path) as f:
        boundaries_raw = json.load(f)
    boundaries = {int(k): np.array(v, dtype=np.float64) for k, v in boundaries_raw.items()}

    # Compute bin midpoints from boundaries if not loaded from sim
    if bin_midpoints is None:
        # Approximate from boundaries (level 1)
        b = boundaries.get(1, np.zeros(4))
        # Simple midpoints: mean of neighboring boundaries, extreme bins extrapolated
        bin_midpoints = np.array([
            b[0] - (b[1] - b[0]),  # extrapolate below
            (b[0] + b[1]) / 2,
            (b[1] + b[2]) / 2,
            (b[2] + b[3]) / 2,
            b[3] + (b[3] - b[2]),  # extrapolate above
        ])

    # --- Load 2026 data ---
    if skip_fetch:
        logger.info("Loading 2026 data from cache...")
        prices_2026 = _load_2026_cached(start_date)
    else:
        logger.info("Fetching 2026 data from API...")
        prices_2026 = _fetch_2026(start_date, end_date)

    if not prices_2026:
        logger.error("No 2026 data available. Try without --skip-fetch.")
        sys.exit(1)

    tickers = sorted(prices_2026.keys())
    logger.info("Loaded %d tickers for 2026", len(tickers))
    for t in tickers:
        ts = prices_2026[t]
        logger.info("  %s: %d bars", t, len(ts.values))

    # --- Build D1 dataset ---
    X_test, te_rets, te_valid, te_lvls, windows = _build_d1_test(prices_2026)
    n_valid = int(te_valid.sum())
    logger.info("2026 dataset: %d windows, %d valid", len(X_test), n_valid)

    if n_valid < 50:
        logger.error("Too few valid samples (%d) for meaningful evaluation", n_valid)
        sys.exit(1)

    # --- Predict ---
    logger.info("Running inference...")
    pred_labels = model.predict(X_test).astype(np.int64)
    pred_proba = model.predict_proba(X_test)

    pred_dir = np.zeros(len(pred_labels), dtype=np.float64)
    pred_dir[pred_labels > MID] = 1.0
    pred_dir[pred_labels < MID] = -1.0

    # --- Assign ground truth labels ---
    y_test = assign_quantile_labels(te_rets, te_lvls, boundaries)

    # --- Build recent returns ---
    recent_returns = _build_recent_returns(windows, te_rets, max_lookback=5)

    # --- Magnitude signal ---
    # For 2026 we don't have a regression model — use CE signals only
    signals = compute_magnitude_signals(
        pred_proba, bin_midpoints, np.zeros(len(pred_labels)),  # no regression
    )
    # If best was Signal D (regression), fall back to Signal B
    if best_signal_name == "D_reg_abs":
        logger.warning("Regression signal not available for 2026 — falling back to B_expected_abs")
        best_signal_name = "B_expected_abs"
    mag_signal = signals[best_signal_name]

    # --- Run winning config ---
    result_2026 = simulate_pnl(
        pred_dir, te_rets, te_valid, mag_signal, recent_returns,
        trade_filter=trade_filter, magnitude_filter=magnitude_filter,
        sizing_method=sizing_method, lookback=best_lookback,
    )

    # --- Also run ALL (no filter) for comparison ---
    result_2026_all = simulate_pnl(
        pred_dir, te_rets, te_valid, mag_signal, recent_returns,
        trade_filter="all", magnitude_filter="all", sizing_method="flat",
    )

    # --- Compute reversal vs continuation breakdown ---
    vi = np.where(te_valid)[0]
    reversal_correct = 0
    reversal_total = 0
    continuation_correct = 0
    continuation_total = 0
    actual_dir = np.sign(te_rets)

    for idx in vi:
        if pred_dir[idx] == 0 or actual_dir[idx] == 0:
            continue
        if np.abs(te_rets[idx]) <= MIN_RETURN_THRESHOLD:
            continue
        rev = is_reversal(pred_dir[idx], recent_returns[idx], best_lookback)
        if rev is None:
            n_rev_none += 1
            continue
        correct = bool(pred_dir[idx] == actual_dir[idx])
        if bool(rev):
            reversal_total += 1
            if correct:
                reversal_correct += 1
        else:
            continuation_total += 1
            if correct:
                continuation_correct += 1

    rev_acc = reversal_correct / reversal_total if reversal_total > 0 else 0.0
    cont_acc = continuation_correct / continuation_total if continuation_total > 0 else 0.0


    # --- Print results ---
    print("\n" + "=" * 70)
    print("2026 OUT-OF-SAMPLE VALIDATION")
    print(f"Config: {best_config}  (lookback={best_lookback})")
    print(f"Model: {MODEL_DIR}")
    print(f"Test: {start_date} to {end_date}")
    print(f"Tickers: {len(tickers)}")
    print("=" * 70)

    print(f"\n{'Metric':<28} {'2025 Test':>12} {'2026 OOS':>12} {'Delta':>10}")
    print("-" * 62)

    def _row(label, v25, v26, fmt=".1%"):
        if v25 is not None:
            d = v26 - v25
            s25 = f"{v25:{fmt}}"
            s26 = f"{v26:{fmt}}"
            sd = f"{d:+{fmt}}" if isinstance(d, float) else f"{d:+,d}"
            print(f"  {label:<26} {s25:>12} {s26:>12} {sd:>10}")
        else:
            s26 = f"{v26:{fmt}}"
            print(f"  {label:<26} {'---':>12} {s26:>12}")

    def _row_f(label, v25, v26, fmt="+.3f"):
        if v25 is not None:
            s25 = f"{v25:{fmt}}"
            s26 = f"{v26:{fmt}}"
            print(f"  {label:<26} {s25:>12} {s26:>12}")
        else:
            s26 = f"{v26:{fmt}}"
            print(f"  {label:<26} {'---':>12} {s26:>12}")

    r25 = ref_2025
    r26 = result_2026

    def _row_int(label, v25, v26):
        if v25 is not None:
            print(f"  {label:<26} {v25:>12,d} {v26:>12,d} {v26-v25:>+10,d}")
        else:
            print(f"  {label:<26} {'---':>12} {v26:>12,d}")

    _row_int("Trades taken", r25["n_trades"] if r25 else None, r26["n_trades"])
    _row("Accuracy (on taken)", r25["accuracy"] if r25 else None, r26["accuracy"])
    _row_f("Sharpe (+costs)", r25["sharpe"] if r25 else None, r26["sharpe"])
    _row("Max drawdown", r25["max_drawdown"] if r25 else None, r26["max_drawdown"])
    _row("Expectancy/trade", r25["expectancy_per_trade"] if r25 else None, r26["expectancy_per_trade"], ".4%")
    _row("Win rate", r25["win_rate"] if r25 else None, r26["win_rate"])

    print(f"\n  {'Reversal accuracy':<26} {'---':>12} {rev_acc:>11.1%}  (n={reversal_total:,})")
    print(f"  {'Continuation accuracy':<26} {'---':>12} {cont_acc:>11.1%}  (n={continuation_total:,})")

    # --- Unfiltered baseline ---
    print(f"\n  {'ALL (no filter) accuracy':<26} {'---':>12} {result_2026_all['accuracy']:>11.1%}")
    all_sharpe_s = f"{result_2026_all['sharpe']:+.3f}"
    print(f"  {'ALL Sharpe':<26} {'---':>12} {all_sharpe_s:>12}")

    # --- Pass/fail ---
    print("\n" + "=" * 70)
    passes = []
    fails = []
    if r26["accuracy"] > 0.70:
        passes.append(f"accuracy {r26['accuracy']:.1%} > 70%")
    else:
        fails.append(f"accuracy {r26['accuracy']:.1%} <= 70%")
    if r26["sharpe"] > 3.0:
        passes.append(f"Sharpe {r26['sharpe']:+.3f} > +3.0")
    else:
        fails.append(f"Sharpe {r26['sharpe']:+.3f} <= +3.0")
    if r26["expectancy_per_trade"] > 0.003:
        passes.append(f"expectancy {r26['expectancy_per_trade']:.4%} > 0.30%")
    else:
        fails.append(f"expectancy {r26['expectancy_per_trade']:.4%} <= 0.30%")

    if not fails:
        verdict = "PASS"
    elif len(passes) >= 2:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    print(f"VERDICT: {verdict}")
    for p in passes:
        print(f"  [PASS] {p}")
    for f_item in fails:
        print(f"  [FAIL] {f_item}")
    print("=" * 70)

    # --- Save ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
        "config": best_config,
        "lookback": best_lookback,
        "magnitude_signal": best_signal_name,
        "test_period": {"start": start_date, "end": end_date},
        "model_path": str(MODEL_DIR),
        "n_tickers": len(tickers),
        "result_2026": _clean(r26),
        "result_2026_all": _clean(result_2026_all),
        "reference_2025": _clean(r25) if r25 else None,
        "reversal_breakdown": {
            "reversal_accuracy": rev_acc,
            "reversal_n": reversal_total,
            "continuation_accuracy": cont_acc,
            "continuation_n": continuation_total,
        },
        "verdict": verdict,
        "pass_criteria": {
            "accuracy_gt_70": r26["accuracy"] > 0.70,
            "sharpe_gt_3": r26["sharpe"] > 3.0,
            "expectancy_gt_0003": r26["expectancy_per_trade"] > 0.003,
        },
    }

    out_path = RESULTS_DIR / "2026_validation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)

    return output


if __name__ == "__main__":
    main()
