#!/usr/bin/env python3
"""Validate level-1-only D1 model on 2026 OOS data and compare with all-levels model.

Loads results from d1_2026_validation.json for the comparison table.

Usage:
    python scripts/validate_d1_level1_2026.py [OPTIONS]

    --model-path PATH   Model directory (default: ~/.wavecast/models/d1_level1_v1)
    --start DATE        Start date (default: 2026-01-01)
    --end DATE          End date (default: 2026-02-16)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wavecast.core.types import TimeSeries
from wavecast.core.universe import DEFAULT_UNIVERSE
from wavecast.data.auxiliary_features import (
    N_AUX_FEATURES,
    compute_detail_auxiliary_features,
)
from wavecast.data.continuous_dataset import build_continuous_dataset
from wavecast.data.sources import fetch_massive
from wavecast.experiments.runner import SECTOR_ID_MAP
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.targets.returns import assign_quantile_labels
from wavecast.wavelets.dwt import decompose

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TICKERS = [a.ticker for a in DEFAULT_UNIVERSE.assets]
TICKER_SECTOR = {a.ticker: a.sector.value for a in DEFAULT_UNIVERSE.assets if a.sector}
SECTOR_ORDER = ["tech", "finance", "energy", "healthcare", "broad_etf", "commodity_etf"]
INTERVAL = "1h"
DETAIL_LEVELS = [1]  # Level-1 only
CONTEXT_LENGTH = 16
N_CLASSES = 5
MIN_RETURN_THRESHOLD = 0.001
COST_BPS = 7.0
N_RANDOM_TRIALS = 500

DEFAULT_MODEL = Path.home() / ".wavecast" / "models" / "d1_level1_v1"
ALL_LEVELS_RESULTS = Path.home() / ".wavecast" / "audit" / "d1_2026_validation.json"


@dataclass
class GroupResult:
    name: str
    n_test: int
    n_valid: int
    econ_dir_accuracy: float
    sharpe_raw: float
    sharpe_with_costs: float
    sharpe_percentile: float
    transition_accuracy: float
    quantile_accuracy: float


def evaluate_group(
    name: str, pred_labels: NDArray, actual_returns: NDArray,
    actual_labels: NDArray, valid_mask: NDArray,
) -> GroupResult:
    n_test = len(pred_labels)
    n_valid = int(valid_mask.sum())
    mid = N_CLASSES // 2

    pred_dir = np.zeros(n_test, dtype=np.float64)
    pred_dir[pred_labels > mid] = 1.0
    pred_dir[pred_labels < mid] = -1.0
    actual_dir = np.sign(actual_returns)

    filt = valid_mask & (np.abs(actual_returns) > MIN_RETURN_THRESHOLD)
    if filt.sum() > 0:
        has_pred = pred_dir[filt] != 0
        econ_dir_acc = float(np.mean(pred_dir[filt][has_pred] == actual_dir[filt][has_pred])) if has_pred.sum() > 0 else 0.5
    else:
        econ_dir_acc = 0.5

    q_acc = float(np.mean(pred_labels[valid_mask] == actual_labels[valid_mask])) if n_valid > 0 else 0.0

    if n_valid > 0:
        pnl_raw = pred_dir[valid_mask] * actual_returns[valid_mask]
        pnl_raw = pnl_raw[~np.isnan(pnl_raw)]
        dir_changes = np.abs(np.diff(pred_dir[valid_mask]))
        cost_per_step = np.zeros(n_valid)
        cost_per_step[1:] = dir_changes * (COST_BPS / 10000)
        cost_per_step[0] = abs(pred_dir[valid_mask][0]) * (COST_BPS / 10000)
        pnl_net = pred_dir[valid_mask] * actual_returns[valid_mask] - cost_per_step
        pnl_net = pnl_net[~np.isnan(pnl_net)]
        sharpe_raw = float(np.mean(pnl_raw) / np.std(pnl_raw) * np.sqrt(252 * 7)) if len(pnl_raw) > 1 and np.std(pnl_raw) > 0 else 0.0
        sharpe_costs = float(np.mean(pnl_net) / np.std(pnl_net) * np.sqrt(252 * 7)) if len(pnl_net) > 1 and np.std(pnl_net) > 0 else 0.0
    else:
        sharpe_raw = 0.0
        sharpe_costs = 0.0

    rng = np.random.default_rng(42)
    random_sharpes = []
    if n_valid > 0:
        valid_rets_clean = actual_returns[valid_mask][~np.isnan(actual_returns[valid_mask])]
        for _ in range(N_RANDOM_TRIALS):
            rand_dir = rng.choice([-1.0, 0.0, 1.0], size=len(valid_rets_clean))
            rand_pnl = rand_dir * valid_rets_clean
            if len(rand_pnl) > 1 and np.std(rand_pnl) > 0:
                random_sharpes.append(float(np.mean(rand_pnl) / np.std(rand_pnl) * np.sqrt(252 * 7)))
    sharpe_pctile = float(np.mean(np.array(random_sharpes) < sharpe_costs) * 100) if random_sharpes else 50.0

    if n_valid > 10:
        valid_idx = np.where(valid_mask)[0]
        transitions = np.where(np.diff(np.sign(actual_dir[valid_idx])) != 0)[0] + 1
        trans_acc = float(np.mean(pred_dir[valid_idx][transitions] == actual_dir[valid_idx][transitions])) if len(transitions) > 0 else 0.5
    else:
        trans_acc = 0.5

    return GroupResult(
        name=name, n_test=n_test, n_valid=n_valid,
        econ_dir_accuracy=econ_dir_acc, sharpe_raw=sharpe_raw,
        sharpe_with_costs=sharpe_costs, sharpe_percentile=sharpe_pctile,
        transition_accuracy=trans_acc, quantile_accuracy=q_acc,
    )


def main() -> None:
    model_path = DEFAULT_MODEL
    start_date = "2026-01-01"
    end_date = "2026-02-16"

    for arg in sys.argv[1:]:
        if arg.startswith("--model-path="):
            model_path = Path(arg.split("=", 1)[1])
        elif arg.startswith("--start="):
            start_date = arg.split("=", 1)[1]
        elif arg.startswith("--end="):
            end_date = arg.split("=", 1)[1]

    model_dir = Path(model_path)
    if not (model_dir / "config.json").exists():
        logger.error("Model not found at %s — run scripts/train_d1_level1.py first", model_path)
        sys.exit(1)

    # Load model + boundaries
    model = WaveletGPT.load(model_dir)
    with open(model_dir / "quantile_boundaries.json") as f:
        boundaries: dict[int, NDArray] = {
            int(k): np.array(v, dtype=np.float64) for k, v in json.load(f).items()
        }

    # Load all-levels comparison data
    ref_data = None
    if ALL_LEVELS_RESULTS.exists():
        with open(ALL_LEVELS_RESULTS) as f:
            ref_data = json.load(f)

    # Fetch 2026 data
    test_prices: dict[str, TimeSeries] = {}
    for i, ticker in enumerate(TICKERS):
        logger.info("Fetching %s (%d/%d) ...", ticker, i + 1, len(TICKERS))
        try:
            ts = fetch_massive(ticker, start=start_date, end=end_date, interval=INTERVAL)
            if len(ts.values) >= 50:
                test_prices[ticker] = ts
        except Exception as e:
            logger.error("  %s: %s", ticker, e)
        if i < len(TICKERS) - 1:
            time.sleep(12.5)

    tickers = sorted(test_prices.keys())
    logger.info("Loaded %d tickers", len(tickers))

    ac_map = {
        a.ticker: SECTOR_ID_MAP.get(a.sector.value, 0)
        for a in DEFAULT_UNIVERSE.assets if a.sector
    }

    # Build level-1-only D1 dataset
    test_series: dict[tuple[str, int], NDArray] = {}
    test_aux_series: dict[tuple[str, int], NDArray] = {}

    for ticker in tickers:
        decomp = decompose(test_prices[ticker], level=5)
        for lvl in DETAIL_LEVELS:
            detail = decomp.detail_at_level(lvl)
            deltas = np.diff(detail) if len(detail) > 1 else detail
            if len(deltas) > CONTEXT_LENGTH:
                test_series[(ticker, lvl)] = deltas
                test_aux_series[(ticker, lvl)] = compute_detail_auxiliary_features(
                    detail, decomp.approximation
                )

    test_ds = build_continuous_dataset(test_series, CONTEXT_LENGTH, ac_map, normalize=True)
    logger.info("Test dataset: %d windows (level-1 only)", len(test_ds.windows))

    # Compute returns
    test_rets = np.full(len(test_ds.windows), np.nan)
    test_valid = np.zeros(len(test_ds.windows), dtype=np.bool_)
    for i, w in enumerate(test_ds.windows):
        key = (w.ticker, w.level)
        if key not in test_series or w.ticker not in test_prices:
            continue
        target_pos = w.token_position
        if target_pos >= len(test_series[key]):
            continue
        span = 2 ** w.level
        bar_start = target_pos * span
        bar_end = bar_start + span
        pv = test_prices[w.ticker].values
        if bar_end < len(pv) and pv[bar_start] > 0:
            test_rets[i] = (pv[bar_end] - pv[bar_start]) / pv[bar_start]
            test_valid[i] = True

    test_lvls = np.array([w.level for w in test_ds.windows], dtype=np.int64)
    y_test = assign_quantile_labels(test_rets, test_lvls, boundaries)

    # Build X
    ctx_arr, lvl_arr, ac_arr = test_ds.to_arrays()
    aux_windows = []
    for (_t, _l), aux in sorted(test_aux_series.items()):
        if len(aux) <= CONTEXT_LENGTH:
            continue
        for j in range(len(aux) - CONTEXT_LENGTH):
            aux_windows.append(aux[j:j + CONTEXT_LENGTH])
    aux_win = np.array(aux_windows, dtype=np.float64) if aux_windows else np.empty(
        (0, CONTEXT_LENGTH, N_AUX_FEATURES), dtype=np.float64
    )
    aux_flat = aux_win.reshape(len(aux_win), -1)
    X_test = np.column_stack([ctx_arr, aux_flat, lvl_arr, ac_arr])

    # Predict
    predicted = model.predict(X_test).astype(np.int64)

    # Evaluate
    test_tickers = np.array([w.ticker for w in test_ds.windows])
    test_sectors = np.array([TICKER_SECTOR.get(w.ticker, "unknown") for w in test_ds.windows])

    agg = evaluate_group("L1-MODEL", predicted, test_rets, y_test, test_valid)

    ticker_results = []
    for ticker in sorted(set(test_tickers)):
        idx = np.where(test_tickers == ticker)[0]
        ticker_results.append(evaluate_group(
            ticker, predicted[idx], test_rets[idx], y_test[idx], test_valid[idx],
        ))

    sector_results = []
    for sector in SECTOR_ORDER:
        mask = test_sectors == sector
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        sector_results.append(evaluate_group(
            sector, predicted[idx], test_rets[idx], y_test[idx], test_valid[idx],
        ))

    # --- Comparison table ---
    w = 120
    print()
    print("=" * w)
    print("D1 LEVEL-1-ONLY MODEL vs ALL-LEVELS MODEL (2026 OOS)")
    print(f"L1 Model: {model_path}")
    print(f"Test: {start_date} to {end_date}, {len(tickers)} tickers")
    print("=" * w)

    # Get reference metrics
    ref_all = ref_data["aggregate"] if ref_data else {}
    ref_l1 = ref_data["level_1_only"] if ref_data else {}

    print(f"\n{'':20s} {'All-Levels':>12} {'All-Lvl→L1':>12} {'L1 Model':>12}")
    print(f"{'':20s} {'(all test)':>12} {'(filtered)':>12} {'(this run)':>12}")
    print("-" * 60)
    print(f"{'Samples':<20} {ref_all.get('n_test', '?'):>12} {ref_l1.get('n_test', '?'):>12} {agg.n_test:>12}")
    print(f"{'Valid':<20} {ref_all.get('n_valid', '?'):>12} {ref_l1.get('n_valid', '?'):>12} {agg.n_valid:>12}")

    def fmt_pct(v):
        return f"{v:.1%}" if isinstance(v, float) else str(v)

    def fmt_sharpe(v):
        return f"{v:+.3f}" if isinstance(v, float) else str(v)

    print(f"{'Econ Dir Acc':<20} {fmt_pct(ref_all.get('econ_dir_accuracy', '?')):>12} "
          f"{fmt_pct(ref_l1.get('econ_dir_accuracy', '?')):>12} {agg.econ_dir_accuracy:>11.1%}")
    print(f"{'Sharpe (+costs)':<20} {fmt_sharpe(ref_all.get('sharpe_with_costs', '?')):>12} "
          f"{fmt_sharpe(ref_l1.get('sharpe_with_costs', '?')):>12} {agg.sharpe_with_costs:>+11.3f}")
    print(f"{'vs Random':<20} {fmt_pct(ref_all.get('sharpe_percentile', '?') / 100 if isinstance(ref_all.get('sharpe_percentile'), (int, float)) else '?'):>12} "
          f"{fmt_pct(ref_l1.get('sharpe_percentile', '?') / 100 if isinstance(ref_l1.get('sharpe_percentile'), (int, float)) else '?'):>12} {agg.sharpe_percentile:>11.1f}%")
    print(f"{'Transition Acc':<20} {fmt_pct(ref_all.get('transition_accuracy', '?')):>12} "
          f"{fmt_pct(ref_l1.get('transition_accuracy', '?')):>12} {agg.transition_accuracy:>11.1%}")

    # Verdict
    ref_l1_econ = ref_l1.get("econ_dir_accuracy", 0.5) if ref_l1 else 0.5
    delta = agg.econ_dir_accuracy - ref_l1_econ
    if delta > 0.01:
        comparison = "IMPROVED"
    elif delta > -0.01:
        comparison = "SAME"
    else:
        comparison = "DEGRADED"

    print(f"\nVERDICT: {comparison} (L1 model {delta:+.1%} vs all-levels filtered to L1)")

    # Per-ticker
    print("\n" + "=" * w)
    print("PER-TICKER (L1-ONLY MODEL)")
    print("=" * w)
    print(f"{'Ticker':<7} {'Sector':<14} {'Valid':>7} {'Econ Dir':>8} {'S+Cost':>7} {'vs Rnd':>7} {'Trans':>6}")
    print("-" * w)
    for r in sorted(ticker_results, key=lambda x: x.econ_dir_accuracy, reverse=True):
        print(f"{r.name:<7} {TICKER_SECTOR.get(r.name, '?'):<14} {r.n_valid:>7d} "
              f"{r.econ_dir_accuracy:>7.1%} {r.sharpe_with_costs:>+7.3f} "
              f"{r.sharpe_percentile:>6.1f}% {r.transition_accuracy:>5.1%}")

    # Per-sector
    print("\n" + "=" * w)
    print("PER-SECTOR (L1-ONLY MODEL)")
    print("=" * w)
    for r in sector_results:
        print(f"{r.name:<14} {r.n_valid:>7d} {r.econ_dir_accuracy:>7.1%} "
              f"{r.sharpe_with_costs:>+7.3f} {r.sharpe_percentile:>6.1f}%")

    print("=" * w)

    # Save
    results_dir = Path.home() / ".wavecast" / "audit"
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "test_period": {"start": start_date, "end": end_date},
        "model_path": str(model_path),
        "level_1_only": True,
        "aggregate": asdict(agg),
        "per_ticker": [asdict(r) for r in ticker_results],
        "per_sector": [asdict(r) for r in sector_results],
        "comparison": {
            "all_levels_aggregate": ref_all,
            "all_levels_l1_filtered": ref_l1,
            "l1_model": asdict(agg),
            "verdict": comparison,
            "delta_econ_dir": round(delta, 4),
        },
    }
    out_path = results_dir / "d1_level1_2026_validation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
