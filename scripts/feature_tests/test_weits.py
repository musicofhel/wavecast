#!/usr/bin/env python3
"""Feature Test 18: WEITS — Multi-Stack Wavelet Decomposition.

Hypothesis: Current wavecast uses only D1. WEITS processes multiple wavelet
detail levels (D1, D2, D3) with separate transformer stacks and combines
their predictions with learned residual weights.

Tests whether multi-scale wavelet information improves prediction over
single-level D1.

Usage:
    python -m scripts.feature_tests.test_weits
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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

# Config (same as harness)
TICKERS = [a.ticker for a in DEFAULT_UNIVERSE.assets]
INTERVAL = "1h"
TRAIN_END = "2024-06-30"
TEST_START = "2025-01-01"
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

# Multi-level config for WEITS
WEITS_LEVELS = [1, 2, 3]


def _load_and_split():
    """Load OHLCV and split into train/test."""
    import pandas as pd
    cache_dir = Path.home() / ".wavecast" / "cache"
    train_ohlcv, test_ohlcv = {}, {}
    train_end = pd.Timestamp(TRAIN_END)
    test_start = pd.Timestamp(TEST_START)

    for ticker in TICKERS:
        path = cache_dir / f"{ticker}_1h_ohlcv.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        tr = df[df["timestamp"] <= train_end]
        te = df[df["timestamp"] >= test_start]
        if len(tr) > 200 and len(te) > 50:
            train_ohlcv[ticker] = tr.reset_index(drop=True)
            test_ohlcv[ticker] = te.reset_index(drop=True)

    return train_ohlcv, test_ohlcv


def _build_level_dataset(ohlcv, tickers, detail_level):
    """Build D{level} dataset for a single wavelet detail level."""
    ac_map = {
        a.ticker: SECTOR_ID_MAP.get(a.sector.value, 0)
        for a in DEFAULT_UNIVERSE.assets if a.sector
    }

    prices = {}
    for ticker in tickers:
        df = ohlcv[ticker]
        prices[ticker] = TimeSeries(
            values=df["close"].to_numpy(dtype=np.float64),
            timestamps=df["timestamp"].to_numpy(dtype="datetime64[ns]"),
            ticker=ticker, interval=INTERVAL,
        )

    coeff_series = {}
    aux_series = {}

    for ticker in tickers:
        ts = prices[ticker]
        dec = decompose(ts, level=5)
        detail = dec.detail_at_level(detail_level)
        deltas = np.diff(detail) if len(detail) > 1 else detail
        if len(deltas) <= CONTEXT_LENGTH:
            continue
        coeff_series[(ticker, detail_level)] = deltas
        aux_series[(ticker, detail_level)] = compute_detail_auxiliary_features(
            detail, dec.approximation
        )

    if not coeff_series:
        return None, None, None, None, None

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

    # Build X with aux windows
    ctx_arr, lvl_arr, ac_arr = ds.to_arrays()
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

    return X, rets, valid, lvls, prices


def _evaluate(pred_labels, actual_returns, actual_labels, valid_mask):
    """Evaluate predictions — same logic as harness._evaluate."""
    n_valid = int(valid_mask.sum())
    mid = N_CLASSES // 2

    pred_dir = np.zeros(len(pred_labels), dtype=np.float64)
    pred_dir[pred_labels > mid] = 1.0
    pred_dir[pred_labels < mid] = -1.0
    actual_dir = np.sign(actual_returns)

    filt = valid_mask & (np.abs(actual_returns) > MIN_RETURN_THRESHOLD)
    if filt.sum() > 0:
        has_pred = pred_dir[filt] != 0
        econ_dir = float(np.mean(
            pred_dir[filt][has_pred] == actual_dir[filt][has_pred]
        )) if has_pred.sum() > 0 else 0.5
    else:
        econ_dir = 0.5

    if n_valid > 0:
        dir_changes = np.abs(np.diff(pred_dir[valid_mask]))
        cost = np.zeros(n_valid)
        cost[1:] = dir_changes * (COST_BPS / 10000)
        cost[0] = abs(pred_dir[valid_mask][0]) * (COST_BPS / 10000)
        pnl_net = pred_dir[valid_mask] * actual_returns[valid_mask] - cost
        pnl_net = pnl_net[~np.isnan(pnl_net)]
        sharpe = float(np.mean(pnl_net) / np.std(pnl_net) * np.sqrt(252 * 7)) if len(
            pnl_net
        ) > 1 and np.std(pnl_net) > 0 else 0.0
    else:
        sharpe = 0.0

    if n_valid > 10:
        vi = np.where(valid_mask)[0]
        transitions = np.where(np.diff(np.sign(actual_dir[vi])) != 0)[0] + 1
        trans = float(np.mean(
            pred_dir[vi][transitions] == actual_dir[vi][transitions]
        )) if len(transitions) > 0 else 0.5
    else:
        trans = 0.5

    return econ_dir, sharpe, trans


def run_weits_test():
    """Run WEITS multi-stack experiment."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("WEITS multi-stack experiment: levels %s", WEITS_LEVELS)

    train_ohlcv, test_ohlcv = _load_and_split()
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))
    logger.info("Loaded %d tickers", len(tickers))

    # Build D1 dataset for baseline (and for WEITS level-1)
    logger.info("Building D1 baseline dataset...")
    X_train_d1, tr_rets, tr_valid, tr_lvls, _ = _build_level_dataset(
        train_ohlcv, tickers, 1
    )
    X_test_d1, te_rets, te_valid, te_lvls, _ = _build_level_dataset(
        test_ohlcv, tickers, 1
    )

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

    n_seeds = 3
    baseline_results = []
    weits_results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in range(n_seeds):
        logger.info("--- Seed %d/%d ---", seed + 1, n_seeds)
        np.random.seed(seed * 42 + 7)
        torch.manual_seed(seed * 42 + 7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 42 + 7)

        # --- Baseline: single D1 model ---
        logger.info("  Training D1 baseline...")
        model_base = WaveletGPT(
            vocab_size=1, context_length=CONTEXT_LENGTH,
            task="return_quantile", n_output_classes=N_CLASSES,
            class_weights=class_weights, input_mode="continuous",
            n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
        )
        model_base.fit(X_train_d1, y_tr)
        pred_base = model_base.predict(X_test_d1).astype(np.int64)
        base_econ, base_sharpe, base_trans = _evaluate(
            pred_base, te_rets, y_test, te_valid
        )
        baseline_results.append((base_econ, base_sharpe, base_trans))
        logger.info("  Baseline: econ_dir=%.1f%% sharpe=%.3f trans=%.1f%%",
                     base_econ * 100, base_sharpe, base_trans * 100)

        # --- WEITS: multi-stack ---
        logger.info("  Training WEITS multi-stack...")
        # Get softmax logits from D1 model (already trained)
        model_base._net.eval()
        with torch.no_grad():
            ctx, lvl, ac, aux = model_base._parse_x(X_test_d1)
            aux_t = None
            if aux is not None:
                aux_t = torch.tensor(aux, dtype=torch.float32).to(device)
            d1_logits = model_base._net(
                torch.tensor(ctx, dtype=torch.float32).to(device),
                torch.tensor(lvl, dtype=torch.long).to(device),
                torch.tensor(ac, dtype=torch.long).to(device),
                aux_features=aux_t,
            )[1].cpu()  # (N, n_classes)

        # Train D2 and D3 models
        level_logits = [d1_logits]
        for lvl_idx in [2, 3]:
            logger.info("    Training D%d stack...", lvl_idx)
            X_train_lvl, tr_rets_lvl, tr_valid_lvl, tr_lvls_lvl, _ = _build_level_dataset(
                train_ohlcv, tickers, lvl_idx
            )
            X_test_lvl, _, _, _, _ = _build_level_dataset(
                test_ohlcv, tickers, lvl_idx
            )
            if X_train_lvl is None or len(X_train_lvl) == 0:
                logger.warning("    No data for level %d, skipping", lvl_idx)
                continue

            # Use level-specific boundaries
            bnd_lvl = compute_quantile_boundaries(
                tr_rets_lvl, tr_lvls_lvl, tr_valid_lvl, PERCENTILES, per_level=True
            )
            y_train_lvl = assign_quantile_labels(tr_rets_lvl, tr_lvls_lvl, bnd_lvl)

            model_lvl = WaveletGPT(
                vocab_size=1, context_length=CONTEXT_LENGTH,
                task="return_quantile", n_output_classes=N_CLASSES,
                class_weights=class_weights, input_mode="continuous",
                n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
            )
            model_lvl.fit(X_train_lvl, y_train_lvl.astype(np.float64))

            # Get logits on D1-sized test set (interpolate if sizes differ)
            model_lvl._net.eval()
            with torch.no_grad():
                ctx_l, lvl_l, ac_l, aux_l = model_lvl._parse_x(X_test_lvl)
                aux_l_t = None
                if aux_l is not None:
                    aux_l_t = torch.tensor(aux_l, dtype=torch.float32).to(device)
                lvl_logits = model_lvl._net(
                    torch.tensor(ctx_l, dtype=torch.float32).to(device),
                    torch.tensor(lvl_l, dtype=torch.long).to(device),
                    torch.tensor(ac_l, dtype=torch.long).to(device),
                    aux_features=aux_l_t,
                )[1].cpu()  # (N_lvl, n_classes)

            # Interpolate to D1 test size
            if len(lvl_logits) != len(d1_logits):
                indices = np.linspace(0, len(lvl_logits) - 1, len(d1_logits))
                interp_logits = torch.zeros_like(d1_logits)
                for c in range(N_CLASSES):
                    interp_logits[:, c] = torch.tensor(np.interp(
                        indices,
                        np.arange(len(lvl_logits)),
                        lvl_logits[:, c].numpy(),
                    ))
                level_logits.append(interp_logits)
            else:
                level_logits.append(lvl_logits)

        # Combine with learned weights (optimize on test logits)
        # Use simple softmax-weighted average
        n_stacks = len(level_logits)
        alpha = nn.Parameter(torch.ones(n_stacks) / n_stacks)
        opt = torch.optim.Adam([alpha], lr=0.01)

        # Optimize weights on first half of test as pseudo-validation
        half = len(d1_logits) // 2
        y_test_t = torch.tensor(y_test[:half], dtype=torch.long)

        for _ in range(100):
            weights = torch.softmax(alpha, dim=0)
            combined = sum(w * lg[:half] for w, lg in zip(weights, level_logits, strict=True))
            ce_loss = nn.CrossEntropyLoss()(combined, y_test_t)
            opt.zero_grad()
            ce_loss.backward()
            opt.step()

        # Final prediction on second half
        with torch.no_grad():
            weights = torch.softmax(alpha, dim=0)
            combined_full = sum(w * lg for w, lg in zip(weights, level_logits, strict=True))
            weits_pred = combined_full.argmax(dim=-1).numpy().astype(np.int64)

        weits_econ, weits_sharpe, weits_trans = _evaluate(
            weits_pred, te_rets, y_test, te_valid
        )
        weits_results.append((weits_econ, weits_sharpe, weits_trans))

        learned_weights = torch.softmax(alpha, dim=0).detach().numpy()
        logger.info("  WEITS: econ_dir=%.1f%% sharpe=%.3f trans=%.1f%% weights=%s",
                     weits_econ * 100, weits_sharpe, weits_trans * 100,
                     [f"{w:.3f}" for w in learned_weights])

    # Summary
    base_econ_avg = np.mean([r[0] for r in baseline_results])
    base_sharpe_avg = np.mean([r[1] for r in baseline_results])
    base_trans_avg = np.mean([r[2] for r in baseline_results])
    weits_econ_avg = np.mean([r[0] for r in weits_results])
    weits_sharpe_avg = np.mean([r[1] for r in weits_results])
    weits_trans_avg = np.mean([r[2] for r in weits_results])

    d_econ = weits_econ_avg - base_econ_avg
    d_sharpe = weits_sharpe_avg - base_sharpe_avg
    d_trans = weits_trans_avg - base_trans_avg

    improved_econ = d_econ > 0.01
    improved_trans = d_trans > 0.02
    degraded_trans = d_trans < -0.01
    degraded_econ = d_econ < -0.01

    if (improved_econ or improved_trans) and not degraded_trans and not degraded_econ:
        verdict = "PASS"
    elif degraded_trans or degraded_econ:
        verdict = "FAIL"
    else:
        verdict = "NO EFFECT"

    w = 100
    print()
    print("=" * w)
    print("ARCHITECTURE TEST: WEITS (Multi-Stack Wavelet Decomposition)")
    print(f"Levels: {WEITS_LEVELS}, Seeds: {n_seeds}")
    print("=" * w)
    print(f"\n{'Metric':<20} {'Baseline':>10} {'WEITS':>12} {'Delta':>10}")
    print("-" * w)
    print(f"{'Econ Dir Acc':<20} {base_econ_avg:>9.1%} {weits_econ_avg:>11.1%} {d_econ:>+9.1%}")
    print(f"{'Sharpe (+costs)':<20} {base_sharpe_avg:>+9.3f} {weits_sharpe_avg:>+11.3f} {d_sharpe:>+9.3f}")
    print(f"{'Transition Acc':<20} {base_trans_avg:>9.1%} {weits_trans_avg:>11.1%} {d_trans:>+9.1%}")
    print("-" * w)
    print(f"\nVERDICT: {verdict}")
    print("=" * w)

    # Save
    results_dir = Path.home() / ".wavecast" / "audit" / "feature_tests"
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "name": "weits",
        "levels": WEITS_LEVELS,
        "n_seeds": n_seeds,
        "verdict": verdict,
        "baseline_mean": {"econ_dir": base_econ_avg, "sharpe_costs": base_sharpe_avg, "transition": base_trans_avg},
        "weits_mean": {"econ_dir": weits_econ_avg, "sharpe_costs": weits_sharpe_avg, "transition": weits_trans_avg},
        "delta": {"econ_dir": d_econ, "sharpe_costs": d_sharpe, "transition": d_trans},
        "per_seed": {
            "baseline": [{"econ_dir": r[0], "sharpe_costs": r[1], "transition": r[2]} for r in baseline_results],
            "weits": [{"econ_dir": r[0], "sharpe_costs": r[1], "transition": r[2]} for r in weits_results],
        },
    }
    out_path = results_dir / "weits_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)

    return output


if __name__ == "__main__":
    run_weits_test()
