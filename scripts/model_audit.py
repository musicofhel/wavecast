#!/usr/bin/env python3
"""WaveCast Model Audit — Stress-test reported accuracy metrics.

Answers 13 questions about whether 60.8% token accuracy and 95.8% directional
accuracy represent real economic value or are inflated by persistence prediction
and symbolic (non-price-based) directional accuracy.

Usage:
    cd ~/wavecast && source .venv/bin/activate
    python scripts/model_audit.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wavecast.core.config import SAXConfig
from wavecast.core.types import TimeSeries
from wavecast.core.universe import DEFAULT_UNIVERSE
from wavecast.data.cache import ParquetCache
from wavecast.experiments.runner import ASSET_CLASS_ID_MAP, SECTOR_ID_MAP
from wavecast.experiments.splitter import walk_forward_split
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.sax.bow import extract_words
from wavecast.sax.sax import sax_transform
from wavecast.signals.backtest import SignalBacktest
from wavecast.signals.costs import TransactionCostModel
from wavecast.signals.generator import SignalGenerator
from wavecast.signals.position import PositionSizer
from wavecast.signals.types import SignalSeries, TradingSignal
from wavecast.tokenizer.vocabulary import PAD_ID, SAXVocabulary
from wavecast.wavelets.dwt import decompose

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Constants (match train_forward_model.py) ---
ALPHABET_SIZE = 7
N_SEGMENTS = 256
WORD_LENGTH = 4
WORD_STRIDE = 1
DWT_LEVELS = [1, 2, 5]
CACHE_DIR = Path.home() / ".wavecast" / "cache"
MODEL_DIR = Path.home() / ".wavecast" / "models" / "forward_ready"
OUTPUT_DIR = Path.home() / ".wavecast" / "audit"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
COST_SPREAD_BPS = 5.0
COST_SLIPPAGE_BPS = 2.0
COST_TOTAL_BPS = COST_SPREAD_BPS + COST_SLIPPAGE_BPS


def build_asset_class_map(tickers: list[str]) -> dict[str, int]:
    from wavecast.core.universe import LEGACY_UNIVERSE

    m: dict[str, int] = {}
    for a in LEGACY_UNIVERSE.assets:
        m[a.ticker] = ASSET_CLASS_ID_MAP.get(a.asset_class.value, 0)
    for a in DEFAULT_UNIVERSE.assets:
        if a.sector is not None:
            m[a.ticker] = SECTOR_ID_MAP.get(a.sector.value, 0)
    return {t: m.get(t, 0) for t in tickers}


def econ_direction(token: int, vocab_size: int) -> int:
    """Map token to economic direction: +1 UP, -1 DOWN, 0 FLAT."""
    mid = vocab_size // 2
    quarter = vocab_size // 4
    if token >= mid + quarter:
        return 1
    elif token <= mid - quarter:
        return -1
    return 0


# ============================================================
# Pipeline Reproduction with Metadata
# ============================================================

def reproduce_pipeline(
    train_prices: dict[str, TimeSeries],
    test_prices: dict[str, TimeSeries],
    common_tickers: list[str],
    vocabulary: SAXVocabulary,
    context_length: int,
) -> tuple[NDArray, NDArray, dict]:
    """Reproduce experiment pipeline tracking per-sample metadata.

    Returns (X_test_full, y_test, meta).
    """
    sax_config = SAXConfig(
        n_segments=N_SEGMENTS, alphabet_size=ALPHABET_SIZE,
        word_length=WORD_LENGTH, word_stride=WORD_STRIDE,
    )
    dwt_level = 5
    asset_class_map = build_asset_class_map(common_tickers)

    # --- Train targets (for Q9) ---
    train_targets: list[int] = []
    for ticker in common_tickers:
        train_decomp = decompose(train_prices[ticker], level=dwt_level)
        for lvl in DWT_LEVELS:
            coeffs = train_decomp.detail_at_level(lvl)
            if len(coeffs) < 2:
                continue
            n_seg = min(sax_config.n_segments, len(coeffs))
            sax_res = sax_transform(coeffs, n_seg, sax_config.alphabet_size)
            words = extract_words(sax_res.symbols, sax_config.word_length, sax_config.word_stride)
            token_ids = vocabulary.encode_sequence(words)
            if len(token_ids) <= context_length:
                padded = [PAD_ID] * (context_length + 1 - len(token_ids)) + token_ids
                token_ids = padded
            for i in range(len(token_ids) - context_length):
                train_targets.append(token_ids[i + context_length])

    # --- Test data with metadata ---
    test_ctx: list[list[int]] = []
    test_tgt: list[int] = []
    test_lvl: list[int] = []
    test_ac: list[int] = []
    m_tickers: list[str] = []
    m_token_pos: list[int] = []
    m_bar_idx: list[int] = []
    m_returns: list[float] = []
    token_seqs: dict[tuple[str, int], list[int]] = {}

    for ticker in common_tickers:
        ac_id = asset_class_map.get(ticker, 0)
        test_decomp = decompose(test_prices[ticker], level=dwt_level)
        test_values = test_prices[ticker].values

        for lvl in DWT_LEVELS:
            coeffs = test_decomp.detail_at_level(lvl)
            if len(coeffs) < 2:
                continue
            n_coeffs = len(coeffs)
            n_seg = min(sax_config.n_segments, n_coeffs)
            sax_res = sax_transform(coeffs, n_seg, sax_config.alphabet_size)
            n_symbols = len(sax_res.symbols)
            words = extract_words(sax_res.symbols, sax_config.word_length, sax_config.word_stride)
            token_ids = vocabulary.encode_sequence(words)
            token_seqs[(ticker, lvl)] = list(token_ids)

            if len(token_ids) <= context_length:
                continue

            for i in range(len(token_ids) - context_length):
                ctx = token_ids[i : i + context_length]
                tgt = token_ids[i + context_length]
                token_pos = i + context_length

                # Temporal mapping
                coeff_idx = token_pos * n_coeffs // n_symbols if n_symbols > 0 else 0
                bar_start = coeff_idx * (2 ** lvl)
                bar_end = bar_start + (2 ** lvl)

                if (bar_end < len(test_values) and bar_start < len(test_values)
                        and test_values[bar_start] != 0):
                    ret = float(
                        (test_values[bar_end] - test_values[bar_start])
                        / test_values[bar_start]
                    )
                else:
                    ret = float("nan")

                test_ctx.append(ctx)
                test_tgt.append(tgt)
                test_lvl.append(lvl)
                test_ac.append(ac_id)
                m_tickers.append(ticker)
                m_token_pos.append(token_pos)
                m_bar_idx.append(bar_start)
                m_returns.append(ret)

    X_test = np.array(test_ctx, dtype=np.int64)
    y_test = np.array(test_tgt, dtype=np.int64)
    levels_arr = np.array(test_lvl, dtype=np.int64)
    ac_arr = np.array(test_ac, dtype=np.int64)
    X_test_full = np.column_stack([X_test, levels_arr, ac_arr])

    meta = {
        "tickers": m_tickers,
        "levels": levels_arr,
        "token_positions": np.array(m_token_pos, dtype=np.int64),
        "bar_indices": np.array(m_bar_idx, dtype=np.int64),
        "actual_returns": np.array(m_returns, dtype=np.float64),
        "token_seqs": token_seqs,
        "y_train": np.array(train_targets, dtype=np.int64),
        "X_test": X_test,
    }
    return X_test_full, y_test, meta


# ============================================================
# Phase 0: Temporal Mapping Validation
# ============================================================

def validate_temporal_mapping(meta: dict, test_prices: dict[str, TimeSeries]) -> dict:
    returns = meta["actual_returns"]
    tickers = meta["tickers"]
    levels = meta["levels"]
    bar_indices = meta["bar_indices"]
    valid = np.where(~np.isnan(returns))[0]

    # Pick 5 diverse samples
    seen: set[int] = set()
    picks: list[int] = []
    for idx in valid:
        lvl = int(levels[idx])
        if lvl not in seen and len(picks) < 5:
            seen.add(lvl)
            picks.append(int(idx))
    seen_tk: set[str] = {tickers[i] for i in picks}
    for idx in valid:
        if len(picks) >= 5:
            break
        if tickers[int(idx)] not in seen_tk:
            picks.append(int(idx))
            seen_tk.add(tickers[int(idx)])

    samples = []
    for idx in picks[:5]:
        tk = tickers[idx]
        lvl = int(levels[idx])
        bi = int(bar_indices[idx])
        tv = test_prices[tk].values
        be = bi + (2 ** lvl)
        samples.append({
            "sample_idx": idx, "ticker": tk, "level": lvl,
            "bar_idx": bi, "bar_end": be, "n_bars": 2 ** lvl,
            "price_start": round(float(tv[bi]), 2) if bi < len(tv) else None,
            "price_end": round(float(tv[be]), 2) if be < len(tv) else None,
            "actual_return": round(float(returns[idx]), 6),
            "series_len": len(tv),
        })
    return {"samples": samples}


# ============================================================
# Q1: Filtered Directional Accuracy
# ============================================================

def q1_filtered_directional(
    predicted: NDArray, y_test: NDArray, meta: dict, vs: int,
) -> dict:
    rets = meta["actual_returns"]
    valid = ~np.isnan(rets)
    rows = []
    for thresh in [0.0, 0.001, 0.0025, 0.005, 0.01]:
        mask = valid & (np.abs(rets) > thresh)
        n = int(np.sum(mask))
        if n == 0:
            rows.append({"threshold": thresh, "n": 0, "econ": None, "symbolic": None})
            continue
        pd = np.array([econ_direction(int(p), vs) for p in predicted[mask]])
        ad = np.sign(rets[mask]).astype(int)
        em = (pd != 0) & (ad != 0)
        econ = float(np.mean(pd[em] == ad[em])) if np.sum(em) > 0 else None
        sym = float(np.mean(predicted[mask] == y_test[mask]))
        rows.append({"threshold": thresh, "n": n, "n_dir": int(np.sum(em)),
                      "econ": econ, "symbolic": sym})
    return {"rows": rows}


# ============================================================
# Q2: Near-Zero Return Distribution
# ============================================================

def q2_near_zero(meta: dict) -> dict:
    rets = meta["actual_returns"]
    levels = meta["levels"]
    valid = ~np.isnan(rets)
    ar = np.abs(rets)
    threshs = [0.001, 0.0025, 0.005]
    overall = {f"<{t*100:.1f}%": float(np.mean(ar[valid] < t)) * 100 for t in threshs}
    by_level: dict[int, dict] = {}
    for lvl in sorted(set(int(lv) for lv in levels)):
        lm = valid & (levels == lvl)
        by_level[lvl] = {f"<{t*100:.1f}%": float(np.mean(ar[lm] < t)) * 100 for t in threshs}
    return {"overall": overall, "by_level": by_level,
            "n_valid": int(np.sum(valid)), "n_total": len(rets)}


# ============================================================
# Q3: Level 5 Token Change Frequency
# ============================================================

def q3_level5_freq(meta: dict) -> dict:
    token_seqs = meta["token_seqs"]
    per_ticker: dict[str, dict] = {}
    for (ticker, lvl), tokens in token_seqs.items():
        if lvl != 5 or len(tokens) < 2:
            continue
        runs = []
        cur = 1
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i - 1]:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
        same = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i - 1])
        per_ticker[ticker] = {
            "n_tokens": len(tokens),
            "mean_run": round(float(np.mean(runs)), 1),
            "median_run": float(np.median(runs)),
            "pct_same": round(same / (len(tokens) - 1) * 100, 1),
            "n_unique": len(set(tokens)),
        }
    agg_pct = np.mean([v["pct_same"] for v in per_ticker.values()]) if per_ticker else 0
    agg_run = np.mean([v["mean_run"] for v in per_ticker.values()]) if per_ticker else 0
    return {"per_ticker": per_ticker, "agg_pct_same": round(float(agg_pct), 1),
            "agg_mean_run": round(float(agg_run), 1)}


# ============================================================
# Q4: Persistence Decomposition
# ============================================================

def q4_persistence(
    predicted: NDArray, y_test: NDArray, meta: dict, vs: int,
) -> dict:
    X_test = meta["X_test"]
    rets = meta["actual_returns"]
    valid = ~np.isnan(rets)
    ctx_last = X_test[:, -1]
    pers_mask = predicted == ctx_last
    change_mask = ~pers_mask

    def stats(mask: NDArray, label: str) -> dict:
        n = int(np.sum(mask))
        if n == 0:
            return {"label": label, "count": 0, "pct": 0.0}
        ta = float(np.mean(predicted[mask] == y_test[mask]))
        both = mask & valid
        nv = int(np.sum(both))
        econ = None
        if nv > 0:
            pd = np.array([econ_direction(int(p), vs) for p in predicted[both]])
            ad = np.sign(rets[both]).astype(int)
            em = (pd != 0) & (ad != 0)
            econ = float(np.mean(pd[em] == ad[em])) if np.sum(em) > 0 else None
        return {"label": label, "count": n, "pct": round(n / len(predicted) * 100, 1),
                "token_acc": round(ta, 4), "econ_dir": econ, "n_valid": nv}

    return {"persistence": stats(pers_mask, "persistence"),
            "change": stats(change_mask, "change"), "total": len(predicted)}


# ============================================================
# Q5: Subsampled Accuracy
# ============================================================

def q5_subsampled(predicted: NDArray, y_test: NDArray) -> dict:
    rows = []
    for stride in [1, 2, 4, 8, 16]:
        idx = np.arange(0, len(predicted), stride)
        acc = float(np.mean(predicted[idx] == y_test[idx]))
        rows.append({"stride": stride, "n": len(idx), "token_acc": round(acc, 4)})
    return {"rows": rows}


# ============================================================
# Q6 + Q11 + Q13: Signal Backtest
# ============================================================

def _run_bt(
    predicted: NDArray, proba: NDArray, returns: NDArray,
    vs: int, label: str, sizing: str,
) -> dict:
    n = len(predicted)
    if n < 10:
        return {"label": label, "sizing": sizing, "n": n, "error": "too few"}
    gen = SignalGenerator(vocab_size=vs, alphabet_size=ALPHABET_SIZE)
    base = np.datetime64("2025-01-01")
    ts = np.array([base + np.timedelta64(i, "h") for i in range(n)])
    signals = gen.generate(proba, predicted, ts, "audit")
    ar = returns.copy()
    nan_m = np.isnan(ar)
    ar[nan_m] = 0.0
    cost = TransactionCostModel(commission_rate=0.0,
                                spread_bps=COST_SPREAD_BPS, slippage_bps=COST_SLIPPAGE_BPS)
    bt = SignalBacktest(cost_model=cost, position_sizer=PositionSizer(method=sizing))
    res = bt.run(signals, ar, ts)
    tr = np.array([t.net_return for t in res.trades]) if res.trades else np.array([])
    gr = np.array([t.gross_return for t in res.trades]) if res.trades else np.array([])
    pct_clear = float(np.mean(np.abs(gr) > COST_TOTAL_BPS / 10000)) * 100 if len(gr) > 0 else 0
    return {
        "label": label, "sizing": sizing, "n": n, "n_nan": int(np.sum(nan_m)),
        "sharpe": round(res.metrics.get("sharpe_ratio", 0.0), 4),
        "total_return": round(res.metrics.get("total_return", 0.0), 6),
        "max_dd": round(res.metrics.get("max_drawdown", 0.0), 6),
        "win_rate": round(res.metrics.get("win_rate", 0.0), 4),
        "n_trades": int(res.metrics.get("num_trades", 0)),
        "per_trade": {
            "mean": round(float(np.mean(tr)), 6) if len(tr) > 0 else 0,
            "median": round(float(np.median(tr)), 6) if len(tr) > 0 else 0,
            "p25": round(float(np.percentile(tr, 25)), 6) if len(tr) > 0 else 0,
            "p75": round(float(np.percentile(tr, 75)), 6) if len(tr) > 0 else 0,
            "pct_clear_7bps": round(pct_clear, 1),
        },
    }


def q6_backtest(predicted: NDArray, proba: NDArray, meta: dict, vs: int) -> dict:
    valid = ~np.isnan(meta["actual_returns"])
    p, pr, r = predicted[valid], proba[valid], meta["actual_returns"][valid]
    return {
        "fixed": _run_bt(p, pr, r, vs, "all", "fixed"),
        "kelly": _run_bt(p, pr, r, vs, "all", "fractional_kelly"),
    }


def q11_level_isolation(predicted: NDArray, proba: NDArray, meta: dict, vs: int) -> dict:
    levels = meta["levels"]
    valid = ~np.isnan(meta["actual_returns"])
    out: dict[str, dict] = {}
    for label, lvls in [("level5", {5}), ("levels12", {1, 2}), ("all", {1, 2, 5})]:
        mask = valid & np.isin(levels, list(lvls))
        out[label] = _run_bt(
            predicted[mask], proba[mask], meta["actual_returns"][mask], vs, label, "fixed",
        )
    return out


# ============================================================
# Q7: Direction-Change Accuracy
# ============================================================

def q7_dir_change(predicted: NDArray, meta: dict, vs: int) -> dict:
    rets = meta["actual_returns"]
    tickers = meta["tickers"]
    levels = meta["levels"]
    valid = ~np.isnan(rets)
    pred_dir = np.array([econ_direction(int(p), vs) for p in predicted])
    actual_dir = np.sign(rets)

    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i in range(len(predicted)):
        groups[(tickers[i], int(levels[i]))].append(i)

    trans_c, trans_t, cont_c, cont_t = 0, 0, 0, 0
    for indices in groups.values():
        for j in range(1, len(indices)):
            idx, prev = indices[j], indices[j - 1]
            if not valid[idx] or not valid[prev]:
                continue
            if actual_dir[idx] == 0 or actual_dir[prev] == 0 or pred_dir[idx] == 0:
                continue
            if actual_dir[idx] != actual_dir[prev]:
                trans_t += 1
                if pred_dir[idx] == int(actual_dir[idx]):
                    trans_c += 1
            else:
                cont_t += 1
                if pred_dir[idx] == int(actual_dir[idx]):
                    cont_c += 1

    return {
        "transition": {"n": trans_t, "acc": round(trans_c / trans_t, 4) if trans_t else None},
        "continuation": {"n": cont_t, "acc": round(cont_c / cont_t, 4) if cont_t else None},
    }


# ============================================================
# Q8: Token Confusion Matrix
# ============================================================

def q8_confusion(predicted: NDArray, y_test: NDArray, proba: NDArray, vs: int) -> dict:
    confusion: Counter[tuple[int, int]] = Counter()
    for p, a in zip(predicted, y_test, strict=True):
        if p != a:
            confusion[(int(p), int(a))] += 1
    top10 = [{"pred": p, "actual": a, "count": c} for (p, a), c in confusion.most_common(10)]
    eps = 1e-10
    ent = -np.sum(proba * np.log(proba + eps), axis=1)
    return {
        "top10": top10, "mean_entropy": round(float(np.mean(ent)), 3),
        "median_entropy": round(float(np.median(ent)), 3),
        "max_entropy": round(float(np.log(vs)), 3),
        "n_unique_pred": len(set(int(p) for p in predicted)),
        "n_unique_actual": len(set(int(a) for a in y_test)),
    }


# ============================================================
# Q9: Token Concentration
# ============================================================

def q9_concentration(y_train: NDArray) -> dict:
    ctr = Counter(int(t) for t in y_train if int(t) > 1)
    total = sum(ctr.values())
    sorted_c = sorted(ctr.values(), reverse=True)
    cs = np.cumsum(sorted_c)
    coverage = {}
    for pct in [0.5, 0.75, 0.9, 0.95]:
        coverage[f"{pct*100:.0f}%"] = int(np.searchsorted(cs, total * pct) + 1)
    top10 = [{"id": k, "count": v, "pct": round(v / total * 100, 1)}
             for k, v in ctr.most_common(10)]
    return {"total": total, "unique": len(ctr), "coverage": coverage, "top10": top10}


# ============================================================
# Q10: Confidence Calibration
# ============================================================

def q10_calibration(
    predicted: NDArray, y_test: NDArray, proba: NDArray, meta: dict, vs: int,
) -> dict:
    rets = meta["actual_returns"]
    valid = ~np.isnan(rets)
    max_conf = np.max(proba, axis=1)
    edges = np.percentile(max_conf, np.arange(0, 101, 10))
    rows = []
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        mask = (max_conf >= lo) & (max_conf <= hi if i == 9 else max_conf < hi)
        n = int(np.sum(mask))
        if n == 0:
            continue
        mc = round(float(np.mean(max_conf[mask])), 4)
        ta = round(float(np.mean(predicted[mask] == y_test[mask])), 4)
        both = mask & valid
        econ = None
        if np.sum(both) > 0:
            pd = np.array([econ_direction(int(p), vs) for p in predicted[both]])
            ad = np.sign(rets[both]).astype(int)
            em = (pd != 0) & (ad != 0)
            econ = round(float(np.mean(pd[em] == ad[em])), 4) if np.sum(em) > 0 else None
        rows.append({"decile": i + 1, "range": f"[{lo:.3f},{hi:.3f}]",
                      "n": n, "mean_conf": mc, "token_acc": ta, "econ_dir": econ})
    return {"deciles": rows}


# ============================================================
# Q12: Random Model Baseline
# ============================================================

def q12_random_baseline(predicted: NDArray, proba: NDArray, meta: dict, vs: int) -> dict:
    rets = meta["actual_returns"]
    valid = ~np.isnan(rets)
    p_v, pr_v, r_v = predicted[valid], proba[valid], rets[valid].copy()
    r_v = np.where(np.isnan(r_v), 0.0, r_v)
    n = len(p_v)
    if n < 10:
        return {"error": "too few samples"}

    gen = SignalGenerator(vocab_size=vs, alphabet_size=ALPHABET_SIZE)
    base = np.datetime64("2025-01-01")
    ts = np.array([base + np.timedelta64(i, "h") for i in range(n)])
    real_sigs = gen.generate(pr_v, p_v, ts, "audit")
    real_dirs = real_sigs.directions
    real_confs = real_sigs.confidences

    dir_vals, dir_counts = np.unique(real_dirs, return_counts=True)
    dir_probs = dir_counts / dir_counts.sum()

    cost = TransactionCostModel(commission_rate=0.0,
                                spread_bps=COST_SPREAD_BPS, slippage_bps=COST_SLIPPAGE_BPS)
    rng = np.random.default_rng(42)
    rand_sharpes: list[float] = []

    for _ in range(100):
        dirs = rng.choice(dir_vals, size=n, p=dir_probs)
        confs = rng.permutation(real_confs)
        sigs = SignalSeries(
            signals=[
                TradingSignal(timestamp=ts[i], direction=int(dirs[i]),
                              confidence=float(confs[i]), raw_probability=float(confs[i]),
                              token_id=0)
                for i in range(n)
            ], ticker="random",
        )
        bt = SignalBacktest(cost_model=cost, position_sizer=PositionSizer(method="fixed"))
        res = bt.run(sigs, r_v, ts)
        rand_sharpes.append(res.metrics.get("sharpe_ratio", 0.0))

    rs = np.array(rand_sharpes)
    # Real model Sharpe
    real_bt = SignalBacktest(cost_model=cost, position_sizer=PositionSizer(method="fixed"))
    real_res = real_bt.run(real_sigs, r_v, ts)
    real_s = real_res.metrics.get("sharpe_ratio", 0.0)

    return {
        "real_sharpe": round(real_s, 4),
        "rand_mean": round(float(np.mean(rs)), 4),
        "rand_std": round(float(np.std(rs)), 4),
        "rand_p5": round(float(np.percentile(rs, 5)), 4),
        "rand_p95": round(float(np.percentile(rs, 95)), 4),
        "exceeds_p95": bool(real_s > np.percentile(rs, 95)),
        "percentile": round(float(np.mean(rs < real_s) * 100), 1),
    }


# ============================================================
# Report Printing
# ============================================================

def print_report(results: dict, vs: int) -> None:
    print("\n" + "=" * 70)
    print("WAVECAST MODEL AUDIT REPORT")
    print(f"Test samples: {results['n_test']}, Vocab: {vs}, "
          f"Token accuracy: {results['token_accuracy']:.4f}")
    print("=" * 70)

    print("\n--- Phase 0: Temporal Mapping Validation ---")
    for s in results["phase0"]["samples"]:
        print(f"  [{s['sample_idx']}] {s['ticker']} L{s['level']}: "
              f"bar {s['bar_idx']}→{s['bar_end']} ({s['n_bars']} bars), "
              f"${s['price_start']}→${s['price_end']}, ret={s['actual_return']:.4%}")

    print("\n--- Q1: Filtered Directional Accuracy ---")
    print(f"  {'Thresh':>8} {'N':>7} {'EconDir':>9} {'Symbolic':>9}")
    for r in results["q1"]["rows"]:
        e = f"{r['econ']:.3f}" if r["econ"] is not None else "  N/A"
        s = f"{r['symbolic']:.3f}" if r["symbolic"] is not None else "  N/A"
        print(f"  {r['threshold']:>8.2%} {r['n']:>7} {e:>9} {s:>9}")

    print("\n--- Q2: Near-Zero Return Distribution ---")
    q2 = results["q2"]
    print(f"  Valid: {q2['n_valid']}/{q2['n_total']}")
    for k, v in q2["overall"].items():
        print(f"  |ret| {k}: {v:.1f}%")
    for lvl, d in q2["by_level"].items():
        print(f"  L{lvl}: " + ", ".join(f"{k}:{v:.1f}%" for k, v in d.items()))

    print("\n--- Q3: Level 5 Token Change Frequency ---")
    q3 = results["q3"]
    print(f"  Aggregate: {q3['agg_pct_same']}% same, mean run={q3['agg_mean_run']}")
    for tk, d in list(q3["per_ticker"].items())[:5]:
        print(f"  {tk}: {d['pct_same']}% same, run={d['mean_run']}, {d['n_unique']} unique")

    print("\n--- Q4: Persistence Decomposition ---")
    q4 = results["q4"]
    for k in ["persistence", "change"]:
        b = q4[k]
        ta = f"{b['token_acc']:.4f}" if b.get("count", 0) > 0 else "N/A"
        ed = f"{b['econ_dir']:.4f}" if b.get("econ_dir") is not None else "N/A"
        print(f"  {k.upper():>12}: {b['count']:>6} ({b['pct']:.1f}%) "
              f"token_acc={ta} econ_dir={ed}")

    print("\n--- Q5: Subsampled Accuracy ---")
    for r in results["q5"]["rows"]:
        print(f"  stride={r['stride']:>2}: n={r['n']:>6}, acc={r['token_acc']:.4f}")

    print("\n--- Q6: Signal Backtest ---")
    for key in ["fixed", "kelly"]:
        b = results["q6"][key]
        if "error" in b:
            print(f"  {key}: {b['error']}")
            continue
        pt = b["per_trade"]
        print(f"  {key.upper():>6}: Sharpe={b['sharpe']:.4f} ret={b['total_return']:.4%} "
              f"DD={b['max_dd']:.4%} win={b['win_rate']:.3f} trades={b['n_trades']}")
        print(f"         per-trade: mean={pt['mean']:.6f} med={pt['median']:.6f} "
              f"P25={pt['p25']:.6f} P75={pt['p75']:.6f} >{COST_TOTAL_BPS}bps={pt['pct_clear_7bps']:.1f}%")

    print("\n--- Q7: Direction-Change Accuracy ---")
    q7 = results["q7"]
    for k in ["transition", "continuation"]:
        d = q7[k]
        a = f"{d['acc']:.4f}" if d["acc"] is not None else "N/A"
        print(f"  {k:>14}: n={d['n']:>6} acc={a}")

    print("\n--- Q8: Confusion & Entropy ---")
    q8 = results["q8"]
    print(f"  Entropy: {q8['mean_entropy']:.3f} mean / {q8['max_entropy']:.3f} max")
    print(f"  Unique pred: {q8['n_unique_pred']}, unique actual: {q8['n_unique_actual']}")
    for p in q8["top10"][:5]:
        print(f"    pred={p['pred']} actual={p['actual']}: {p['count']}x")

    print("\n--- Q9: Token Concentration ---")
    q9 = results["q9"]
    print(f"  {q9['total']} targets, {q9['unique']} unique tokens")
    for k, v in q9["coverage"].items():
        print(f"  {k} coverage: {v} tokens")

    print("\n--- Q10: Confidence Calibration ---")
    print(f"  {'Dec':>4} {'N':>6} {'Conf':>7} {'TokAcc':>7} {'EconDir':>8}")
    for r in results["q10"]["deciles"]:
        ed = f"{r['econ_dir']:.4f}" if r["econ_dir"] is not None else "  N/A"
        print(f"  {r['decile']:>4} {r['n']:>6} {r['mean_conf']:>7.4f} "
              f"{r['token_acc']:>7.4f} {ed:>8}")

    print("\n--- Q11: Level Isolation ---")
    for label, b in results["q11"].items():
        if "error" in b:
            print(f"  {label}: {b['error']}")
        else:
            print(f"  {label:>10}: Sharpe={b['sharpe']:.4f} ret={b['total_return']:.4%} "
                  f"win={b['win_rate']:.3f} trades={b['n_trades']}")

    print("\n--- Q12: Random Baseline ---")
    q12 = results["q12"]
    if "error" not in q12:
        print(f"  Real Sharpe:  {q12['real_sharpe']:.4f}")
        print(f"  Random:       {q12['rand_mean']:.4f} ± {q12['rand_std']:.4f}")
        print(f"  Random [P5,P95]: [{q12['rand_p5']:.4f}, {q12['rand_p95']:.4f}]")
        print(f"  Exceeds P95:  {q12['exceeds_p95']}")
        print(f"  Percentile:   {q12['percentile']:.1f}%")

    print("\n--- Q13: Fixed vs Kelly ---")
    f_s = results["q6"]["fixed"].get("sharpe", 0)
    k_s = results["q6"]["kelly"].get("sharpe", 0)
    print(f"  Fixed Sharpe: {f_s:.4f}")
    print(f"  Kelly Sharpe: {k_s:.4f}")
    if f_s > 0 and k_s <= 0:
        print("  → CALIBRATION PROBLEM: signal works but Kelly hurts")
    elif f_s > 0 and k_s > 0:
        print("  → Confidence signal exploitable")
    else:
        print("  → NO TRADEABLE EDGE")

    print("\n" + "=" * 70)


def save_json(results: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "model_audit_results.json"

    class Enc(json.JSONEncoder):
        def default(self, o: object) -> object:
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.bool_):
                return bool(o)
            return super().default(o)

    with open(path, "w") as f:
        json.dump(results, f, indent=2, cls=Enc)
    logger.info("Saved %s", path)


# ============================================================
# Main
# ============================================================

def main() -> None:
    t0 = time.monotonic()
    logger.info("=== WaveCast Model Audit ===")

    # Load model + vocab
    model = WaveletGPT.load(MODEL_DIR)
    vocabulary = SAXVocabulary.load(MODEL_DIR / "vocabulary.json")
    vs = vocabulary.size
    ctx_len = model._config["context_length"]
    logger.info("Model: vocab=%d ctx=%d", vs, ctx_len)

    # Load prices + split
    cache = ParquetCache(CACHE_DIR)
    prices = {}
    for tk in DEFAULT_UNIVERSE.tickers:
        ts = cache.get(tk, "1h")
        if ts is not None:
            prices[tk] = ts
    logger.info("Loaded %d tickers", len(prices))

    train_prices, test_prices = walk_forward_split(prices, TRAIN_END, TEST_START)
    common = sorted(set(train_prices) & set(test_prices))
    logger.info("Split: %d common tickers", len(common))

    # Reproduce pipeline
    logger.info("Reproducing pipeline...")
    X_test_full, y_test, meta = reproduce_pipeline(
        train_prices, test_prices, common, vocabulary, ctx_len,
    )
    logger.info("Pipeline: %d test samples", len(y_test))

    # Predictions
    logger.info("Running predictions...")
    predicted = model.predict(X_test_full)
    proba = model.predict_proba(X_test_full)
    ta = float(np.mean(predicted == y_test))
    logger.info("Token accuracy: %.4f (%.1f%%)", ta, ta * 100)

    # Run audit
    results: dict = {"token_accuracy": ta, "n_test": len(y_test), "vocab_size": vs}

    logger.info("Phase 0: Temporal mapping validation...")
    results["phase0"] = validate_temporal_mapping(meta, test_prices)

    logger.info("Q1: Filtered directional accuracy...")
    results["q1"] = q1_filtered_directional(predicted, y_test, meta, vs)

    logger.info("Q2: Near-zero return distribution...")
    results["q2"] = q2_near_zero(meta)

    logger.info("Q3: Level 5 token change frequency...")
    results["q3"] = q3_level5_freq(meta)

    logger.info("Q4: Persistence decomposition...")
    results["q4"] = q4_persistence(predicted, y_test, meta, vs)

    logger.info("Q5: Subsampled accuracy...")
    results["q5"] = q5_subsampled(predicted, y_test)

    logger.info("Q6: Signal backtest...")
    results["q6"] = q6_backtest(predicted, proba, meta, vs)

    logger.info("Q7: Direction-change accuracy...")
    results["q7"] = q7_dir_change(predicted, meta, vs)

    logger.info("Q8: Confusion matrix...")
    results["q8"] = q8_confusion(predicted, y_test, proba, vs)

    logger.info("Q9: Token concentration...")
    results["q9"] = q9_concentration(meta["y_train"])

    logger.info("Q10: Calibration curve...")
    results["q10"] = q10_calibration(predicted, y_test, proba, meta, vs)

    logger.info("Q11: Level isolation...")
    results["q11"] = q11_level_isolation(predicted, proba, meta, vs)

    logger.info("Q12: Random baseline (100 runs)...")
    results["q12"] = q12_random_baseline(predicted, proba, meta, vs)

    # Output
    print_report(results, vs)
    save_json(results)

    elapsed = time.monotonic() - t0
    logger.info("Audit complete in %.1fs", elapsed)


if __name__ == "__main__":
    main()
