#!/usr/bin/env python3
"""Quick audit of a trained return-quantile model.

Usage:
    python scripts/return_audit.py [MODEL_DIR]

Default MODEL_DIR: ~/.wavecast/models/return_quantile_v1/

Evaluates: economic directional accuracy, quantile accuracy, Sharpe,
confidence calibration, comparison to random baseline.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from wavecast.core.types import TimeSeries
from wavecast.core.universe import DEFAULT_UNIVERSE
from wavecast.data.cache import ParquetCache
from wavecast.evaluation.return_eval import evaluate_return_predictions
from wavecast.experiments.metrics import compute_return_baselines
from wavecast.experiments.runner import SECTOR_ID_MAP
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.sax.bow import extract_words
from wavecast.sax.sax import sax_transform
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_sample_returns,
)
from wavecast.tokenizer.dataset import build_return_target_dataset
from wavecast.tokenizer.vocabulary import SAXVocabulary
from wavecast.wavelets.dwt import decompose

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TICKERS = [a.ticker for a in DEFAULT_UNIVERSE.assets]
INTERVAL = "1h"
TEST_START = "2025-01-01"
DWT_LEVELS = [1, 2, 5]
CONTEXT_LENGTH = 16
N_CLASSES = 5


def main() -> None:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path.home() / ".wavecast" / "models" / "return_quantile_v1"
    )

    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        print("Run scripts/train_return_model.py first.")
        sys.exit(1)

    # Load model, vocab, boundaries
    model = WaveletGPT.load(model_dir)
    vocab = SAXVocabulary.load(model_dir / "vocabulary.json")
    with open(model_dir / "boundaries.json") as f:
        raw_bounds = json.load(f)
    boundaries = {int(k): np.array(v) for k, v in raw_bounds.items()}

    logger.info("Loaded model (task=%s), vocab (size=%d)", model.task, vocab.size)

    # Load test data
    cache = ParquetCache(Path.home() / ".wavecast" / "cache")

    test_prices: dict[str, TimeSeries] = {}
    for ticker in TICKERS:
        ts = cache.get(ticker, INTERVAL)
        if ts is None:
            continue
        mask = ts.timestamps >= np.datetime64(TEST_START)
        if mask.sum() > 50:
            test_prices[ticker] = TimeSeries(
                values=ts.values[mask], timestamps=ts.timestamps[mask],
                ticker=ticker, interval=INTERVAL,
            )

    tickers = sorted(test_prices.keys())
    logger.info("Test tickers: %d", len(tickers))

    # DWT → SAX → tokenize
    test_token_seqs_raw = []
    test_coeffs_map: dict[tuple[str, int], int] = {}
    test_symbols_map: dict[tuple[str, int], int] = {}

    for ticker in tickers:
        decomp = decompose(test_prices[ticker], level=5)
        level_words: dict[int, list[str]] = {}
        for lvl in DWT_LEVELS:
            coeffs = decomp.detail_at_level(lvl)
            if len(coeffs) >= 2:
                ns = min(512, len(coeffs))
                sr = sax_transform(coeffs, ns, 7)
                words = extract_words(sr.symbols, 4, 1)
                level_words[lvl] = words
                test_coeffs_map[(ticker, lvl)] = len(coeffs)
                test_symbols_map[(ticker, lvl)] = len(sr.symbols)
        test_token_seqs_raw.append((ticker, level_words))

    from wavecast.core.types import MultiLevelTokenSequence, TokenSequence

    test_mlts = []
    for ticker, lw in test_token_seqs_raw:
        ls = {}
        for lvl, words in lw.items():
            ids = vocab.encode_sequence(words)
            ls[lvl] = TokenSequence(
                token_ids=ids, words=words, ticker=ticker,
                interval=INTERVAL, wavelet_level=lvl,
            )
        test_mlts.append(MultiLevelTokenSequence(
            ticker=ticker, interval=INTERVAL, level_sequences=ls,
        ))

    asset_class_map = {}
    for a in DEFAULT_UNIVERSE.assets:
        if a.sector:
            asset_class_map[a.ticker] = SECTOR_ID_MAP.get(a.sector.value, 0)

    test_ds = build_return_target_dataset(
        test_mlts, vocab, CONTEXT_LENGTH, asset_class_map,
        n_coeffs_map=test_coeffs_map, n_symbols_map=test_symbols_map,
    )

    # Compute returns
    tp = np.array([s.token_position for s in test_ds.samples], dtype=np.int64)
    lvls = np.array([s.level for s in test_ds.samples], dtype=np.int64)
    nc = np.array([s.n_coeffs for s in test_ds.samples], dtype=np.int64)
    ns = np.array([s.n_symbols for s in test_ds.samples], dtype=np.int64)

    sample_tickers: list[str] = []
    for mlt in test_mlts:
        for _, seq in mlt.level_sequences.items():
            n_s = max(0, len(seq.token_ids) - CONTEXT_LENGTH)
            sample_tickers.extend([mlt.ticker] * n_s)

    test_rets = np.full(len(test_ds.samples), np.nan)
    test_valid = np.zeros(len(test_ds.samples), dtype=np.bool_)

    if len(sample_tickers) == len(test_ds.samples):
        tarr = np.array(sample_tickers)
        for t in set(sample_tickers):
            m = tarr == t
            if t not in test_prices:
                continue
            r, v = compute_sample_returns(tp[m], lvls[m], test_prices[t].values, nc[m], ns[m])
            test_rets[m] = r
            test_valid[m] = v

    y_test_labels = assign_quantile_labels(test_rets, lvls, boundaries)

    # Predict
    X_test, _, _, _ = test_ds.to_arrays()
    test_lvl_arr = np.array([s.level for s in test_ds.samples], dtype=np.int64)
    test_ac_arr = np.array([s.asset_class_id for s in test_ds.samples], dtype=np.int64)
    X_test_full = np.column_stack([X_test, test_lvl_arr, test_ac_arr])

    predicted = model.predict(X_test_full)
    proba = model.predict_proba(X_test_full)
    pred_labels = predicted.astype(np.int64)

    # Evaluate
    rt_metrics = evaluate_return_predictions(
        pred_labels, test_rets, y_test_labels, n_classes=N_CLASSES, probs=proba,
    )

    # Baselines
    baselines = compute_return_baselines(
        y_test_labels[test_valid], y_test_labels[test_valid], N_CLASSES
    )

    # Sharpe ratio from economic positions
    valid_mask = test_valid
    mid = N_CLASSES // 2
    pred_dir = np.zeros(len(pred_labels), dtype=np.float64)
    pred_dir[pred_labels > mid] = 1.0
    pred_dir[pred_labels < mid] = -1.0

    pnl = pred_dir[valid_mask] * test_rets[valid_mask]
    pnl = pnl[~np.isnan(pnl)]
    sharpe = float(np.mean(pnl) / np.std(pnl) * np.sqrt(252 * 7)) if len(pnl) > 1 else 0.0

    # Confidence calibration: split by softmax confidence deciles
    max_probs = np.max(proba, axis=1)
    decile_results = []
    for d in range(10):
        lo = np.percentile(max_probs, d * 10)
        hi = np.percentile(max_probs, (d + 1) * 10)
        mask = (max_probs >= lo) & (max_probs < hi) & valid_mask
        if d == 9:
            mask = (max_probs >= lo) & valid_mask
        if mask.sum() > 0:
            q_acc = float(np.mean(pred_labels[mask] == y_test_labels[mask]))
            d_acc_samples = pred_dir[mask] * np.sign(test_rets[mask])
            has_dir = test_rets[mask] != 0
            d_acc = float(np.mean(d_acc_samples[has_dir] > 0)) if has_dir.any() else 0.5
            decile_results.append((d, lo, hi, int(mask.sum()), q_acc, d_acc))

    # Random baseline comparison (Monte Carlo)
    rng = np.random.default_rng(42)
    random_sharpes = []
    for _ in range(1000):
        rand_dir = rng.choice([-1.0, 0.0, 1.0], size=valid_mask.sum())
        rand_pnl = rand_dir * test_rets[valid_mask]
        rand_pnl = rand_pnl[~np.isnan(rand_pnl)]
        if len(rand_pnl) > 1 and np.std(rand_pnl) > 0:
            random_sharpes.append(np.mean(rand_pnl) / np.std(rand_pnl) * np.sqrt(252 * 7))
    random_sharpes = sorted(random_sharpes)
    model_percentile = float(np.mean(np.array(random_sharpes) < sharpe) * 100)

    # Per-level results
    per_level_results = {}
    for lvl in sorted(set(int(v) for v in lvls)):
        mask = (lvls == lvl) & valid_mask
        if mask.sum() > 0:
            q_acc = float(np.mean(pred_labels[mask] == y_test_labels[mask]))
            has_dir = test_rets[mask] != 0
            d_acc = float(np.mean((pred_dir[mask] == np.sign(test_rets[mask]))[has_dir])) if has_dir.any() else 0.5
            per_level_results[lvl] = {"quantile_acc": q_acc, "dir_acc": d_acc, "n_samples": int(mask.sum())}

    # Print report
    print("\n" + "=" * 70)
    print("RETURN-QUANTILE MODEL AUDIT")
    print("=" * 70)
    print(f"\nModel: {model_dir}")
    print(f"Test period: {TEST_START}+")
    print(f"Tickers: {len(tickers)}")
    print(f"Test samples: {len(test_ds.samples)} ({valid_mask.sum()} valid)")

    print("\n--- PRIMARY METRICS ---")
    print(f"Quantile accuracy:     {rt_metrics.quantile_accuracy:.4f}")
    print(f"Directional accuracy:  {rt_metrics.directional_accuracy:.4f}")
    print(f"Strong signal acc:     {rt_metrics.strong_signal_accuracy:.4f}")
    print(f"Economic value:        {rt_metrics.economic_value:+.6f}")
    print(f"Annualized Sharpe:     {sharpe:+.4f}")

    print("\n--- BASELINES ---")
    print(f"Random (1/{N_CLASSES}):            {baselines['random']:.4f}")
    print(f"Always flat:           {baselines['always_flat']:.4f}")
    print(f"Always up:             {baselines['always_up']:.4f}")
    print(f"Level majority:        {baselines['level_majority']:.4f}")

    print("\n--- MEAN RETURN PER PREDICTED CLASS ---")
    for c, r in sorted(rt_metrics.mean_return_per_class.items()):
        label = ["strong_down", "down", "flat", "up", "strong_up"][c]
        print(f"  {label:12s} (class {c}): {r:+.6f}")

    print("\n--- PER-LEVEL RESULTS ---")
    for lvl, res in sorted(per_level_results.items()):
        print(f"  Level {lvl}: quantile_acc={res['quantile_acc']:.4f}  dir_acc={res['dir_acc']:.4f}  n={res['n_samples']}")

    print("\n--- CONFIDENCE CALIBRATION ---")
    print(f"  {'Decile':>7}  {'Range':>15}  {'N':>6}  {'Q.Acc':>6}  {'D.Acc':>6}")
    for d, lo, hi, n, qa, da in decile_results:
        print(f"  D{d:>5d}  [{lo:.3f}, {hi:.3f})  {n:>6d}  {qa:.4f}  {da:.4f}")

    print("\n--- RANDOM BASELINE COMPARISON ---")
    print(f"Model Sharpe: {sharpe:+.4f}")
    print(f"Random Sharpe P5/P50/P95: {np.percentile(random_sharpes, 5):+.4f} / {np.percentile(random_sharpes, 50):+.4f} / {np.percentile(random_sharpes, 95):+.4f}")
    print(f"Model percentile: {model_percentile:.1f}%")

    print("\n" + "=" * 70)
    if rt_metrics.directional_accuracy > 0.55:
        print("VERDICT: STRONG EDGE — directional accuracy > 55%")
    elif rt_metrics.directional_accuracy > 0.50:
        print("VERDICT: MARGINAL EDGE — directional accuracy > 50%, needs more validation")
    else:
        print("VERDICT: NO EDGE — SAX features appear uninformative for returns")
        print("NEXT: Try Stage B (regression target) or bypass SAX entirely")
    print("=" * 70)

    # Save results
    results_dir = Path.home() / ".wavecast" / "audit"
    results_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "quantile_accuracy": rt_metrics.quantile_accuracy,
        "directional_accuracy": rt_metrics.directional_accuracy,
        "strong_signal_accuracy": rt_metrics.strong_signal_accuracy,
        "economic_value": rt_metrics.economic_value,
        "sharpe": sharpe,
        "model_percentile": model_percentile,
        "baselines": baselines,
        "per_level": per_level_results,
        "mean_return_per_class": rt_metrics.mean_return_per_class,
    }
    with open(results_dir / "return_audit_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", results_dir / "return_audit_results.json")


if __name__ == "__main__":
    main()
