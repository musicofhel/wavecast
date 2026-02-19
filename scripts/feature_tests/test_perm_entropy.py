#!/usr/bin/env python3
"""Feature Test: Permutation Entropy and Complexity.

Hypothesis: Ordinal pattern statistics on D1 coefficient deltas capture
the degree of structural regularity vs randomness in the wavelet domain.
Low permutation entropy = highly predictable regime (trend or mean-reversion).
High permutation entropy = noisy/random regime.

2 new features per timestep:
  - perm_entropy: Normalized Shannon entropy of order-m ordinal patterns
    in a trailing window. Range [0, 1]. 0 = perfectly ordered, 1 = maximally
    disordered (uniform pattern distribution).
  - perm_complexity: Jensen-Shannon statistical complexity.
    C = H_norm * D_JS(pattern_dist, uniform). Peaks at intermediate disorder
    where structure is present but not trivial.

Ordinal patterns of order m=3 (3! = 6 possible patterns) with sliding
window of W=16 positions.

Reference: Bandt & Pompe (2002) "Permutation entropy: A natural complexity
measure for time series", PRL 88.

Usage:
    python -m scripts.feature_tests.test_perm_entropy
"""

from __future__ import annotations  # noqa: I001

import logging
import math
import time

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from scripts.feature_tests.exp2_helpers import (
    evaluate_5_metrics,
    print_5_metrics,
    save_results,
    verdict_from_metrics,
)
from scripts.feature_tests.harness import (
    CONTEXT_LENGTH,
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
from wavecast.targets.returns import assign_quantile_labels, compute_quantile_boundaries

logger = logging.getLogger(__name__)

N_NEW_FEATURES = 2
ORDER = 3  # Ordinal pattern order (m=3 -> 3!=6 patterns)
WINDOW_SIZE = 16  # Trailing window for pattern extraction
N_PATTERNS = math.factorial(ORDER)  # 6


def _ordinal_pattern(segment: NDArray) -> int:
    """Map a segment of length m to its ordinal pattern index.

    The ordinal pattern is the rank permutation of the segment values.
    We encode the permutation as a mixed-radix number for a unique integer key.

    Args:
        segment: Array of length m (ORDER).

    Returns:
        Integer in [0, m!) representing the ordinal pattern.
    """
    m = len(segment)
    ranks = np.argsort(np.argsort(segment))
    # Encode permutation as a mixed-radix number (Lehmer code)
    code = 0
    for i in range(m):
        # Count how many elements after position i have a smaller rank
        count = 0
        for j in range(i + 1, m):
            if ranks[j] < ranks[i]:
                count += 1
        code = code * (m - i) + count
    return code


def _shannon_entropy(counts: NDArray, total: int) -> float:
    """Compute Shannon entropy in nats from pattern counts.

    Args:
        counts: Array of pattern occurrence counts.
        total: Total number of patterns observed.

    Returns:
        Shannon entropy H in nats.
    """
    if total == 0:
        return 0.0
    probs = counts / total
    nonzero = probs > 0
    return -float(np.sum(probs[nonzero] * np.log(probs[nonzero])))


def _js_divergence(p: NDArray, q: NDArray) -> float:
    """Jensen-Shannon divergence between distributions p and q.

    D_JS = 0.5 * KL(p || m) + 0.5 * KL(q || m), where m = 0.5*(p+q).

    Args:
        p: First probability distribution.
        q: Second probability distribution.

    Returns:
        JS divergence (non-negative, in nats).
    """
    m = 0.5 * (p + q)
    # Avoid log(0): only compute where both p and m are positive
    d = 0.0
    for i in range(len(p)):
        if p[i] > 0 and m[i] > 0:
            d += 0.5 * p[i] * np.log(p[i] / m[i])
        if q[i] > 0 and m[i] > 0:
            d += 0.5 * q[i] * np.log(q[i] / m[i])
    return max(0.0, d)


def _compute_perm_entropy_complexity(
    data: NDArray,
    order: int = ORDER,
    window: int = WINDOW_SIZE,
) -> tuple[NDArray, NDArray]:
    """Compute permutation entropy and complexity for a 1-D sequence.

    For each position t >= window-1, extracts all order-m ordinal patterns
    from data[t-window+1:t+1], computes normalized Shannon entropy and
    Jensen-Shannon statistical complexity.

    Args:
        data: 1-D array of observations.
        order: Ordinal pattern order (default 3).
        window: Trailing window size (default 16).

    Returns:
        perm_entropy: (len(data),) normalized entropy in [0, 1].
        perm_complexity: (len(data),) JS complexity.
    """
    n = len(data)
    n_pats = math.factorial(order)
    max_entropy = np.log(n_pats)  # log(m!) in nats
    uniform = np.full(n_pats, 1.0 / n_pats, dtype=np.float64)

    perm_ent = np.zeros(n, dtype=np.float64)
    perm_cplx = np.zeros(n, dtype=np.float64)

    # Number of ordinal patterns extractable from a window of size W
    # with order m is W - m + 1.
    n_patterns_per_window = window - order + 1

    for t in range(n):
        if t < window - 1:
            # Not enough data for a full window; will be forward-filled below
            continue

        win = data[t - window + 1: t + 1]

        # Extract all order-m patterns from the window
        counts = np.zeros(n_pats, dtype=np.float64)
        for i in range(n_patterns_per_window):
            seg = win[i: i + order]
            pat_idx = _ordinal_pattern(seg)
            counts[pat_idx] += 1

        total = int(counts.sum())
        h = _shannon_entropy(counts, total)

        # Normalized entropy
        h_norm = h / max_entropy if max_entropy > 0 else 0.0
        perm_ent[t] = h_norm

        # Statistical complexity: C = H_norm * D_JS(observed, uniform)
        if total > 0:
            observed = counts / total
            d_js = _js_divergence(observed, uniform)
            # Normalize D_JS by its maximum (log(2) for binary case,
            # but we use raw D_JS scaled by H_norm)
            perm_cplx[t] = h_norm * d_js
        else:
            perm_cplx[t] = 0.0

    # Forward-fill initial positions with the first valid value
    if window - 1 < n:
        first_valid_ent = perm_ent[window - 1]
        first_valid_cplx = perm_cplx[window - 1]
        perm_ent[:window - 1] = first_valid_ent
        perm_cplx[:window - 1] = first_valid_cplx

    return perm_ent, perm_cplx


def perm_entropy_features(
    ohlcv_df: pd.DataFrame,
    detail_coeffs: NDArray,
    approx_coeffs: NDArray,
    level: int,
) -> NDArray:
    """Permutation entropy and complexity on D1 coefficient deltas.

    Returns (n_deltas, 2): [perm_entropy, perm_complexity].
    """
    n_deltas = len(detail_coeffs) - 1
    if n_deltas <= 0:
        return np.empty((0, N_NEW_FEATURES), dtype=np.float64)

    deltas = np.diff(detail_coeffs)
    perm_ent, perm_cplx = _compute_perm_entropy_complexity(deltas)

    return np.column_stack([perm_ent, perm_cplx])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Permutation entropy feature test (2 features: perm_entropy, perm_complexity)")

    # --- Data ---
    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))
    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)
    logger.info("Loaded %d tickers", len(tickers))

    # --- Build pipelines ---
    logger.info("Building baseline dataset (4 aux features)...")
    X_train_base, tr_rets, tr_valid, tr_lvls, _, _ = _build_d1_pipeline(
        train_prices, None, tickers, None, 0,
    )
    X_test_base, te_rets, te_valid, te_lvls, _, _ = _build_d1_pipeline(
        test_prices, None, tickers, None, 0,
    )

    logger.info("Building perm-entropy challenger dataset (4 + 2 aux features)...")
    X_train_chal, _, _, _, _, _ = _build_d1_pipeline(
        train_prices, train_ohlcv, tickers, perm_entropy_features, N_NEW_FEATURES,
    )
    X_test_chal, _, _, _, _, _ = _build_d1_pipeline(
        test_prices, test_ohlcv, tickers, perm_entropy_features, N_NEW_FEATURES,
    )

    logger.info("Train: %d windows, Test: %d windows", len(X_train_base), len(X_test_base))

    # --- Labels and class weights ---
    boundaries = compute_quantile_boundaries(tr_rets, tr_lvls, tr_valid, PERCENTILES, per_level=True)
    y_train = assign_quantile_labels(tr_rets, tr_lvls, boundaries)
    y_test = assign_quantile_labels(te_rets, te_lvls, boundaries)

    valid_labels = y_train[tr_valid]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Baseline ---
    logger.info("Training CE baseline (4 aux)...")
    model_base = WaveletGPT(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=N_CLASSES,
        class_weights=class_weights, input_mode="continuous",
        n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
    )
    t0 = time.time()
    model_base.fit(X_train_base, y_train.astype(np.float64))
    base_time = time.time() - t0
    pred_base = model_base.predict(X_test_base).astype(np.int64)
    m_base = evaluate_5_metrics(pred_base, te_rets, y_test, te_valid)
    print_5_metrics("CE Baseline (4 aux)", m_base)

    # --- Perm-Entropy Challenger ---
    logger.info("Training perm-entropy challenger (4 + 2 aux)...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_chal = WaveletGPT(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=N_CLASSES,
        class_weights=class_weights, input_mode="continuous",
        n_aux_features=N_AUX_FEATURES + N_NEW_FEATURES, **MODEL_KWARGS,
    )
    t0 = time.time()
    model_chal.fit(X_train_chal, y_train.astype(np.float64))
    chal_time = time.time() - t0
    pred_chal = model_chal.predict(X_test_chal).astype(np.int64)
    m_chal = evaluate_5_metrics(pred_chal, te_rets, y_test, te_valid)
    print_5_metrics("Perm-Entropy Challenger (6 aux)", m_chal)

    # --- Comparison ---
    verdict = verdict_from_metrics(m_base, m_chal)

    w = 100
    print("\n" + "=" * w)
    print("  PERMUTATION ENTROPY FEATURE TEST COMPARISON")
    print("=" * w)
    print(f"  {'Metric':<22} {'Baseline':>10} {'Challenger':>12} {'Delta':>10}")
    print("  " + "-" * (w - 4))
    for key, label in [
        ("econ_dir", "Econ Dir Acc"),
        ("transition_acc", "Transition Acc"),
        ("large_move_acc", "Large-Move Acc"),
        ("sharpe_costs", "Sharpe (+costs)"),
        ("quantile_acc", "Quantile Acc"),
    ]:
        bv = m_base[key]
        cv = m_chal[key]
        delta = cv - bv
        if key == "sharpe_costs":
            print(f"  {label:<22} {bv:>+9.3f} {cv:>+11.3f} {delta:>+9.3f}")
        else:
            print(f"  {label:<22} {bv:>9.1%} {cv:>11.1%} {delta:>+9.1%}")
    print("  " + "-" * (w - 4))
    print(f"  Pred Dist (base):  up={m_base['pred_dist']['up']:.1%}"
          f"  flat={m_base['pred_dist']['flat']:.1%}"
          f"  down={m_base['pred_dist']['down']:.1%}")
    print(f"  Pred Dist (chal):  up={m_chal['pred_dist']['up']:.1%}"
          f"  flat={m_chal['pred_dist']['flat']:.1%}"
          f"  down={m_chal['pred_dist']['down']:.1%}")
    print(f"\n  Baseline train time:    {base_time:.1f}s")
    print(f"  Challenger train time:  {chal_time:.1f}s")
    print(f"\n  VERDICT: {verdict}")
    print("=" * w)

    save_results("perm_entropy", {
        "baseline": m_base,
        "challenger": m_chal,
        "verdict": verdict,
        "baseline_train_time": base_time,
        "challenger_train_time": chal_time,
        "perm_entropy_config": {
            "order": ORDER,
            "window_size": WINDOW_SIZE,
            "n_patterns": N_PATTERNS,
        },
    })


if __name__ == "__main__":
    main()
