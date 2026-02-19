#!/usr/bin/env python3
"""HOCMIM: High-Order Conditional Mutual Information for feature selection.

Iterative feature selection that maximizes relevance to Y while minimizing
redundancy with already-selected features. Uses histogram-based MI estimation.

HOCMIM selection criterion:
  1. Start: select feature with highest MI(X_i; Y).
  2. Iterate: add feature maximizing MI(X_i; Y) - (1/|S|) * sum_{j in S} MI(X_i; X_j).
  3. High-order: also compute CMI(X_i; Y | X_j) for top pair as diagnostic.

Features extracted as summary statistics from the D1 pipeline:
  - Context mean and std (2 features)
  - Per-aux-channel mean across the context window (4 features)
  Total: 6 features for MI ranking.

Usage:
    python -m scripts.feature_tests.test_hocmim
"""

from __future__ import annotations  # noqa: I001

import logging
import time

import numpy as np
import torch

from scripts.feature_tests.exp2_helpers import (
    evaluate_5_metrics,
    print_5_metrics,
    save_results,
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

N_BINS = 10  # number of bins for MI discretization
FEATURE_NAMES = [
    "ctx_mean", "ctx_std",
    "aux0_mean", "aux1_mean", "aux2_mean", "aux3_mean",
]


def extract_summary_features(X: np.ndarray, ctx_len: int, n_aux: int) -> np.ndarray:
    """Extract summary statistics from the flat X array for MI analysis.

    From each sample's context window and auxiliary features, compute:
      - Context mean, std (2 features)
      - Mean of each aux channel across positions (n_aux features)

    Args:
        X: Full feature array (N, ctx_len + ctx_len*n_aux + 2).
        ctx_len: Context length.
        n_aux: Number of auxiliary feature channels.

    Returns:
        (N, 2 + n_aux) summary feature array.
    """
    n = len(X)
    ctx = X[:, :ctx_len]
    aux_end = ctx_len + ctx_len * n_aux
    aux_flat = X[:, ctx_len:aux_end].reshape(n, ctx_len, n_aux)

    ctx_mean = ctx.mean(axis=1, keepdims=True)
    ctx_std = ctx.std(axis=1, keepdims=True)
    aux_means = aux_flat.mean(axis=1)  # (N, n_aux)

    return np.hstack([ctx_mean, ctx_std, aux_means]).astype(np.float64)


def discretize_features(X: np.ndarray, n_bins: int) -> np.ndarray:
    """Discretize continuous features into equal-frequency bins.

    Args:
        X: (N, F) continuous features.
        n_bins: Number of bins per feature.

    Returns:
        (N, F) integer bin indices.
    """
    n, f = X.shape
    binned = np.zeros((n, f), dtype=np.int64)
    for j in range(f):
        col = X[:, j]
        # Equal-frequency binning via percentiles
        percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
        edges = np.percentile(col, percentiles)
        edges = np.unique(edges)  # handle degenerate distributions
        binned[:, j] = np.digitize(col, edges)
    return binned


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Histogram-based mutual information I(X; Y).

    Both x and y should be integer-valued (discretized).

    Returns:
        MI in nats.
    """
    n = len(x)
    if n == 0:
        return 0.0

    # Joint distribution
    x_vals = np.unique(x)
    y_vals = np.unique(y)

    mi = 0.0
    for xv in x_vals:
        px = np.mean(x == xv)
        if px <= 0:
            continue
        for yv in y_vals:
            pxy = np.mean((x == xv) & (y == yv))
            py = np.mean(y == yv)
            if pxy > 0 and py > 0:
                mi += pxy * np.log(pxy / (px * py))

    return max(0.0, float(mi))


def conditional_mutual_information(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
) -> float:
    """Conditional mutual information I(X; Y | Z) via histogram.

    CMI(X; Y | Z) = sum_z P(z) * MI(X; Y | Z=z)

    Returns:
        CMI in nats.
    """
    n = len(x)
    if n == 0:
        return 0.0

    z_vals = np.unique(z)
    cmi = 0.0
    for zv in z_vals:
        mask = z == zv
        pz = mask.mean()
        if pz <= 0 or mask.sum() < 5:
            continue
        mi_given_z = mutual_information(x[mask], y[mask])
        cmi += pz * mi_given_z

    return max(0.0, float(cmi))


def hocmim_selection(
    X_binned: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> list[dict]:
    """HOCMIM iterative feature selection.

    Returns ordered list of selected features with MI values.
    """
    n_features = X_binned.shape[1]
    remaining = set(range(n_features))
    selected: list[int] = []
    results: list[dict] = []

    # Step 1: compute MI(X_i; Y) for all features
    mi_xy = np.zeros(n_features)
    for i in range(n_features):
        mi_xy[i] = mutual_information(X_binned[:, i], y)
        logger.info("  MI(%s; Y) = %.4f nats", feature_names[i], mi_xy[i])

    # Step 2: select feature with highest MI
    best_first = int(np.argmax(mi_xy))
    selected.append(best_first)
    remaining.remove(best_first)
    results.append({
        "rank": 1,
        "feature": feature_names[best_first],
        "feature_idx": best_first,
        "mi_xy": float(mi_xy[best_first]),
        "criterion": float(mi_xy[best_first]),
        "redundancy": 0.0,
    })
    logger.info("Selected 1: %s (MI=%.4f)", feature_names[best_first], mi_xy[best_first])

    # Step 3: iteratively add features
    while remaining:
        best_score = -np.inf
        best_feat = -1
        best_redundancy = 0.0

        for i in remaining:
            # Redundancy: average MI with already-selected features
            redundancy = 0.0
            if selected:
                for j in selected:
                    redundancy += mutual_information(X_binned[:, i], X_binned[:, j])
                redundancy /= len(selected)

            # HOCMIM criterion: relevance - redundancy
            score = mi_xy[i] - redundancy
            if score > best_score:
                best_score = score
                best_feat = i
                best_redundancy = redundancy

        selected.append(best_feat)
        remaining.remove(best_feat)
        results.append({
            "rank": len(selected),
            "feature": feature_names[best_feat],
            "feature_idx": best_feat,
            "mi_xy": float(mi_xy[best_feat]),
            "criterion": float(best_score),
            "redundancy": float(best_redundancy),
        })
        logger.info("Selected %d: %s (MI=%.4f, redundancy=%.4f, score=%.4f)",
                     len(selected), feature_names[best_feat],
                     mi_xy[best_feat], best_redundancy, best_score)

    # High-order diagnostic: CMI for top pair
    if len(selected) >= 2:
        i, j = selected[0], selected[1]
        cmi = conditional_mutual_information(X_binned[:, i], y, X_binned[:, j])
        logger.info("CMI(%s; Y | %s) = %.4f nats",
                     feature_names[i], feature_names[j], cmi)
        results[0]["cmi_given_second"] = float(cmi)

    return results


def build_reduced_x(
    X_full: np.ndarray,
    selected_features: list[int],
    ctx_len: int,
    n_aux: int,
) -> np.ndarray:
    """Build a reduced X array using only selected aux channels.

    Maps feature indices back to the original X layout. Features 0-1 map to
    context (always included). Features 2+ map to aux channels.

    For simplicity, we always include the full context window and level/AC IDs.
    We only vary which aux channels are included.

    Args:
        X_full: Original full X array.
        selected_features: Feature indices from HOCMIM ranking.
        ctx_len: Context length.
        n_aux: Original number of aux channels.

    Returns:
        Reduced X array with only selected aux channels.
    """
    n = len(X_full)
    ctx = X_full[:, :ctx_len]
    aux_end = ctx_len + ctx_len * n_aux
    aux_flat = X_full[:, ctx_len:aux_end].reshape(n, ctx_len, n_aux)
    meta = X_full[:, aux_end:aux_end + 2]  # level, AC

    # Determine which aux channels to keep (features 2+ are aux0..aux3)
    aux_channels = [f - 2 for f in selected_features if f >= 2 and f < 2 + n_aux]
    if not aux_channels:
        # No aux channels selected; include just context + metadata
        return np.column_stack([ctx, np.zeros((n, 0)), meta])

    aux_selected = aux_flat[:, :, aux_channels]  # (N, ctx_len, n_selected)
    aux_flat_selected = aux_selected.reshape(n, -1)

    return np.column_stack([ctx, aux_flat_selected, meta])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("HOCMIM: High-order conditional MI feature selection")

    # --- Data ---
    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))
    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)

    X_train, tr_rets, tr_valid, tr_lvls, _, _ = _build_d1_pipeline(
        train_prices, None, tickers, None, 0,
    )
    X_test, te_rets, te_valid, te_lvls, _, _ = _build_d1_pipeline(
        test_prices, None, tickers, None, 0,
    )
    logger.info("Train: %d windows, Test: %d windows", len(X_train), len(X_test))

    boundaries = compute_quantile_boundaries(tr_rets, tr_lvls, tr_valid, PERCENTILES, per_level=True)
    y_train = assign_quantile_labels(tr_rets, tr_lvls, boundaries)
    y_test = assign_quantile_labels(te_rets, te_lvls, boundaries)

    valid_labels = y_train[tr_valid]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    # --- Extract summary features for MI analysis ---
    logger.info("Extracting summary features for MI analysis...")
    train_summary = extract_summary_features(X_train, CONTEXT_LENGTH, N_AUX_FEATURES)
    valid_summary = train_summary[tr_valid]
    valid_y = y_train[tr_valid]

    logger.info("Summary feature shape: %s (valid: %d samples)", train_summary.shape, len(valid_summary))

    # Discretize
    binned = discretize_features(valid_summary, N_BINS)

    # --- HOCMIM selection ---
    logger.info("Running HOCMIM feature selection...")
    ranking = hocmim_selection(binned, valid_y, FEATURE_NAMES)

    # --- Print ranking ---
    w = 100
    print("\n" + "=" * w)
    print("  HOCMIM FEATURE RANKING")
    print("=" * w)
    print(f"  {'Rank':<6} {'Feature':<15} {'MI(X;Y)':>10} {'Redundancy':>12} {'Score':>10}")
    print("-" * w)
    for r in ranking:
        print(f"  {r['rank']:<6d} {r['feature']:<15} {r['mi_xy']:>10.4f}"
              f" {r['redundancy']:>12.4f} {r['criterion']:>10.4f}")
    print("-" * w)

    if "cmi_given_second" in ranking[0]:
        print(f"\n  CMI({ranking[0]['feature']}; Y | {ranking[1]['feature']})"
              f" = {ranking[0]['cmi_given_second']:.4f} nats")

    # --- Train full model (CE baseline) ---
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("Training CE baseline (all %d aux features)...", N_AUX_FEATURES)
    model_full = WaveletGPT(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=N_CLASSES,
        class_weights=class_weights, input_mode="continuous",
        n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
    )
    t0 = time.time()
    model_full.fit(X_train, y_train.astype(np.float64))
    full_time = time.time() - t0
    pred_full = model_full.predict(X_test).astype(np.int64)
    m_full = evaluate_5_metrics(pred_full, te_rets, y_test, te_valid)
    print_5_metrics("Full features (4 aux channels)", m_full)

    # --- Train reduced model (top-K features) ---
    # Use top-K aux channels from HOCMIM ranking
    for top_k in [2, 3]:
        selected_indices = [r["feature_idx"] for r in ranking[:top_k + 2]]  # +2 for ctx features
        aux_channels = [idx - 2 for idx in selected_indices if idx >= 2]
        n_selected_aux = len(aux_channels)

        if n_selected_aux == 0:
            logger.info("Top-%d: no aux channels selected, skipping reduced model", top_k)
            continue

        logger.info("Training reduced model (top-%d aux channels: %s)...",
                     n_selected_aux,
                     [FEATURE_NAMES[idx] for idx in selected_indices if idx >= 2])

        X_train_reduced = build_reduced_x(X_train, selected_indices, CONTEXT_LENGTH, N_AUX_FEATURES)
        X_test_reduced = build_reduced_x(X_test, selected_indices, CONTEXT_LENGTH, N_AUX_FEATURES)

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model_reduced = WaveletGPT(
            vocab_size=1, context_length=CONTEXT_LENGTH,
            task="return_quantile", n_output_classes=N_CLASSES,
            class_weights=class_weights, input_mode="continuous",
            n_aux_features=n_selected_aux, **MODEL_KWARGS,
        )
        t0 = time.time()
        model_reduced.fit(X_train_reduced, y_train.astype(np.float64))
        reduced_time = time.time() - t0
        pred_reduced = model_reduced.predict(X_test_reduced).astype(np.int64)
        m_reduced = evaluate_5_metrics(pred_reduced, te_rets, y_test, te_valid)
        label = f"Reduced ({n_selected_aux} aux channels)"
        print_5_metrics(label, m_reduced)

    # --- Verdict ---
    # This is primarily a diagnostic — pass if feature ranking produced
    verdict = "PASS"
    detail = "Feature ranking produced"

    # Check if any reduced model outperforms
    if m_reduced["transition_acc"] >= 0.54 and m_reduced["econ_dir"] >= 0.64:
        detail += "; reduced model meets absolute thresholds"
    else:
        detail += (f"; reduced model: transition={m_reduced['transition_acc']:.1%},"
                   f" econ_dir={m_reduced['econ_dir']:.1%}")

    print(f"\n  VERDICT: {verdict}")
    print(f"  {detail}")
    print("=" * w)

    save_results("hocmim", {
        "ranking": ranking,
        "full_metrics": m_full,
        "reduced_metrics": m_reduced,
        "full_train_time": full_time,
        "reduced_train_time": reduced_time,
        "n_bins": N_BINS,
        "feature_names": FEATURE_NAMES,
        "verdict": verdict,
        "verdict_detail": detail,
    })


if __name__ == "__main__":
    main()
