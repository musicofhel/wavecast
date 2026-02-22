#!/usr/bin/env python3
"""Focal Loss + Psi-gamma Calibration Correction.

Standard focal loss biases the softmax posterior: the down-weighting of
easy examples shifts the output distribution away from the true class
posteriors. The Psi_gamma inverse function corrects this at inference.

For a focal loss with focusing parameter gamma, the calibration map is:
  p_calibrated_k = p_k^(1+gamma) / sum_j p_j^(1+gamma)
where p_k is the raw softmax probability for class k.

This script compares three variants:
  1. CE baseline (properly calibrated by construction)
  2. Focal loss (gamma=2.0), no correction
  3. Focal loss (gamma=2.0) + psi-gamma inverse correction at inference

Also reports Expected Calibration Error (ECE) for each variant.

Usage:
    python -m scripts.feature_tests.test_focal_psi
"""

from __future__ import annotations  # noqa: I001

import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f_nn
from torch.utils.data import DataLoader, TensorDataset

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
from wavecast.models.wavelet_gpt import WaveletGPT, WaveletGPTNet
from wavecast.targets.returns import assign_quantile_labels, compute_quantile_boundaries

logger = logging.getLogger(__name__)

NET_KWARGS = {k: MODEL_KWARGS[k] for k in ("embed_dim", "num_heads", "num_layers", "dropout")}
GAMMA = 2.0
N_ECE_BINS = 10


class FocalLoss(nn.Module):
    """Standard focal loss (Lin et al. 2017)."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = f_nn.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def psi_gamma_inverse(probs: np.ndarray, gamma: float) -> np.ndarray:
    """Apply psi-gamma inverse calibration correction to softmax probabilities.

    Maps focal-loss-biased posteriors back to true posteriors:
      p_calibrated_k = p_k^(1+gamma) / sum_j p_j^(1+gamma)

    Args:
        probs: (N, C) softmax probabilities from focal-trained model.
        gamma: Focal loss focusing parameter.

    Returns:
        (N, C) calibrated probabilities summing to 1.
    """
    powered = np.power(probs, 1.0 + gamma)
    denom = powered.sum(axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-10)
    return powered / denom


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_ECE_BINS) -> float:
    """Compute Expected Calibration Error.

    Bins predictions by confidence, computes |accuracy - confidence| per bin,
    weighted by bin count.

    Args:
        probs: (N, C) predicted probabilities.
        labels: (N,) true class labels.
        n_bins: Number of confidence bins.

    Returns:
        ECE scalar.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(labels)
    if total == 0:
        return 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        count = mask.sum()
        if count == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (count / total) * abs(bin_acc - bin_conf)

    return float(ece)


def parse_x(
    X: np.ndarray, ctx_len: int = CONTEXT_LENGTH, n_aux: int = N_AUX_FEATURES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse the flat X array into (ctx, lvl, ac, aux) numpy arrays."""
    ctx = X[:, :ctx_len].astype(np.float32)
    aux_end = ctx_len + ctx_len * n_aux
    aux = X[:, ctx_len:aux_end].astype(np.float32).reshape(-1, ctx_len, n_aux)
    lvl = X[:, aux_end].astype(np.int64)
    ac = X[:, aux_end + 1].astype(np.int64)
    return ctx, lvl, ac, aux


def train_focal(
    net: WaveletGPTNet,
    criterion: FocalLoss,
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
) -> float:
    """Train with focal loss. Returns final epoch loss."""
    ctx, lvl, ac, aux = parse_x(X_train)
    y = y_train.astype(np.int64)

    dataset = TensorDataset(
        torch.tensor(ctx, dtype=torch.float32),
        torch.tensor(lvl, dtype=torch.long),
        torch.tensor(ac, dtype=torch.long),
        torch.tensor(aux.reshape(len(aux), -1), dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=MODEL_KWARGS["batch_size"], shuffle=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=MODEL_KWARGS["learning_rate"])

    avg_loss = 0.0
    for _epoch in range(MODEL_KWARGS["epochs"]):
        net.train()
        total, n = 0.0, 0
        for batch in loader:
            ctx_b = batch[0].to(device)
            lvl_b = batch[1].to(device)
            ac_b = batch[2].to(device)
            aux_b = batch[3].to(device).reshape(-1, CONTEXT_LENGTH, N_AUX_FEATURES)
            tgt = batch[4].to(device)

            optimizer.zero_grad()
            logits_dict = net(ctx_b, lvl_b, ac_b, aux_features=aux_b)
            loss = criterion(logits_dict[1], tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            n += 1
        avg_loss = total / max(n, 1)
    return avg_loss


def predict_probs_from_net(
    net: WaveletGPTNet, X: np.ndarray, device: torch.device,
) -> np.ndarray:
    """Get softmax probabilities from a raw WaveletGPTNet. Returns (N, C)."""
    ctx, lvl, ac, aux = parse_x(X)
    net.eval()
    with torch.no_grad():
        logits_dict = net(
            torch.tensor(ctx, dtype=torch.float32).to(device),
            torch.tensor(lvl, dtype=torch.long).to(device),
            torch.tensor(ac, dtype=torch.long).to(device),
            aux_features=torch.tensor(aux, dtype=torch.float32).to(device),
        )
    probs = f_nn.softmax(logits_dict[1], dim=-1).cpu().numpy()
    return probs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Focal loss + psi-gamma calibration correction experiment")

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
    cw_tensor = torch.tensor(class_weights, dtype=torch.float32)

    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- CE Baseline ---
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Training CE baseline...")
    model_ce = WaveletGPT(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=N_CLASSES,
        class_weights=class_weights, input_mode="continuous",
        n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
    )
    t0 = time.time()
    model_ce.fit(X_train, y_train.astype(np.float64))
    ce_time = time.time() - t0

    # CE probabilities via predict_proba
    pred_ce = model_ce.predict(X_test).astype(np.int64)
    ce_probs = model_ce.predict_proba(X_test)
    m_ce = evaluate_5_metrics(pred_ce, te_rets, y_test, te_valid)
    ece_ce = compute_ece(ce_probs[te_valid], y_test[te_valid])
    print_5_metrics("CE Baseline", m_ce, label=f"ECE={ece_ce:.4f}")

    # --- Focal (no correction) ---
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Training focal loss (gamma=%.1f, no correction)...", GAMMA)

    net_focal = WaveletGPTNet(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=N_CLASSES,
        input_mode="continuous", n_aux_features=N_AUX_FEATURES,
        **NET_KWARGS,
    ).to(device)

    focal_criterion = FocalLoss(gamma=GAMMA, weight=cw_tensor.to(device))
    t0 = time.time()
    focal_loss = train_focal(net_focal, focal_criterion, X_train, y_train, device)
    focal_time = time.time() - t0

    focal_probs = predict_probs_from_net(net_focal, X_test, device)
    pred_focal = focal_probs.argmax(axis=1).astype(np.int64)
    m_focal = evaluate_5_metrics(pred_focal, te_rets, y_test, te_valid)
    ece_focal = compute_ece(focal_probs[te_valid], y_test[te_valid])
    print_5_metrics("Focal (no correction)", m_focal, label=f"ECE={ece_focal:.4f}")

    # --- Focal + psi correction ---
    logger.info("Applying psi-gamma inverse correction (gamma=%.1f)...", GAMMA)
    corrected_probs = psi_gamma_inverse(focal_probs, GAMMA)
    pred_corrected = corrected_probs.argmax(axis=1).astype(np.int64)
    m_corrected = evaluate_5_metrics(pred_corrected, te_rets, y_test, te_valid)
    ece_corrected = compute_ece(corrected_probs[te_valid], y_test[te_valid])
    print_5_metrics("Focal + Psi Correction", m_corrected, label=f"ECE={ece_corrected:.4f}")

    # --- Comparison ---
    verdict = verdict_from_metrics(m_ce, m_corrected)

    w = 100
    print("\n" + "=" * w)
    print("Focal + Psi-gamma Calibration Results")
    print("=" * w)
    print(f"  {'Variant':<25} {'Econ Dir':>9} {'Trans':>8} {'Large':>8}"
          f" {'Sharpe':>8} {'ECE':>8}")
    print("-" * w)
    print(f"  {'CE Baseline':<25} {m_ce['econ_dir']:>8.1%} {m_ce['transition_acc']:>7.1%}"
          f" {m_ce['large_move_acc']:>7.1%} {m_ce['sharpe_costs']:>+7.3f} {ece_ce:>7.4f}")
    print(f"  {'Focal (no correction)':<25} {m_focal['econ_dir']:>8.1%}"
          f" {m_focal['transition_acc']:>7.1%} {m_focal['large_move_acc']:>7.1%}"
          f" {m_focal['sharpe_costs']:>+7.3f} {ece_focal:>7.4f}")
    print(f"  {'Focal + Psi correction':<25} {m_corrected['econ_dir']:>8.1%}"
          f" {m_corrected['transition_acc']:>7.1%} {m_corrected['large_move_acc']:>7.1%}"
          f" {m_corrected['sharpe_costs']:>+7.3f} {ece_corrected:>7.4f}")
    print("-" * w)

    # Calibration improvement
    ece_improvement = ece_focal - ece_corrected
    print(f"\n  Psi correction ECE improvement: {ece_improvement:+.4f}"
          f" ({ece_focal:.4f} -> {ece_corrected:.4f})")
    if ece_corrected < 0.05:
        print("  ECE < 0.05: well-calibrated after correction")
    else:
        print(f"  ECE = {ece_corrected:.4f}: calibration still needs work")

    print(f"\n  VERDICT: {verdict}")
    print("=" * w)

    save_results("focal_psi", {
        "ce_baseline": m_ce, "ce_ece": ece_ce, "ce_train_time": ce_time,
        "focal_uncorrected": m_focal, "focal_ece": ece_focal,
        "focal_corrected": m_corrected, "corrected_ece": ece_corrected,
        "focal_train_loss": focal_loss, "focal_train_time": focal_time,
        "gamma": GAMMA, "ece_improvement": ece_improvement,
        "verdict": verdict,
    })


if __name__ == "__main__":
    main()
