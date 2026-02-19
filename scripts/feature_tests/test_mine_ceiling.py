#!/usr/bin/env python3
"""MINE Ceiling Diagnostic: Estimate I(X;Y) and Fano inequality bound.

Uses Mutual Information Neural Estimation (Belghazi et al. 2018) to estimate
the mutual information between D1 input features X and return-direction
class labels Y. Applies the Fano inequality to compute a lower bound on
the Bayes error rate, giving an information-theoretic ceiling on accuracy.

This is a DIAGNOSTIC -- no model changes. The ceiling tells us how much
accuracy headroom remains given the current feature representation.

Reference: Belghazi et al. "Mutual Information Neural Estimation" (ICML 2018)

Usage:
    python -m scripts.feature_tests.test_mine_ceiling
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn as nn
from scripts.feature_tests.exp2_helpers import save_results
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

from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# MINE training hyperparameters
MINE_HIDDEN = 128
MINE_EPOCHS = 200
MINE_BATCH_SIZE = 512
MINE_LR = 1e-4
MINE_EMA_DECAY = 0.01
MAX_SAMPLES = 50000

# CE baseline performance (from harness docs)
CE_ECON_DIR = 0.64


class MINENetwork(nn.Module):
    """Statistics network T(x, y) for MINE estimation."""

    def __init__(self, x_dim: int, y_dim: int, hidden: int = MINE_HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=1))


def train_mine(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    device: torch.device,
) -> float:
    """Train MINE and return MI estimate in nats.

    Uses the Donsker-Varadhan representation with exponential moving average
    for the log-sum-exp term (Belghazi trick) to stabilize training.

    Args:
        x_data: Feature tensor (N, x_dim).
        y_data: One-hot label tensor (N, n_classes).
        device: Torch device.

    Returns:
        MI estimate in nats.
    """
    x_dim = x_data.shape[1]
    y_dim = y_data.shape[1]
    n_samples = x_data.shape[0]

    net = MINENetwork(x_dim, y_dim, MINE_HIDDEN).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=MINE_LR)

    # Running EMA for the denominator (Belghazi stabilization trick)
    running_mean = torch.tensor(1.0, device=device)

    mi_estimates: list[float] = []

    for epoch in range(MINE_EPOCHS):
        # Sample a batch
        idx = torch.randint(0, n_samples, (MINE_BATCH_SIZE,), device=device)
        x_batch = x_data[idx]
        y_batch = y_data[idx]

        # Marginal samples: shuffle y to break the joint distribution
        shuffle_idx = torch.randperm(MINE_BATCH_SIZE, device=device)
        y_shuffled = y_batch[shuffle_idx]

        # Joint: T(x, y)
        t_joint = net(x_batch, y_batch)
        # Marginal: T(x, y_shuffled)
        t_marginal = net(x_batch, y_shuffled)

        # Donsker-Varadhan with EMA stabilization
        mean_joint = t_joint.mean()
        exp_marginal = torch.exp(t_marginal)
        # Update running mean with EMA
        running_mean = (
            (1 - MINE_EMA_DECAY) * running_mean.detach()
            + MINE_EMA_DECAY * exp_marginal.mean().detach()
        )
        # Corrected log-mean-exp using running mean
        log_mean_exp = torch.log(exp_marginal.mean()) - torch.log(
            running_mean
        ) + torch.log(exp_marginal.mean().detach())

        # Maximize MI estimate = E[T(x,y)] - log(E[exp(T(x,y_shuffled))])
        # Equivalently, minimize negative MI
        loss = -(mean_joint - log_mean_exp)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimizer.step()

        mi_est = mean_joint.item() - torch.log(exp_marginal.mean()).item()
        mi_estimates.append(mi_est)

        if (epoch + 1) % 50 == 0:
            avg_mi = float(np.mean(mi_estimates[-20:]))
            logger.info(
                "  MINE epoch %d/%d: MI ~ %.4f nats (%.4f bits)",
                epoch + 1, MINE_EPOCHS, avg_mi, avg_mi / math.log(2),
            )

    # Final MI: average over last 50 epochs for stability
    final_mi_nats = float(np.mean(mi_estimates[-50:]))
    # Clamp to non-negative (MI >= 0 by definition)
    return max(0.0, final_mi_nats)


def main() -> None:
    """Run MINE ceiling diagnostic."""
    logger.info("=== MINE Ceiling Diagnostic ===")

    # --- Load data ---
    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))
    logger.info("Loaded %d tickers", len(tickers))

    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)

    # --- Build D1 pipeline ---
    logger.info("Building D1 pipeline (test set)...")
    X_test, te_rets, te_valid, te_lvls, _, _ = _build_d1_pipeline(
        test_prices, None, tickers, None, 0,
    )

    # Also need train data for quantile boundaries
    X_train, tr_rets, tr_valid, tr_lvls, _, _ = _build_d1_pipeline(
        train_prices, None, tickers, None, 0,
    )

    # --- Compute labels ---
    boundaries = compute_quantile_boundaries(
        tr_rets, tr_lvls, tr_valid, PERCENTILES, per_level=True,
    )
    y_test = assign_quantile_labels(te_rets, te_lvls, boundaries)

    # Filter to valid samples only
    valid_idx = np.where(te_valid)[0]
    X_valid = X_test[valid_idx]
    y_valid = y_test[valid_idx]
    n_total = len(X_valid)
    logger.info("Valid test samples: %d", n_total)

    # Subsample if too large
    if n_total > MAX_SAMPLES:
        rng = np.random.default_rng(42)
        subsample_idx = rng.choice(n_total, MAX_SAMPLES, replace=False)
        X_valid = X_valid[subsample_idx]
        y_valid = y_valid[subsample_idx]
        logger.info("Subsampled to %d samples", MAX_SAMPLES)

    n_samples = len(X_valid)

    # --- Prepare tensors ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Normalize X features to zero mean, unit variance for MINE stability
    x_mean = X_valid.mean(axis=0)
    x_std = X_valid.std(axis=0)
    x_std = np.where(x_std < 1e-10, 1.0, x_std)
    X_norm = (X_valid - x_mean) / x_std

    x_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device)
    x_dim = x_tensor.shape[1]

    # One-hot encode Y
    y_onehot = np.zeros((n_samples, N_CLASSES), dtype=np.float32)
    y_onehot[np.arange(n_samples), y_valid] = 1.0
    y_tensor = torch.tensor(y_onehot, dtype=torch.float32, device=device)

    # --- Train MINE ---
    logger.info("Training MINE (%d epochs, hidden=%d)...", MINE_EPOCHS, MINE_HIDDEN)
    mi_nats = train_mine(x_tensor, y_tensor, device)
    mi_bits = mi_nats / math.log(2)

    # --- Fano inequality ---
    # H(Y|X) >= H(Y) - I(X;Y)
    # Fano: P_error >= 1 - (I(X;Y) + 1) / log2(K)
    log2_k = math.log2(N_CLASSES)
    entropy_y = log2_k  # Upper bound: uniform over K classes

    # Compute actual H(Y) from class distribution
    class_counts = np.bincount(y_valid, minlength=N_CLASSES).astype(np.float64)
    class_probs = class_counts / class_counts.sum()
    # Avoid log(0)
    nonzero = class_probs > 0
    actual_h_y = -np.sum(class_probs[nonzero] * np.log2(class_probs[nonzero]))

    fano_error_bound = max(0.0, 1.0 - (mi_bits + 1.0) / log2_k)
    ceiling_accuracy = 1.0 - fano_error_bound
    observed_error = 1.0 - CE_ECON_DIR
    gap = ceiling_accuracy - CE_ECON_DIR

    # Class distribution
    class_dist_str = ", ".join(
        f"c{i}={class_probs[i]:.1%}" for i in range(N_CLASSES)
    )

    # --- Summary ---
    w = 100
    print()
    print("=" * w)
    print("  MINE CEILING DIAGNOSTIC")
    print("=" * w)
    print(f"  Samples:              {n_samples}")
    print(f"  Feature dim (X):      {x_dim}")
    print(f"  Classes (K):          {N_CLASSES}")
    print(f"  Class distribution:   {class_dist_str}")
    print(f"  H(Y) actual:          {actual_h_y:.4f} bits")
    print(f"  H(Y) uniform:         {entropy_y:.4f} bits")
    print("-" * w)
    print(f"  MI estimate I(X;Y):   {mi_bits:.4f} bits ({mi_nats:.4f} nats)")
    print(f"  Fano error bound:     P_error >= {fano_error_bound:.4f}")
    print(f"  Ceiling accuracy:     <= {ceiling_accuracy:.1%}")
    print("-" * w)
    print(f"  CE baseline accuracy: {CE_ECON_DIR:.1%}")
    print(f"  Observed error rate:  {observed_error:.1%}")
    print(f"  Headroom (gap):       {gap:+.1%}")
    print("-" * w)

    if fano_error_bound > observed_error:
        interpretation = (
            "INFORMATION BOTTLENECK: Fano bound EXCEEDS observed error. "
            "The feature representation does not contain enough information "
            "to significantly improve beyond current accuracy. "
            "New features or a different representation are needed."
        )
    elif gap < 0.05:
        interpretation = (
            "NEAR CEILING: Less than 5pp headroom. "
            "Current accuracy is close to the information-theoretic limit. "
            "Gains require richer input features."
        )
    else:
        interpretation = (
            f"HEADROOM EXISTS: {gap:.1%} gap between current accuracy and ceiling. "
            "Model or training improvements may capture this remaining signal."
        )

    print(f"  Interpretation:       {interpretation}")
    print("=" * w)

    # --- Save results ---
    results = {
        "n_samples": n_samples,
        "x_dim": x_dim,
        "n_classes": N_CLASSES,
        "class_probs": class_probs.tolist(),
        "h_y_actual_bits": float(actual_h_y),
        "h_y_uniform_bits": float(entropy_y),
        "mi_nats": float(mi_nats),
        "mi_bits": float(mi_bits),
        "fano_error_bound": float(fano_error_bound),
        "ceiling_accuracy": float(ceiling_accuracy),
        "ce_baseline_accuracy": CE_ECON_DIR,
        "observed_error": float(observed_error),
        "headroom": float(gap),
        "interpretation": interpretation,
        "mine_config": {
            "hidden": MINE_HIDDEN,
            "epochs": MINE_EPOCHS,
            "batch_size": MINE_BATCH_SIZE,
            "lr": MINE_LR,
            "ema_decay": MINE_EMA_DECAY,
        },
        "model_kwargs": MODEL_KWARGS,
        "context_length": CONTEXT_LENGTH,
        "n_aux_features": N_AUX_FEATURES,
    }
    out_path = save_results("mine_ceiling", results)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
