#!/usr/bin/env python3
"""BOLT Ceiling Diagnostic: Smooth 0-1 loss ceiling estimation.

Trains WaveletGPT with a smooth surrogate of the 0-1 loss (BOLT-style)
to directly optimize classification accuracy. The converged accuracy
at progressively sharper sigma values approximates the Bayes-optimal
accuracy -- the theoretical ceiling for this feature representation.

Approach:
  1. Warm-start with cross-entropy for 30 epochs (standard training).
  2. Switch to SmoothZeroOneLoss at decreasing sigma (1.0, 0.5, 0.1).
  3. Report converged accuracy at each sigma as a ceiling estimate.

This is a DIAGNOSTIC -- no model changes.

Usage:
    python -m scripts.feature_tests.test_bolt_ceiling
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.nn as nn
from scripts.feature_tests.exp2_helpers import MID, save_results
from scripts.feature_tests.harness import (
    CONTEXT_LENGTH,
    MIN_RETURN_THRESHOLD,
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
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Training config
CE_WARMUP_EPOCHS = 30
BOLT_EPOCHS = 20
SIGMA_VALUES = [1.0, 0.5, 0.1]
BOLT_LR = 1e-4
BOLT_BATCH_SIZE = 64

# CE baseline performance (from harness docs)
CE_ECON_DIR = 0.64


class SmoothZeroOneLoss(nn.Module):
    """Smooth surrogate for the 0-1 classification loss.

    Approximates the indicator 1[y_pred != y_true] using a sigmoid on
    the margin between the correct-class logit and the largest wrong-class
    logit. As sigma -> 0, this converges to the true 0-1 loss.

    Args:
        sigma: Temperature parameter controlling sharpness.
            sigma=1.0 is a smooth approximation.
            sigma->0 approaches the hard 0-1 loss.
    """

    def __init__(self, sigma: float = 1.0) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute smooth 0-1 loss.

        Args:
            logits: (batch, n_classes) raw model outputs.
            targets: (batch,) integer class labels.

        Returns:
            Scalar loss value.
        """
        # Extract logit for the correct class
        correct_logit = logits.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)  # (batch,)

        # Mask correct class and find max wrong-class logit
        max_wrong = logits.clone()
        max_wrong.scatter_(1, targets.unsqueeze(1), -1e9)
        max_wrong_logit = max_wrong.max(dim=1).values  # (batch,)

        # Margin: positive when correct class has highest logit
        margin = correct_logit - max_wrong_logit

        # Smooth 0-1 loss: 1 - sigmoid(margin / sigma)
        # When margin >> 0 (correct prediction), loss -> 0
        # When margin << 0 (wrong prediction), loss -> 1
        return (1.0 - torch.sigmoid(margin / self.sigma)).mean()


def compute_econ_dir_accuracy(
    pred_labels: np.ndarray,
    actual_returns: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    """Compute economic directional accuracy."""
    pred_dir = np.zeros(len(pred_labels), dtype=np.float64)
    pred_dir[pred_labels > MID] = 1.0
    pred_dir[pred_labels < MID] = -1.0
    actual_dir = np.sign(actual_returns)

    filt = valid_mask & (np.abs(actual_returns) > MIN_RETURN_THRESHOLD)
    if filt.sum() == 0:
        return 0.5
    has_pred = pred_dir[filt] != 0
    if has_pred.sum() == 0:
        return 0.5
    return float(np.mean(pred_dir[filt][has_pred] == actual_dir[filt][has_pred]))


def bolt_finetune(
    model: WaveletGPT,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sigma: float,
) -> None:
    """Fine-tune a pre-trained WaveletGPT with SmoothZeroOneLoss.

    Accesses model internals (_net, _device, _parse_x) to run a custom
    training loop with the BOLT loss while reusing the model's architecture.

    Args:
        model: Pre-trained WaveletGPT with _net already initialized.
        X_train: Training features.
        y_train: Training labels.
        sigma: SmoothZeroOneLoss temperature.
    """
    net = model._net
    device = model._device
    criterion = SmoothZeroOneLoss(sigma=sigma)

    ctx, lvl, ac, aux = model._parse_x(X_train)
    ctx_tensor = torch.tensor(ctx, dtype=torch.float32).to(device)
    lvl_tensor = torch.tensor(lvl, dtype=torch.long).to(device)
    ac_tensor = torch.tensor(ac, dtype=torch.long).to(device)
    aux_tensor = None
    if aux is not None:
        aux_tensor = torch.tensor(aux, dtype=torch.float32).to(device)

    y_tensor = torch.tensor(y_train.astype(np.int64), dtype=torch.long).to(device)

    n_samples = len(X_train)
    optimizer = torch.optim.Adam(net.parameters(), lr=BOLT_LR)

    net.train()
    for epoch in range(BOLT_EPOCHS):
        # Shuffle indices
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, BOLT_BATCH_SIZE):
            end = min(start + BOLT_BATCH_SIZE, n_samples)
            idx = perm[start:end]

            ctx_b = ctx_tensor[idx]
            lvl_b = lvl_tensor[idx]
            ac_b = ac_tensor[idx]
            aux_b = None
            if aux_tensor is not None:
                n_aux = model._n_aux_features
                ctx_len = model._config["context_length"]
                aux_b = aux_tensor[idx].reshape(-1, ctx_len, n_aux)
            y_b = y_tensor[idx]

            optimizer.zero_grad()
            logits_dict = net(ctx_b, lvl_b, ac_b, aux_features=aux_b)
            logits = logits_dict[model._prediction_horizons[0]]
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0:
            logger.info(
                "    BOLT (sigma=%.2f) epoch %d/%d: loss=%.4f",
                sigma, epoch + 1, BOLT_EPOCHS, avg_loss,
            )


def main() -> None:
    """Run BOLT ceiling diagnostic."""
    logger.info("=== BOLT Ceiling Diagnostic ===")

    # --- Load data ---
    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))
    logger.info("Loaded %d tickers", len(tickers))

    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)

    # --- Build D1 pipeline ---
    logger.info("Building D1 pipeline...")
    X_train, tr_rets, tr_valid, tr_lvls, _, _ = _build_d1_pipeline(
        train_prices, None, tickers, None, 0,
    )
    X_test, te_rets, te_valid, te_lvls, _, _ = _build_d1_pipeline(
        test_prices, None, tickers, None, 0,
    )
    logger.info("Train: %d windows, Test: %d windows", len(X_train), len(X_test))

    # --- Compute labels ---
    boundaries = compute_quantile_boundaries(
        tr_rets, tr_lvls, tr_valid, PERCENTILES, per_level=True,
    )
    y_train = assign_quantile_labels(tr_rets, tr_lvls, boundaries)
    y_test = assign_quantile_labels(te_rets, te_lvls, boundaries)

    # Class weights
    valid_labels = y_train[tr_valid]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    # --- Phase 1: CE warm-start ---
    logger.info("Phase 1: CE warm-start (%d epochs)...", CE_WARMUP_EPOCHS)
    ce_kwargs = dict(MODEL_KWARGS)
    ce_kwargs["epochs"] = CE_WARMUP_EPOCHS

    model = WaveletGPT(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=N_CLASSES,
        class_weights=class_weights, input_mode="continuous",
        n_aux_features=N_AUX_FEATURES, **ce_kwargs,
    )

    t0 = time.time()
    ce_metrics = model.fit(X_train, y_train.astype(np.float64))
    ce_time = time.time() - t0
    logger.info("CE warm-start done in %.1fs (loss=%.4f)", ce_time, ce_metrics["train_loss"])

    # CE baseline accuracy on test set
    ce_preds = model.predict(X_test).astype(np.int64)
    ce_quantile_acc = float(np.mean(ce_preds[te_valid] == y_test[te_valid]))
    ce_econ_dir = compute_econ_dir_accuracy(ce_preds, te_rets, te_valid)
    logger.info(
        "CE baseline: quantile_acc=%.1f%%, econ_dir=%.1f%%",
        ce_quantile_acc * 100, ce_econ_dir * 100,
    )

    # --- Phase 2: BOLT fine-tuning at each sigma ---
    sigma_results: list[dict] = []

    for sigma in SIGMA_VALUES:
        logger.info("Phase 2: BOLT fine-tuning (sigma=%.2f, %d epochs)...", sigma, BOLT_EPOCHS)

        t0 = time.time()
        bolt_finetune(model, X_train, y_train, sigma)
        bolt_time = time.time() - t0

        # Evaluate on test set
        bolt_preds = model.predict(X_test).astype(np.int64)
        bolt_quantile_acc = float(np.mean(bolt_preds[te_valid] == y_test[te_valid]))
        bolt_econ_dir = compute_econ_dir_accuracy(bolt_preds, te_rets, te_valid)

        logger.info(
            "  sigma=%.2f: quantile_acc=%.1f%%, econ_dir=%.1f%% (%.1fs)",
            sigma, bolt_quantile_acc * 100, bolt_econ_dir * 100, bolt_time,
        )

        sigma_results.append({
            "sigma": sigma,
            "quantile_acc": bolt_quantile_acc,
            "econ_dir": bolt_econ_dir,
            "train_time": bolt_time,
        })

    # --- Summary ---
    best_result = max(sigma_results, key=lambda r: r["econ_dir"])
    ceiling_econ_dir = best_result["econ_dir"]
    ceiling_quantile_acc = best_result["quantile_acc"]
    best_sigma = best_result["sigma"]

    headroom_econ = ceiling_econ_dir - CE_ECON_DIR
    headroom_quantile = ceiling_quantile_acc - ce_quantile_acc

    w = 100
    print()
    print("=" * w)
    print("  BOLT CEILING DIAGNOSTIC")
    print("=" * w)
    print(f"  Train windows:        {len(X_train)}")
    print(f"  Test windows:         {len(X_test)}")
    print(f"  Valid test samples:   {int(te_valid.sum())}")
    print("-" * w)
    print(f"  {'Phase':<30} {'Quantile Acc':>14} {'Econ Dir Acc':>14}")
    print("-" * w)
    print(f"  {'CE warm-start (' + str(CE_WARMUP_EPOCHS) + ' epochs)':<30}"
          f" {ce_quantile_acc:>13.1%} {ce_econ_dir:>13.1%}")
    for sr in sigma_results:
        label = f"BOLT sigma={sr['sigma']:.2f} (+{BOLT_EPOCHS} epochs)"
        print(f"  {label:<30} {sr['quantile_acc']:>13.1%} {sr['econ_dir']:>13.1%}")
    print("-" * w)
    print(f"  {'Best ceiling (sigma=' + str(best_sigma) + ')':<30}"
          f" {ceiling_quantile_acc:>13.1%} {ceiling_econ_dir:>13.1%}")
    print(f"  {'CE reference':<30} {ce_quantile_acc:>13.1%} {CE_ECON_DIR:>13.1%}")
    print(f"  {'Headroom':<30} {headroom_quantile:>+13.1%} {headroom_econ:>+13.1%}")
    print("-" * w)

    if headroom_econ < 0.02:
        interpretation = (
            "AT CEILING: Less than 2pp headroom. The current CE loss is "
            "already near-optimal for this feature representation. "
            "Improving accuracy requires richer features, not better losses."
        )
    elif headroom_econ < 0.05:
        interpretation = (
            f"SMALL HEADROOM: {headroom_econ:.1%} gap. "
            "Moderate gains may be possible via loss function or "
            "training procedure improvements."
        )
    else:
        interpretation = (
            f"SIGNIFICANT HEADROOM: {headroom_econ:.1%} gap. "
            "CE training is substantially sub-optimal. "
            "Loss function, regularization, or optimization changes "
            "could capture meaningful additional accuracy."
        )

    print(f"  Interpretation:       {interpretation}")
    print("=" * w)

    # --- Save results ---
    results = {
        "ce_warmup_epochs": CE_WARMUP_EPOCHS,
        "bolt_epochs_per_sigma": BOLT_EPOCHS,
        "ce_baseline": {
            "quantile_acc": ce_quantile_acc,
            "econ_dir": ce_econ_dir,
            "train_loss": ce_metrics["train_loss"],
            "train_time": ce_time,
        },
        "sigma_results": sigma_results,
        "ceiling": {
            "best_sigma": best_sigma,
            "quantile_acc": ceiling_quantile_acc,
            "econ_dir": ceiling_econ_dir,
        },
        "headroom": {
            "quantile_acc": headroom_quantile,
            "econ_dir": headroom_econ,
        },
        "interpretation": interpretation,
        "model_kwargs": MODEL_KWARGS,
        "bolt_lr": BOLT_LR,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_valid_test": int(te_valid.sum()),
        "context_length": CONTEXT_LENGTH,
        "n_aux_features": N_AUX_FEATURES,
        "n_classes": N_CLASSES,
    }
    out_path = save_results("bolt_ceiling", results)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
