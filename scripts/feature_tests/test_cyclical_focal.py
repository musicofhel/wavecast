#!/usr/bin/env python3
"""Cyclical Focal Loss Experiment.

Focal loss where gamma varies sinusoidally over training epochs:
  gamma(epoch) = gamma_max * sin(pi * epoch / total_epochs)

This starts at gamma=0 (pure CE), peaks at gamma_max mid-training (hard
example mining), then anneals back to gamma=0. The intuition is that early
training benefits from stable CE gradients, mid-training needs hard-example
focus, and late-training benefits from gentle fine-tuning.

Sweeps gamma_max in {1.0, 2.0, 3.0}. Reports best alongside CE baseline.

Usage:
    python -m scripts.feature_tests.test_cyclical_focal
"""

from __future__ import annotations  # noqa: I001

import logging
import math
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
GAMMA_MAX_VALUES = [1.0, 2.0, 3.0]


class CyclicalFocalLoss(nn.Module):
    """Focal loss with sinusoidally scheduled gamma."""

    def __init__(self, gamma_max: float, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma_max = gamma_max
        self.weight = weight
        self.current_gamma = 0.0

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        self.current_gamma = self.gamma_max * math.sin(math.pi * epoch / total_epochs)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = f_nn.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.current_gamma * ce).mean()


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


def train_cyclical_focal(
    net: WaveletGPTNet,
    criterion: CyclicalFocalLoss,
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
) -> float:
    """Train with cyclical focal loss. Returns final epoch loss."""
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
    total_epochs = MODEL_KWARGS["epochs"]

    avg_loss = 0.0
    for epoch in range(total_epochs):
        criterion.set_epoch(epoch, total_epochs)
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
        if (epoch + 1) % 5 == 0:
            logger.info(
                "  epoch %d/%d  gamma=%.3f  loss=%.4f",
                epoch + 1, total_epochs, criterion.current_gamma, avg_loss,
            )
    return avg_loss


def predict_from_net(net: WaveletGPTNet, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Get argmax predictions from a raw WaveletGPTNet."""
    ctx, lvl, ac, aux = parse_x(X)
    net.eval()
    with torch.no_grad():
        logits_dict = net(
            torch.tensor(ctx, dtype=torch.float32).to(device),
            torch.tensor(lvl, dtype=torch.long).to(device),
            torch.tensor(ac, dtype=torch.long).to(device),
            aux_features=torch.tensor(aux, dtype=torch.float32).to(device),
        )
    return logits_dict[1].argmax(dim=-1).cpu().numpy()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Cyclical focal loss experiment")

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
    pred_ce = model_ce.predict(X_test).astype(np.int64)
    m_ce = evaluate_5_metrics(pred_ce, te_rets, y_test, te_valid)
    print_5_metrics("CE Baseline", m_ce)

    # --- Sweep gamma_max ---
    best_metrics = None
    best_gamma = None
    all_results: dict[str, dict] = {}

    for gamma_max in GAMMA_MAX_VALUES:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        logger.info("Training cyclical focal (gamma_max=%.1f)...", gamma_max)
        net = WaveletGPTNet(
            vocab_size=1, context_length=CONTEXT_LENGTH,
            task="return_quantile", n_output_classes=N_CLASSES,
            input_mode="continuous", n_aux_features=N_AUX_FEATURES,
            **NET_KWARGS,
        ).to(device)

        criterion = CyclicalFocalLoss(gamma_max=gamma_max, weight=cw_tensor.to(device))
        t0 = time.time()
        final_loss = train_cyclical_focal(net, criterion, X_train, y_train, device)
        train_time = time.time() - t0

        preds = predict_from_net(net, X_test, device)
        m = evaluate_5_metrics(preds, te_rets, y_test, te_valid)
        print_5_metrics(f"Cyclical Focal (gamma_max={gamma_max})", m)

        all_results[f"gamma_{gamma_max}"] = {
            "metrics": m, "train_loss": final_loss, "train_time": train_time,
        }

        if best_metrics is None or m["econ_dir"] > best_metrics["econ_dir"]:
            best_metrics = m
            best_gamma = gamma_max

    # --- Verdict ---
    verdict = verdict_from_metrics(m_ce, best_metrics)

    w = 100
    print("\n" + "=" * w)
    print(f"Cyclical Focal Loss -- Best: gamma_max={best_gamma}")
    print("=" * w)
    print(f"  CE  econ_dir={m_ce['econ_dir']:.1%}  transition={m_ce['transition_acc']:.1%}"
          f"  large_move={m_ce['large_move_acc']:.1%}  sharpe={m_ce['sharpe_costs']:+.3f}")
    print(f"  Best econ_dir={best_metrics['econ_dir']:.1%}"
          f"  transition={best_metrics['transition_acc']:.1%}"
          f"  large_move={best_metrics['large_move_acc']:.1%}"
          f"  sharpe={best_metrics['sharpe_costs']:+.3f}")
    print(f"\n  VERDICT: {verdict}")
    print("=" * w)

    save_results("cyclical_focal", {
        "ce_baseline": m_ce, "ce_train_time": ce_time,
        "gamma_max_sweep": all_results,
        "best_gamma_max": best_gamma,
        "best_metrics": best_metrics,
        "verdict": verdict,
    })


if __name__ == "__main__":
    main()
