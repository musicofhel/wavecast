#!/usr/bin/env python3
"""Focal Loss Flat-Check Diagnostic.

Round 1 focal loss (gamma=2.0) got FAIL: tiny econ_dir/transition degradation.
This script re-runs with 5-metric evaluation to determine whether the FAIL was
honest (genuinely worse but balanced predictions) or a flat-bias collapse like
heteroscedastic loss exhibited.

Usage:
    python -m scripts.feature_tests.test_focal_flatcheck
"""

from __future__ import annotations  # noqa: I001

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.feature_tests.exp2_helpers import evaluate_5_metrics, print_5_metrics, save_results
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


class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017) for class-imbalanced classification."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def _train_with_custom_loss(
    net: WaveletGPTNet,
    criterion: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_ref: WaveletGPT,
) -> float:
    """Manual training loop with a custom loss criterion. Returns final train loss."""
    device = model_ref._device
    ctx, lvl, ac, aux = model_ref._parse_x(X_train)
    y = y_train.astype(np.int64)

    tensors = [
        torch.tensor(ctx, dtype=torch.float32),
        torch.tensor(lvl, dtype=torch.long),
        torch.tensor(ac, dtype=torch.long),
    ]
    if aux is not None:
        tensors.append(torch.tensor(aux.reshape(len(aux), -1), dtype=torch.float32))
    tensors.append(torch.tensor(y, dtype=torch.long))

    loader = DataLoader(
        TensorDataset(*tensors), batch_size=MODEL_KWARGS["batch_size"], shuffle=True,
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=MODEL_KWARGS["learning_rate"])
    has_aux = aux is not None
    target_idx = 4 if has_aux else 3

    avg_loss = 0.0
    for _epoch in range(MODEL_KWARGS["epochs"]):
        net.train()
        total, n = 0.0, 0
        for batch in loader:
            ctx_b = batch[0].to(device)
            lvl_b = batch[1].to(device)
            ac_b = batch[2].to(device)
            aux_b = batch[3].to(device).reshape(-1, CONTEXT_LENGTH, N_AUX_FEATURES) if has_aux else None
            tgt = batch[target_idx].to(device)

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


def _predict_from_net(
    net: WaveletGPTNet, X: np.ndarray, model_ref: WaveletGPT,
) -> np.ndarray:
    """Get argmax predictions from a raw WaveletGPTNet."""
    device = model_ref._device
    ctx, lvl, ac, aux = model_ref._parse_x(X)
    net.eval()
    with torch.no_grad():
        aux_t = None
        if aux is not None:
            aux_t = torch.tensor(aux, dtype=torch.float32).to(device)
        logits_dict = net(
            torch.tensor(ctx, dtype=torch.float32).to(device),
            torch.tensor(lvl, dtype=torch.long).to(device),
            torch.tensor(ac, dtype=torch.long).to(device),
            aux_features=aux_t,
        )
    return logits_dict[1].argmax(dim=-1).cpu().numpy()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Focal loss flat-check diagnostic")

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
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- CE Baseline ---
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

    # --- Focal Loss ---
    logger.info("Training focal loss (gamma=2.0)...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_focal = WaveletGPTNet(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=N_CLASSES,
        input_mode="continuous", n_aux_features=N_AUX_FEATURES,
        **{k: MODEL_KWARGS[k] for k in ("embed_dim", "num_heads", "num_layers", "dropout")},
    ).to(device)

    focal_criterion = FocalLoss(gamma=2.0, weight=cw_tensor.to(device))
    t0 = time.time()
    focal_loss = _train_with_custom_loss(net_focal, focal_criterion, X_train, y_train, model_ce)
    focal_time = time.time() - t0
    pred_focal = _predict_from_net(net_focal, X_test, model_ce)
    m_focal = evaluate_5_metrics(pred_focal, te_rets, y_test, te_valid)
    print_5_metrics("Focal Loss (gamma=2.0)", m_focal)

    # --- Diagnosis ---
    r1_path = Path.home() / ".wavecast" / "audit" / "feature_tests" / "focal_loss_results.json"
    r1 = json.loads(r1_path.read_text()) if r1_path.exists() else None

    flat_ce = m_ce["pred_dist"]["flat"]
    flat_focal = m_focal["pred_dist"]["flat"]
    collapsed = flat_focal > 0.60

    print("\n" + "=" * 100)
    print("DIAGNOSIS: Focal Loss Flat-Check")
    print("=" * 100)
    print(f"  CE flat%:    {flat_ce:.1%}")
    print(f"  Focal flat%: {flat_focal:.1%}")
    if collapsed:
        print("  COLLAPSED: focal loss collapsed to flat predictions (>60% flat)")
    else:
        print("  BALANCED: focal loss produced distributed predictions")
        print("  -> Round 1 FAIL was honest (genuinely worse, not flat-biased)")
    if r1:
        print(f"\n  Round 1 focal econ_dir: {r1['challenger_mean']['econ_dir']:.1%}")
        print(f"  Round 2 focal econ_dir: {m_focal['econ_dir']:.1%}")
    print("=" * 100)

    save_results("focal_flatcheck", {
        "ce_baseline": m_ce,
        "focal_loss": m_focal,
        "ce_train_time": ce_time,
        "focal_train_time": focal_time,
        "focal_train_loss": focal_loss,
        "collapsed_to_flat": collapsed,
        "round1_results": r1,
    })


if __name__ == "__main__":
    main()
