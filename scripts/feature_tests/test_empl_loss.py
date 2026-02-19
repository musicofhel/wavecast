#!/usr/bin/env python3
"""EMPL Loss Experiment: Earth Mover's Pinball Loss for ordinal-aware training.

Standard cross-entropy treats all misclassifications equally: predicting
class 0 (strong_down) when the truth is class 4 (strong_up) has the same
penalty as predicting class 1 (down). This is wrong for ordinal targets
where 0 < 1 < 2 < 3 < 4.

Earth Mover's Distance (Wasserstein-1) on the CDF respects class ordering:
the loss is proportional to how many ordinal steps the prediction is off.
This should reduce wild misclassifications (predicting strong_up when the
market crashed) and improve direction accuracy on large moves.

Implementation:
  - Compute predicted CDF: cumsum(softmax(logits), dim=1)
  - Compute target CDF: cumsum(one_hot(target), dim=1)
  - EMD = sum |CDF_pred - CDF_target| across classes
  - Also measures adjacent-class confusion to verify ordinal improvement.

Usage:
    python -m scripts.feature_tests.test_empl_loss
"""

from __future__ import annotations  # noqa: I001

import logging
import time

import numpy as np
import torch
import torch.nn as nn
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


# ---------------------------------------------------------------------------
# EMPL loss
# ---------------------------------------------------------------------------


class EMPLLoss(nn.Module):
    """Earth Mover's Pinball Loss: ordinal-aware CDF matching.

    The Earth Mover's Distance between two 1-D discrete distributions with
    support {0, 1, ..., C-1} equals the L1 distance between their CDFs:

        EMD(p, q) = sum_{j=0}^{C-1} |CDF_p(j) - CDF_q(j)|

    This is differentiable through the softmax → cumsum pipeline and
    penalizes predictions proportionally to the ordinal distance from truth.
    """

    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute EMD between predicted and target CDFs.

        Args:
            logits: (batch, n_classes) raw logits
            targets: (batch,) integer class labels
        """
        probs = nn.functional.softmax(logits, dim=1)                  # (B, C)
        cum_probs = torch.cumsum(probs, dim=1)                        # (B, C)

        target_one_hot = nn.functional.one_hot(targets, self.n_classes).float()  # (B, C)
        target_cum = torch.cumsum(target_one_hot, dim=1)             # (B, C)

        # EMD: L1 distance between CDFs
        emd = torch.sum(torch.abs(cum_probs - target_cum), dim=1)  # (B,)
        return emd.mean()


# ---------------------------------------------------------------------------
# Data parsing helper
# ---------------------------------------------------------------------------


def parse_x(
    X: np.ndarray,
    ctx_len: int = CONTEXT_LENGTH,
    n_aux: int = N_AUX_FEATURES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse the flat X array into (ctx, lvl, ac, aux) numpy arrays."""
    ctx = X[:, :ctx_len].astype(np.float32)
    aux_end = ctx_len + ctx_len * n_aux
    aux = X[:, ctx_len:aux_end].astype(np.float32).reshape(-1, ctx_len, n_aux)
    lvl = X[:, aux_end].astype(np.int64)
    ac = X[:, aux_end + 1].astype(np.int64)
    return ctx, lvl, ac, aux


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _train_with_custom_loss(
    net: WaveletGPTNet,
    criterion: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
) -> float:
    """Manual training loop with a custom loss criterion. Returns final train loss."""
    ctx, lvl, ac, aux = parse_x(X_train)
    y = y_train.astype(np.int64)

    tensors = [
        torch.tensor(ctx, dtype=torch.float32),
        torch.tensor(lvl, dtype=torch.long),
        torch.tensor(ac, dtype=torch.long),
        torch.tensor(aux.reshape(len(aux), -1), dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    ]

    loader = DataLoader(
        TensorDataset(*tensors), batch_size=MODEL_KWARGS["batch_size"], shuffle=True,
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=MODEL_KWARGS["learning_rate"])

    avg_loss = 0.0
    for epoch in range(MODEL_KWARGS["epochs"]):
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
            logger.info("  EMPL epoch %d/%d  loss=%.4f", epoch + 1, MODEL_KWARGS["epochs"], avg_loss)

    return avg_loss


def _predict_from_net(
    net: WaveletGPTNet,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Adjacent-class confusion analysis
# ---------------------------------------------------------------------------


def compute_adjacent_confusion(
    pred_labels: np.ndarray,
    actual_labels: np.ndarray,
    valid_mask: np.ndarray,
    n_classes: int = N_CLASSES,
) -> dict:
    """Compute confusion statistics focused on ordinal distance.

    Returns:
      - confusion_matrix: (C, C) counts
      - mean_ordinal_error: average |pred - actual| (lower = better)
      - adjacent_rate: fraction of errors that are off-by-1
      - far_error_rate: fraction of errors that are off-by-2+
    """
    vp = pred_labels[valid_mask]
    va = actual_labels[valid_mask]

    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for p, a in zip(vp, va, strict=True):
        cm[a, p] += 1

    # Ordinal distance metrics
    dists = np.abs(vp.astype(np.int64) - va.astype(np.int64))
    mean_ord_err = float(np.mean(dists))

    errors = dists > 0
    n_errors = int(errors.sum())
    if n_errors > 0:
        adjacent_rate = float(np.mean(dists[errors] == 1))
        far_error_rate = float(np.mean(dists[errors] >= 2))
    else:
        adjacent_rate = 0.0
        far_error_rate = 0.0

    return {
        "confusion_matrix": cm,
        "mean_ordinal_error": mean_ord_err,
        "adjacent_rate": adjacent_rate,
        "far_error_rate": far_error_rate,
        "n_errors": n_errors,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("EMPL loss experiment: ordinal-aware Earth Mover's distance")

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

    # Class weights for CE baseline
    valid_labels = y_train[tr_valid]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- CE Baseline ---
    logger.info("Training CE baseline...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_ce = WaveletGPT(
        vocab_size=1,
        context_length=CONTEXT_LENGTH,
        task="return_quantile",
        n_output_classes=N_CLASSES,
        class_weights=class_weights,
        input_mode="continuous",
        n_aux_features=N_AUX_FEATURES,
        **MODEL_KWARGS,
    )
    t0 = time.time()
    metrics_ce = model_ce.fit(X_train, y_train.astype(np.float64))
    ce_time = time.time() - t0
    pred_ce = model_ce.predict(X_test).astype(np.int64)
    m_ce = evaluate_5_metrics(pred_ce, te_rets, y_test, te_valid)
    print_5_metrics("CE Baseline", m_ce)
    adj_ce = compute_adjacent_confusion(pred_ce, y_test, te_valid)

    # --- EMPL Loss ---
    logger.info("Training with EMPL loss...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    net_empl = WaveletGPTNet(
        vocab_size=1,
        context_length=CONTEXT_LENGTH,
        task="return_quantile",
        n_output_classes=N_CLASSES,
        input_mode="continuous",
        n_aux_features=N_AUX_FEATURES,
        **NET_KWARGS,
    ).to(device)

    empl_criterion = EMPLLoss(n_classes=N_CLASSES)
    t0 = time.time()
    empl_loss = _train_with_custom_loss(net_empl, empl_criterion, X_train, y_train, device)
    empl_time = time.time() - t0
    pred_empl = _predict_from_net(net_empl, X_test, device)
    m_empl = evaluate_5_metrics(pred_empl, te_rets, y_test, te_valid)
    print_5_metrics("EMPL Loss", m_empl)
    adj_empl = compute_adjacent_confusion(pred_empl, y_test, te_valid)

    # --- Ordinal Comparison ---
    verdict = verdict_from_metrics(m_ce, m_empl)

    w = 100
    print("\n" + "=" * w)
    print("EMPL vs CE Comparison")
    print("=" * w)
    print(f"  CE   econ_dir={m_ce['econ_dir']:.1%}  transition={m_ce['transition_acc']:.1%}"
          f"  large_move={m_ce['large_move_acc']:.1%}  sharpe={m_ce['sharpe_costs']:+.3f}")
    print(f"  EMPL econ_dir={m_empl['econ_dir']:.1%}  transition={m_empl['transition_acc']:.1%}"
          f"  large_move={m_empl['large_move_acc']:.1%}  sharpe={m_empl['sharpe_costs']:+.3f}")
    delta_econ = m_empl["econ_dir"] - m_ce["econ_dir"]
    delta_trans = m_empl["transition_acc"] - m_ce["transition_acc"]
    delta_large = m_empl["large_move_acc"] - m_ce["large_move_acc"]
    delta_sharpe = m_empl["sharpe_costs"] - m_ce["sharpe_costs"]
    print(f"  Delta: econ_dir={delta_econ:+.1%}  transition={delta_trans:+.1%}"
          f"  large_move={delta_large:+.1%}  sharpe={delta_sharpe:+.3f}")

    # Ordinal error analysis
    print("\n  Ordinal Error Analysis:")
    print(f"    CE   mean_ord_err={adj_ce['mean_ordinal_error']:.3f}"
          f"  adjacent_rate={adj_ce['adjacent_rate']:.1%}"
          f"  far_error_rate={adj_ce['far_error_rate']:.1%}")
    print(f"    EMPL mean_ord_err={adj_empl['mean_ordinal_error']:.3f}"
          f"  adjacent_rate={adj_empl['adjacent_rate']:.1%}"
          f"  far_error_rate={adj_empl['far_error_rate']:.1%}")

    ord_improved = adj_empl["mean_ordinal_error"] < adj_ce["mean_ordinal_error"]
    far_improved = adj_empl["far_error_rate"] < adj_ce["far_error_rate"]
    if ord_improved:
        print(f"    EMPL reduced mean ordinal error by "
              f"{adj_ce['mean_ordinal_error'] - adj_empl['mean_ordinal_error']:.3f}")
    if far_improved:
        print(f"    EMPL reduced far-error rate by "
              f"{adj_ce['far_error_rate'] - adj_empl['far_error_rate']:.1%}")

    # Confusion matrices
    print("\n  CE Confusion Matrix (rows=actual, cols=predicted):")
    cm_ce = adj_ce["confusion_matrix"]
    header = "     " + "  ".join(f"c{i:d}" for i in range(N_CLASSES))
    print(f"    {header}")
    for i in range(N_CLASSES):
        row = "  ".join(f"{cm_ce[i, j]:4d}" for j in range(N_CLASSES))
        print(f"    c{i} {row}")

    print("\n  EMPL Confusion Matrix (rows=actual, cols=predicted):")
    cm_empl = adj_empl["confusion_matrix"]
    print(f"    {header}")
    for i in range(N_CLASSES):
        row = "  ".join(f"{cm_empl[i, j]:4d}" for j in range(N_CLASSES))
        print(f"    c{i} {row}")

    print(f"\n  VERDICT: {verdict}")
    print("=" * w)

    save_results("empl_loss", {
        "ce_baseline": m_ce,
        "empl_loss": m_empl,
        "ce_train_loss": metrics_ce["train_loss"],
        "ce_train_time": ce_time,
        "empl_train_loss": empl_loss,
        "empl_train_time": empl_time,
        "verdict": verdict,
        "delta": {
            "econ_dir": delta_econ,
            "transition_acc": delta_trans,
            "large_move_acc": delta_large,
            "sharpe_costs": delta_sharpe,
        },
        "ordinal_analysis": {
            "ce_mean_ord_err": adj_ce["mean_ordinal_error"],
            "empl_mean_ord_err": adj_empl["mean_ordinal_error"],
            "ce_adjacent_rate": adj_ce["adjacent_rate"],
            "empl_adjacent_rate": adj_empl["adjacent_rate"],
            "ce_far_error_rate": adj_ce["far_error_rate"],
            "empl_far_error_rate": adj_empl["far_error_rate"],
            "ordinal_improved": ord_improved,
            "far_error_improved": far_improved,
        },
    })


if __name__ == "__main__":
    main()
