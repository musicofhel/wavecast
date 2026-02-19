#!/usr/bin/env python3
"""Arctan Pinball Loss: Smooth quantile regression via arctan approximation.

Replaces the 5-class classification head with a 3-output regression head
predicting quantile values at tau = {0.25, 0.5, 0.75}. Uses an arctan-smoothed
pinball loss to avoid the non-differentiable kink at the origin.

For each quantile level tau, the smooth pinball loss is:
  L(err, tau) = err * (tau - (1/pi * arctan(err / delta) + 0.5))
where delta controls smoothing (delta -> 0 recovers the exact pinball loss).

Direction is derived from the median quantile (tau=0.5) sign. Class labels
are assigned for evaluate_5_metrics compatibility:
  Q(0.5) > threshold  -> class 3 (up)
  Q(0.5) < -threshold -> class 1 (down)
  |Q(0.5)| <= threshold -> class 2 (flat)

Usage:
    python -m scripts.feature_tests.test_arctan_pinball
"""

from __future__ import annotations  # noqa: I001

import logging
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.feature_tests.exp2_helpers import (
    MID,
    evaluate_5_metrics,
    print_5_metrics,
    save_results,
    verdict_from_metrics,
)
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
from wavecast.targets.returns import assign_quantile_labels, compute_quantile_boundaries

logger = logging.getLogger(__name__)

NET_KWARGS = {k: MODEL_KWARGS[k] for k in ("embed_dim", "num_heads", "num_layers", "dropout")}
EMBED_DIM = MODEL_KWARGS["embed_dim"]
TAU_LEVELS = [0.25, 0.5, 0.75]
N_QUANTILES = len(TAU_LEVELS)
DELTA = 0.01  # smoothing parameter


class ArctanPinballLoss(nn.Module):
    """Smooth pinball loss using arctan approximation.

    For each tau in tau_levels:
      L(err, tau) = err * (tau - (1/pi * arctan(err/delta) + 0.5))
    Summed across quantile levels, averaged over batch.
    """

    def __init__(self, tau_levels: list[float], delta: float = DELTA) -> None:
        super().__init__()
        self.register_buffer("tau", torch.tensor(tau_levels, dtype=torch.float32))
        self.delta = delta

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute smooth pinball loss.

        Args:
            preds: (B, Q) predicted quantile values.
            targets: (B,) actual return values.
        """
        y = targets.unsqueeze(-1)  # (B, 1)
        err = y - preds  # (B, Q)
        indicator = 1.0 / math.pi * torch.atan(err / self.delta) + 0.5
        loss_per_q = err * (self.tau.unsqueeze(0) - indicator)  # (B, Q)
        return loss_per_q.sum(dim=-1).mean()


class QuantileWaveletModel(nn.Module):
    """Transformer encoder + quantile regression head.

    Replicates WaveletGPTNet's encoder (input_proj, pos_embed, level_embed,
    asset_class_embed, causal transformer, ln_f) but outputs Q quantile
    values instead of class logits.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        context_length: int = CONTEXT_LENGTH,
        n_aux_features: int = N_AUX_FEATURES,
        n_quantiles: int = N_QUANTILES,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.n_aux_features = n_aux_features

        self.input_proj = nn.Linear(1, embed_dim)
        if n_aux_features > 0:
            self.aux_proj = nn.Linear(n_aux_features, embed_dim)
        self.pos_embed = nn.Embedding(context_length, embed_dim)
        self.level_embed = nn.Embedding(6, embed_dim)
        self.asset_class_embed = nn.Embedding(7, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.quantile_head = nn.Linear(embed_dim, n_quantiles)

    def forward(
        self,
        ctx: torch.Tensor,
        lvl: torch.Tensor,
        ac: torch.Tensor,
        aux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s = ctx.shape
        pos = torch.arange(s, device=ctx.device).unsqueeze(0).expand(b, -1)

        x = self.input_proj(ctx.unsqueeze(-1)) + self.pos_embed(pos)
        if self.n_aux_features > 0 and aux is not None:
            x = x + self.aux_proj(aux)
        x = x + self.level_embed(lvl).unsqueeze(1)
        x = x + self.asset_class_embed(ac).unsqueeze(1)

        mask = torch.triu(torch.ones(s, s, device=ctx.device, dtype=torch.bool), diagonal=1)
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.ln_f(x)

        return self.quantile_head(x[:, -1, :])  # (B, Q)


def parse_x(
    X: np.ndarray, ctx_len: int = CONTEXT_LENGTH, n_aux: int = N_AUX_FEATURES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ctx = X[:, :ctx_len].astype(np.float32)
    aux_end = ctx_len + ctx_len * n_aux
    aux = X[:, ctx_len:aux_end].astype(np.float32).reshape(-1, ctx_len, n_aux)
    lvl = X[:, aux_end].astype(np.int64)
    ac = X[:, aux_end + 1].astype(np.int64)
    return ctx, lvl, ac, aux


def quantile_to_labels(median_preds: np.ndarray, threshold: float = MIN_RETURN_THRESHOLD) -> np.ndarray:
    """Convert median quantile predictions to 5-class labels.

    Q(0.5) > threshold  -> 3 (up)
    Q(0.5) < -threshold -> 1 (down)
    else                -> 2 (flat)
    """
    labels = np.full(len(median_preds), MID, dtype=np.int64)
    labels[median_preds > threshold] = 3
    labels[median_preds < -threshold] = 1
    return labels


def train_quantile_model(
    model: QuantileWaveletModel,
    X_train: np.ndarray,
    y_returns: np.ndarray,
    valid_mask: np.ndarray,
    device: torch.device,
) -> float:
    """Train quantile regression model on continuous returns. Returns final loss."""
    ctx, lvl, ac, aux = parse_x(X_train)

    keep = valid_mask & ~np.isnan(y_returns)
    ctx, lvl, ac, aux = ctx[keep], lvl[keep], ac[keep], aux[keep]
    targets = y_returns[keep].astype(np.float32)

    dataset = TensorDataset(
        torch.tensor(ctx, dtype=torch.float32),
        torch.tensor(lvl, dtype=torch.long),
        torch.tensor(ac, dtype=torch.long),
        torch.tensor(aux.reshape(len(aux), -1), dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=MODEL_KWARGS["batch_size"], shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MODEL_KWARGS["learning_rate"])
    criterion = ArctanPinballLoss(TAU_LEVELS, delta=DELTA).to(device)

    avg_loss = 0.0
    for epoch in range(MODEL_KWARGS["epochs"]):
        model.train()
        total, n = 0.0, 0
        for batch in loader:
            ctx_b = batch[0].to(device)
            lvl_b = batch[1].to(device)
            ac_b = batch[2].to(device)
            aux_b = batch[3].to(device).reshape(-1, CONTEXT_LENGTH, N_AUX_FEATURES)
            tgt = batch[4].to(device)

            optimizer.zero_grad()
            q_preds = model(ctx_b, lvl_b, ac_b, aux=aux_b)
            loss = criterion(q_preds, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            n += 1
        avg_loss = total / max(n, 1)
        if (epoch + 1) % 5 == 0:
            logger.info("  epoch %d/%d  loss=%.6f", epoch + 1, MODEL_KWARGS["epochs"], avg_loss)
    return avg_loss


def predict_quantiles(
    model: QuantileWaveletModel, X: np.ndarray, device: torch.device,
) -> np.ndarray:
    """Get quantile predictions. Returns (N, Q)."""
    ctx, lvl, ac, aux = parse_x(X)
    model.eval()
    with torch.no_grad():
        q_preds = model(
            torch.tensor(ctx, dtype=torch.float32).to(device),
            torch.tensor(lvl, dtype=torch.long).to(device),
            torch.tensor(ac, dtype=torch.long).to(device),
            aux=torch.tensor(aux, dtype=torch.float32).to(device),
        )
    return q_preds.cpu().numpy()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Arctan pinball loss experiment")

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

    # --- Arctan Pinball ---
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Training arctan pinball quantile model (tau=%s, delta=%.3f)...", TAU_LEVELS, DELTA)

    qmodel = QuantileWaveletModel(**NET_KWARGS).to(device)
    t0 = time.time()
    q_loss = train_quantile_model(qmodel, X_train, tr_rets, tr_valid, device)
    q_time = time.time() - t0
    logger.info("Quantile training done in %.1fs (final loss=%.6f)", q_time, q_loss)

    q_preds = predict_quantiles(qmodel, X_test, device)
    median_preds = q_preds[:, 1]  # tau=0.5 is index 1
    pred_labels = quantile_to_labels(median_preds)
    m_q = evaluate_5_metrics(pred_labels, te_rets, y_test, te_valid)
    print_5_metrics("Arctan Pinball Quantile", m_q)

    # --- Comparison ---
    verdict = verdict_from_metrics(m_ce, m_q)

    w = 100
    print("\n" + "=" * w)
    print("Arctan Pinball Loss Results")
    print("=" * w)
    print(f"  CE  econ_dir={m_ce['econ_dir']:.1%}  transition={m_ce['transition_acc']:.1%}"
          f"  large_move={m_ce['large_move_acc']:.1%}  sharpe={m_ce['sharpe_costs']:+.3f}")
    print(f"  APB econ_dir={m_q['econ_dir']:.1%}  transition={m_q['transition_acc']:.1%}"
          f"  large_move={m_q['large_move_acc']:.1%}  sharpe={m_q['sharpe_costs']:+.3f}")

    # Quantile spread analysis
    q25_mean = np.mean(q_preds[te_valid, 0])
    q50_mean = np.mean(q_preds[te_valid, 1])
    q75_mean = np.mean(q_preds[te_valid, 2])
    iqr_mean = np.mean(q_preds[te_valid, 2] - q_preds[te_valid, 0])
    print(f"\n  Quantile spread: Q25={q25_mean:.5f}  Q50={q50_mean:.5f}  Q75={q75_mean:.5f}")
    print(f"  Mean IQR: {iqr_mean:.5f}")

    print(f"\n  VERDICT: {verdict}")
    print("=" * w)

    save_results("arctan_pinball", {
        "ce_baseline": m_ce, "ce_train_time": ce_time,
        "arctan_pinball": m_q, "q_train_loss": q_loss, "q_train_time": q_time,
        "tau_levels": TAU_LEVELS, "delta": DELTA,
        "quantile_spread": {
            "q25_mean": float(q25_mean), "q50_mean": float(q50_mean),
            "q75_mean": float(q75_mean), "iqr_mean": float(iqr_mean),
        },
        "verdict": verdict,
    })


if __name__ == "__main__":
    main()
