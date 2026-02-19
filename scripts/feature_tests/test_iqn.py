#!/usr/bin/env python3
"""Implicit Quantile Networks (IQN).

Full continuous quantile function: takes tau in [0,1] as input and outputs Q(tau).
Uses cosine embedding of tau levels combined with hidden states via element-wise
multiplication. Trained with quantile regression (pinball) loss on random tau
samples per batch.

At inference: evaluate at tau={0.1, 0.25, 0.5, 0.75, 0.9}. Direction from
median (tau=0.5) sign. Calibration check: empirical coverage at 50% interval
[Q(0.25), Q(0.75)].

Usage:
    python -m scripts.feature_tests.test_iqn
"""

from __future__ import annotations  # noqa: I001

import logging
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

NET_KWARGS = {k: MODEL_KWARGS[k] for k in ("embed_dim", "num_heads", "num_layers", "dropout")}
EVAL_TAUS = [0.1, 0.25, 0.5, 0.75, 0.9]
N_TAU_TRAIN = 8


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class IQNHead(nn.Module):
    """Implicit Quantile Network head with cosine tau embedding.

    Takes hidden state h and quantile levels tau, produces Q(tau) for each sample.
    """

    def __init__(self, embed_dim: int, n_cos: int = 64) -> None:
        super().__init__()
        self.n_cos = n_cos
        self.cos_embed = nn.Linear(n_cos, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, h: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            h: (batch, embed_dim) hidden states from backbone.
            tau: (batch, n_tau) quantile levels in [0, 1].

        Returns:
            (batch, n_tau) predicted quantile values.
        """
        batch_size, n_tau = tau.shape
        # Cosine embedding: cos(pi * i * tau) for i in 1..n_cos
        i_vals = torch.arange(1, self.n_cos + 1, device=tau.device, dtype=torch.float32)
        # (batch, n_tau, 1) * (n_cos,) -> (batch, n_tau, n_cos)
        i_pi_tau = np.pi * tau.unsqueeze(-1) * i_vals.unsqueeze(0).unsqueeze(0)
        cos_feat = torch.cos(i_pi_tau)  # (batch, n_tau, n_cos)
        tau_embed = torch.relu(self.cos_embed(cos_feat))  # (batch, n_tau, embed_dim)
        # Combine with hidden state via element-wise multiplication
        h_exp = h.unsqueeze(1).expand(-1, n_tau, -1)  # (batch, n_tau, embed_dim)
        combined = h_exp * tau_embed
        return self.fc(combined).squeeze(-1)  # (batch, n_tau)


class QuantileRegressionLoss(nn.Module):
    """Pinball loss for quantile regression."""

    def forward(self, predicted: torch.Tensor, targets: torch.Tensor,
                tau: torch.Tensor) -> torch.Tensor:
        """Compute quantile regression loss.

        Args:
            predicted: (batch, n_tau) predicted quantile values.
            targets: (batch,) actual values.
            tau: (batch, n_tau) quantile levels.

        Returns:
            Scalar loss.
        """
        err = targets.unsqueeze(1) - predicted  # (batch, n_tau)
        loss = torch.where(err > 0, tau * err, (tau - 1) * err)
        return loss.mean()


class IQNModel(nn.Module):
    """TransformerEncoder backbone (matching WaveletGPTNet) + IQN head."""

    def __init__(self, embed_dim: int, num_heads: int, num_layers: int,
                 dropout: float, n_cos: int = 64) -> None:
        super().__init__()
        self.context_length = CONTEXT_LENGTH
        self.n_aux_features = N_AUX_FEATURES
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(1, embed_dim)
        self.pos_embed = nn.Embedding(CONTEXT_LENGTH, embed_dim)
        self.level_embed = nn.Embedding(6, embed_dim)
        self.asset_class_embed = nn.Embedding(7, embed_dim)
        if N_AUX_FEATURES > 0:
            self.aux_proj = nn.Linear(N_AUX_FEATURES, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.iqn_head = IQNHead(embed_dim, n_cos)

    def encode(self, ctx: torch.Tensor, lvl: torch.Tensor,
               ac: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        """Run backbone, return last hidden state (batch, embed_dim)."""
        batch_size, seq_len = ctx.shape
        device = ctx.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.input_proj(ctx.unsqueeze(-1)) + self.pos_embed(positions)
        if self.n_aux_features > 0 and aux is not None:
            x = x + self.aux_proj(aux)
        x = x + self.level_embed(lvl).unsqueeze(1)
        x = x + self.asset_class_embed(ac).unsqueeze(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1,
        )
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.ln_f(x)
        return x[:, -1, :]  # (batch, embed_dim)

    def forward(self, ctx: torch.Tensor, lvl: torch.Tensor,
                ac: torch.Tensor, tau: torch.Tensor,
                aux: torch.Tensor | None = None) -> torch.Tensor:
        """Full forward: encode then predict quantiles at given tau levels.

        Returns:
            (batch, n_tau) predicted quantile values.
        """
        h = self.encode(ctx, lvl, ac, aux)
        return self.iqn_head(h, tau)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_x(
    X: np.ndarray, ctx_len: int = CONTEXT_LENGTH, n_aux: int = N_AUX_FEATURES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse X array into components for the backbone."""
    ctx = X[:, :ctx_len].astype(np.float32)
    aux_end = ctx_len + ctx_len * n_aux
    aux = X[:, ctx_len:aux_end].astype(np.float32).reshape(-1, ctx_len, n_aux)
    lvl = X[:, aux_end].astype(np.int64)
    ac = X[:, aux_end + 1].astype(np.int64)
    return ctx, lvl, ac, aux


def median_to_labels(
    median_vals: np.ndarray, boundaries: np.ndarray,
) -> np.ndarray:
    """Map median quantile values to 5-class labels via return boundaries."""
    labels = np.full(len(median_vals), MID, dtype=np.int64)
    for i, val in enumerate(median_vals):
        if val < boundaries[0]:
            labels[i] = 0
        elif val < boundaries[1]:
            labels[i] = 1
        elif val > boundaries[3]:
            labels[i] = 4
        elif val > boundaries[2]:
            labels[i] = 3
    return labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("IQN experiment")

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

    # Boundary values for mapping quantile predictions back to class labels
    valid_rets = tr_rets[tr_valid & ~np.isnan(tr_rets)]
    ret_boundaries = np.percentile(valid_rets, PERCENTILES)

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

    # --- IQN ---
    logger.info("Training IQN (n_tau=%d per sample)...", N_TAU_TRAIN)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    iqn_model = IQNModel(**NET_KWARGS, n_cos=64).to(device)
    criterion = QuantileRegressionLoss()
    optimizer = torch.optim.AdamW(iqn_model.parameters(), lr=MODEL_KWARGS["learning_rate"])

    ctx_tr, lvl_tr, ac_tr, aux_tr = parse_x(X_train)
    y_rets_train = tr_rets.astype(np.float32)

    ds = TensorDataset(
        torch.tensor(ctx_tr, dtype=torch.float32),
        torch.tensor(lvl_tr, dtype=torch.long),
        torch.tensor(ac_tr, dtype=torch.long),
        torch.tensor(aux_tr, dtype=torch.float32),
        torch.tensor(y_rets_train, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=MODEL_KWARGS["batch_size"], shuffle=True)

    t0 = time.time()
    avg_loss = 0.0
    for epoch in range(MODEL_KWARGS["epochs"]):
        iqn_model.train()
        total, n = 0.0, 0
        for batch in loader:
            ctx_b = batch[0].to(device)
            lvl_b = batch[1].to(device)
            ac_b = batch[2].to(device)
            aux_b = batch[3].to(device)
            tgt = batch[4].to(device)
            bs = ctx_b.size(0)

            # Sample random tau levels for this batch
            tau = torch.rand(bs, N_TAU_TRAIN, device=device)

            optimizer.zero_grad()
            predicted = iqn_model(ctx_b, lvl_b, ac_b, tau, aux_b)
            loss = criterion(predicted, tgt, tau)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(iqn_model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            n += 1
        avg_loss = total / max(n, 1)
        if (epoch + 1) % 5 == 0:
            logger.info("  Epoch %d/%d  loss=%.6f", epoch + 1, MODEL_KWARGS["epochs"], avg_loss)
    iqn_time = time.time() - t0

    # --- Predict at fixed tau levels ---
    logger.info("Evaluating at tau=%s", EVAL_TAUS)
    iqn_model.eval()
    ctx_te, lvl_te, ac_te, aux_te = parse_x(X_test)
    n_test = len(ctx_te)
    tau_eval = torch.tensor([EVAL_TAUS] * n_test, dtype=torch.float32, device=device)

    with torch.no_grad():
        q_test = iqn_model(
            torch.tensor(ctx_te, dtype=torch.float32).to(device),
            torch.tensor(lvl_te, dtype=torch.long).to(device),
            torch.tensor(ac_te, dtype=torch.long).to(device),
            tau_eval,
            torch.tensor(aux_te, dtype=torch.float32).to(device),
        ).cpu().numpy()  # (n_test, len(EVAL_TAUS))

    # q_test columns: [Q(0.1), Q(0.25), Q(0.5), Q(0.75), Q(0.9)]
    median_vals = q_test[:, 2]  # Q(0.5)
    pred_iqn = median_to_labels(median_vals, ret_boundaries)
    m_iqn = evaluate_5_metrics(pred_iqn, te_rets, y_test, te_valid)
    print_5_metrics("IQN", m_iqn)

    # --- Calibration: 50% interval coverage ---
    q25 = q_test[:, 1]  # Q(0.25)
    q75 = q_test[:, 3]  # Q(0.75)
    valid_idx = te_valid & ~np.isnan(te_rets)
    if valid_idx.sum() > 0:
        inside = (te_rets[valid_idx] >= q25[valid_idx]) & (te_rets[valid_idx] <= q75[valid_idx])
        empirical_coverage = float(inside.mean())
    else:
        empirical_coverage = 0.0

    # 80% interval coverage: [Q(0.1), Q(0.9)]
    q10 = q_test[:, 0]
    q90 = q_test[:, 4]
    if valid_idx.sum() > 0:
        inside_80 = (te_rets[valid_idx] >= q10[valid_idx]) & (te_rets[valid_idx] <= q90[valid_idx])
        coverage_80 = float(inside_80.mean())
    else:
        coverage_80 = 0.0

    # Interval width statistics
    interval_50_width = float(np.mean(q75[valid_idx] - q25[valid_idx])) if valid_idx.sum() > 0 else 0.0
    interval_80_width = float(np.mean(q90[valid_idx] - q10[valid_idx])) if valid_idx.sum() > 0 else 0.0

    print("\n" + "=" * 100)
    print("IQN CALIBRATION CHECK")
    print("=" * 100)
    print(f"  50% interval [Q(0.25), Q(0.75)] coverage: {empirical_coverage:.1%} (ideal: 50%)")
    print(f"  80% interval [Q(0.10), Q(0.90)] coverage: {coverage_80:.1%} (ideal: 80%)")
    print(f"  50% interval mean width: {interval_50_width:.6f}")
    print(f"  80% interval mean width: {interval_80_width:.6f}")
    well_calibrated_50 = 0.35 <= empirical_coverage <= 0.65
    well_calibrated_80 = 0.65 <= coverage_80 <= 0.95
    if well_calibrated_50 and well_calibrated_80:
        print("  CALIBRATED: both intervals within tolerance")
    elif well_calibrated_50:
        print("  PARTIAL: 50% interval ok, 80% interval miscalibrated")
    elif well_calibrated_80:
        print("  PARTIAL: 80% interval ok, 50% interval miscalibrated")
    else:
        print("  MISCALIBRATED: both intervals outside tolerance")
    print("=" * 100)

    # --- Comparison ---
    print("\n" + "=" * 100)
    print("COMPARISON: IQN vs CE Baseline")
    print("=" * 100)
    for key in ("econ_dir", "transition_acc", "large_move_acc", "sharpe_costs"):
        delta = m_iqn[key] - m_ce[key]
        fmt = ".1%" if key != "sharpe_costs" else "+.3f"
        print(f"  {key:<22} CE={m_ce[key]:{fmt}}  IQN={m_iqn[key]:{fmt}}  delta={delta:+.4f}")
    print(f"  IQN train loss: {avg_loss:.6f}  time: {iqn_time:.1f}s")
    print(f"  CE  train time: {ce_time:.1f}s")
    print("=" * 100)

    save_results("iqn", {
        "ce_baseline": m_ce,
        "iqn": m_iqn,
        "ce_train_time": ce_time,
        "iqn_train_time": iqn_time,
        "iqn_train_loss": avg_loss,
        "calibration": {
            "coverage_50": empirical_coverage,
            "coverage_80": coverage_80,
            "width_50": interval_50_width,
            "width_80": interval_80_width,
            "well_calibrated_50": well_calibrated_50,
            "well_calibrated_80": well_calibrated_80,
        },
        "n_cos": 64,
        "n_tau_train": N_TAU_TRAIN,
        "eval_taus": EVAL_TAUS,
    })


if __name__ == "__main__":
    main()
