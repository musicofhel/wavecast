#!/usr/bin/env python3
"""Bernstein Quantile Network + CRPS Loss.

Outputs K Bernstein polynomial coefficients defining a monotone quantile function.
Non-crossing by construction (cumulative softplus deltas). Trained with CRPS loss
(pinball loss approximation over uniform tau grid).

Direction signal: sign of median quantile Q(0.5). Mapped to 5-class labels for
evaluate_5_metrics compatibility.

Usage:
    python -m scripts.feature_tests.test_bqn_crps
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

EMBED_DIM = MODEL_KWARGS["embed_dim"]
NET_KWARGS = {k: MODEL_KWARGS[k] for k in ("embed_dim", "num_heads", "num_layers", "dropout")}


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class BQNHead(nn.Module):
    """Bernstein Quantile Network head.

    Projects hidden state to K coefficients, then applies softplus + cumsum
    to produce a monotonically increasing quantile function.
    """

    def __init__(self, embed_dim: int, n_coeffs: int = 12) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, n_coeffs)
        self.n_coeffs = n_coeffs

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raw = self.proj(h)  # (batch, K)
        deltas = torch.nn.functional.softplus(raw)  # positive increments
        quantiles = torch.cumsum(deltas, dim=1) - deltas.sum(dim=1, keepdim=True) / 2
        return quantiles  # (batch, K)


class CRPSLoss(nn.Module):
    """CRPS approximation via pinball loss at uniform tau levels."""

    def __init__(self, n_tau: int = 50) -> None:
        super().__init__()
        self.register_buffer("tau", torch.linspace(0.01, 0.99, n_tau))

    def forward(self, quantiles: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        K = quantiles.size(1)
        loss = torch.zeros(1, device=quantiles.device)
        for t in self.tau:
            idx = min(int(t.item() * K), K - 1)
            q = quantiles[:, idx]
            err = targets - q
            loss = loss + torch.where(err > 0, t * err, (t - 1) * err).mean()
        return loss / len(self.tau)


class BQNModel(nn.Module):
    """TransformerEncoder backbone (matching WaveletGPTNet) + BQN head."""

    def __init__(self, embed_dim: int, num_heads: int, num_layers: int,
                 dropout: float, n_coeffs: int = 12) -> None:
        super().__init__()
        self.context_length = CONTEXT_LENGTH
        self.n_aux_features = N_AUX_FEATURES

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
        self.head = BQNHead(embed_dim, n_coeffs)

    def forward(self, ctx: torch.Tensor, lvl: torch.Tensor,
                ac: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
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
        last_hidden = x[:, -1, :]
        return self.head(last_hidden)  # (batch, K)


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


def quantiles_to_labels(
    quantiles: np.ndarray, boundaries: np.ndarray,
) -> np.ndarray:
    """Map median quantile value to 5-class labels via return boundaries."""
    K = quantiles.shape[1]
    median_idx = K // 2
    median_vals = quantiles[:, median_idx]
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
    logger.info("BQN + CRPS experiment")

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

    # --- BQN + CRPS ---
    logger.info("Training BQN + CRPS...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bqn_model = BQNModel(**NET_KWARGS, n_coeffs=12).to(device)
    criterion = CRPSLoss(n_tau=50).to(device)
    optimizer = torch.optim.AdamW(bqn_model.parameters(), lr=MODEL_KWARGS["learning_rate"])

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
        bqn_model.train()
        total, n = 0.0, 0
        for batch in loader:
            ctx_b = batch[0].to(device)
            lvl_b = batch[1].to(device)
            ac_b = batch[2].to(device)
            aux_b = batch[3].to(device)
            tgt = batch[4].to(device)
            optimizer.zero_grad()
            quantiles = bqn_model(ctx_b, lvl_b, ac_b, aux_b)
            loss = criterion(quantiles, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bqn_model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            n += 1
        avg_loss = total / max(n, 1)
        if (epoch + 1) % 5 == 0:
            logger.info("  Epoch %d/%d  loss=%.6f", epoch + 1, MODEL_KWARGS["epochs"], avg_loss)
    bqn_time = time.time() - t0

    # --- Predict ---
    bqn_model.eval()
    ctx_te, lvl_te, ac_te, aux_te = parse_x(X_test)
    with torch.no_grad():
        q_test = bqn_model(
            torch.tensor(ctx_te, dtype=torch.float32).to(device),
            torch.tensor(lvl_te, dtype=torch.long).to(device),
            torch.tensor(ac_te, dtype=torch.long).to(device),
            torch.tensor(aux_te, dtype=torch.float32).to(device),
        ).cpu().numpy()

    pred_bqn = quantiles_to_labels(q_test, ret_boundaries)
    m_bqn = evaluate_5_metrics(pred_bqn, te_rets, y_test, te_valid)
    print_5_metrics("BQN + CRPS", m_bqn)

    # --- Comparison ---
    print("\n" + "=" * 100)
    print("COMPARISON: BQN+CRPS vs CE Baseline")
    print("=" * 100)
    for key in ("econ_dir", "transition_acc", "large_move_acc", "sharpe_costs"):
        delta = m_bqn[key] - m_ce[key]
        fmt = ".1%" if key != "sharpe_costs" else "+.3f"
        print(f"  {key:<22} CE={m_ce[key]:{fmt}}  BQN={m_bqn[key]:{fmt}}  delta={delta:+.4f}")
    print(f"  BQN train loss: {avg_loss:.6f}  time: {bqn_time:.1f}s")
    print(f"  CE  train time: {ce_time:.1f}s")
    print("=" * 100)

    save_results("bqn_crps", {
        "ce_baseline": m_ce,
        "bqn_crps": m_bqn,
        "ce_train_time": ce_time,
        "bqn_train_time": bqn_time,
        "bqn_train_loss": avg_loss,
        "n_coeffs": 12,
        "n_tau": 50,
    })


if __name__ == "__main__":
    main()
