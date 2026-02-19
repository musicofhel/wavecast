#!/usr/bin/env python3
"""Neural ordinal regression via N3POM (cumulative probabilities).

Proportional-odds model: shared latent scalar f(x) with learned ordered cutpoints.
P(Y <= k) = sigmoid(cutpoint_k - f(x)). Monotonicity guaranteed by shared f +
ordered cutpoints enforced via cumulative softmax.

Key check: verify all 5 classes are predicted >10% (no class collapse like Round 1
ordinal loss exhibited).

Usage:
    python -m scripts.feature_tests.test_n3pom
"""

from __future__ import annotations  # noqa: I001

import logging
import time

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
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.targets.returns import assign_quantile_labels, compute_quantile_boundaries

logger = logging.getLogger(__name__)

NET_KWARGS = {k: MODEL_KWARGS[k] for k in ("embed_dim", "num_heads", "num_layers", "dropout")}


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class N3POMHead(nn.Module):
    """Neural proportional-odds ordinal head.

    Shared scalar projection f(h), combined with K-1 ordered cutpoints
    to produce cumulative probabilities, then differenced to class probabilities.
    """

    def __init__(self, embed_dim: int, n_classes: int = 5) -> None:
        super().__init__()
        self.n_classes = n_classes
        # Cutpoints initialized as evenly spaced; ordering enforced via cumulative softplus
        self.raw_cutpoints = nn.Parameter(torch.zeros(n_classes - 1))
        self.proj = nn.Linear(embed_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        f = self.proj(h)  # (batch, 1)
        # Enforce monotonic cutpoints: base + cumulative softplus deltas
        deltas = torch.nn.functional.softplus(self.raw_cutpoints)
        cutpoints = torch.cumsum(deltas, dim=0) - deltas.sum() / 2
        # Cumulative P(Y <= k) = sigmoid(cutpoint_k - f)
        cum_probs = torch.sigmoid(cutpoints.unsqueeze(0) - f)  # (batch, K-1)
        # Class probabilities via differencing
        probs = torch.zeros(h.size(0), self.n_classes, device=h.device)
        probs[:, 0] = cum_probs[:, 0]
        for k in range(1, self.n_classes - 1):
            probs[:, k] = cum_probs[:, k] - cum_probs[:, k - 1]
        probs[:, -1] = 1.0 - cum_probs[:, -1]
        return torch.clamp(probs, min=1e-7)


class N3POMLoss(nn.Module):
    """Negative log-likelihood on ordinal class probabilities."""

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = probs.size(0)
        selected = probs[torch.arange(batch_size, device=probs.device), targets]
        return -torch.log(selected).mean()


class N3POMModel(nn.Module):
    """TransformerEncoder backbone (matching WaveletGPTNet) + N3POM head."""

    def __init__(self, embed_dim: int, num_heads: int, num_layers: int,
                 dropout: float, n_classes: int = 5) -> None:
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
        self.head = N3POMHead(embed_dim, n_classes)

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
        return self.head(last_hidden)  # (batch, n_classes) — probabilities


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("N3POM ordinal regression experiment")

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

    # --- N3POM ---
    logger.info("Training N3POM ordinal regression...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n3pom = N3POMModel(**NET_KWARGS, n_classes=N_CLASSES).to(device)
    criterion = N3POMLoss()
    optimizer = torch.optim.AdamW(n3pom.parameters(), lr=MODEL_KWARGS["learning_rate"])

    ctx_tr, lvl_tr, ac_tr, aux_tr = parse_x(X_train)

    ds = TensorDataset(
        torch.tensor(ctx_tr, dtype=torch.float32),
        torch.tensor(lvl_tr, dtype=torch.long),
        torch.tensor(ac_tr, dtype=torch.long),
        torch.tensor(aux_tr, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=MODEL_KWARGS["batch_size"], shuffle=True)

    t0 = time.time()
    avg_loss = 0.0
    for epoch in range(MODEL_KWARGS["epochs"]):
        n3pom.train()
        total, n = 0.0, 0
        for batch in loader:
            ctx_b = batch[0].to(device)
            lvl_b = batch[1].to(device)
            ac_b = batch[2].to(device)
            aux_b = batch[3].to(device)
            tgt = batch[4].to(device)
            optimizer.zero_grad()
            probs = n3pom(ctx_b, lvl_b, ac_b, aux_b)
            loss = criterion(probs, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(n3pom.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            n += 1
        avg_loss = total / max(n, 1)
        if (epoch + 1) % 5 == 0:
            logger.info("  Epoch %d/%d  loss=%.6f", epoch + 1, MODEL_KWARGS["epochs"], avg_loss)
    n3pom_time = time.time() - t0

    # --- Predict ---
    n3pom.eval()
    ctx_te, lvl_te, ac_te, aux_te = parse_x(X_test)
    with torch.no_grad():
        probs_test = n3pom(
            torch.tensor(ctx_te, dtype=torch.float32).to(device),
            torch.tensor(lvl_te, dtype=torch.long).to(device),
            torch.tensor(ac_te, dtype=torch.long).to(device),
            torch.tensor(aux_te, dtype=torch.float32).to(device),
        ).cpu().numpy()

    pred_n3pom = probs_test.argmax(axis=1).astype(np.int64)
    m_n3pom = evaluate_5_metrics(pred_n3pom, te_rets, y_test, te_valid)
    print_5_metrics("N3POM Ordinal", m_n3pom)

    # --- Class collapse check ---
    valid_preds = pred_n3pom[te_valid]
    class_pcts = {}
    for c in range(N_CLASSES):
        class_pcts[c] = float((valid_preds == c).mean()) if len(valid_preds) > 0 else 0.0

    all_above_10 = all(v > 0.10 for v in class_pcts.values())
    any_below_5 = any(v < 0.05 for v in class_pcts.values())

    print("\n" + "=" * 100)
    print("N3POM CLASS COLLAPSE CHECK")
    print("=" * 100)
    for c, pct in sorted(class_pcts.items()):
        flag = " <-- COLLAPSED" if pct < 0.05 else ""
        print(f"  Class {c}: {pct:.1%}{flag}")
    if all_above_10:
        print("  BALANCED: all classes > 10% (no collapse)")
    elif any_below_5:
        print("  COLLAPSED: at least one class < 5%")
    else:
        print("  MARGINAL: some classes between 5-10%")
    print("=" * 100)

    # --- Comparison ---
    print("\n" + "=" * 100)
    print("COMPARISON: N3POM vs CE Baseline")
    print("=" * 100)
    for key in ("econ_dir", "transition_acc", "large_move_acc", "sharpe_costs"):
        delta = m_n3pom[key] - m_ce[key]
        fmt = ".1%" if key != "sharpe_costs" else "+.3f"
        print(f"  {key:<22} CE={m_ce[key]:{fmt}}  N3POM={m_n3pom[key]:{fmt}}  delta={delta:+.4f}")
    print(f"  N3POM train loss: {avg_loss:.6f}  time: {n3pom_time:.1f}s")
    print(f"  CE    train time: {ce_time:.1f}s")
    print("=" * 100)

    save_results("n3pom", {
        "ce_baseline": m_ce,
        "n3pom": m_n3pom,
        "ce_train_time": ce_time,
        "n3pom_train_time": n3pom_time,
        "n3pom_train_loss": avg_loss,
        "class_pcts": {str(k): v for k, v in class_pcts.items()},
        "all_above_10pct": all_above_10,
        "any_below_5pct": any_below_5,
    })


if __name__ == "__main__":
    main()
