#!/usr/bin/env python3
"""Temporal Neighborhood Coding (TNC) — contrastive pre-training experiment.

Windows from the same market regime are positive pairs; windows from different
regimes are negatives.  Phase 1 learns representations via the TNC contrastive
loss, then Phase 2 freezes the encoder and trains a linear classification head
with cross-entropy.

Regime proxy: sign of the next-step return.  Same direction within a sliding
window of W=32 positions from the same ticker/level is treated as a positive
neighbor; opposite direction or distant samples are negatives.

Usage:
    python -m scripts.feature_tests.test_tnc
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
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.targets.returns import assign_quantile_labels, compute_quantile_boundaries

logger = logging.getLogger(__name__)

EMBED_DIM = MODEL_KWARGS["embed_dim"]
NET_KWARGS = {k: MODEL_KWARGS[k] for k in ("embed_dim", "num_heads", "num_layers", "dropout")}
REPR_DIM = 32
NEIGHBORHOOD_W = 32
TNC_EPOCHS = 30
FINETUNE_EPOCHS = 20
TEMPERATURE = 0.5


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class TemporalEncoder(nn.Module):
    """Transformer backbone outputting a representation vector instead of logits.

    Architecture mirrors WaveletGPTNet (continuous input mode) up to the last
    hidden state, then projects to ``repr_dim``.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        context_length: int = CONTEXT_LENGTH,
        n_aux_features: int = N_AUX_FEATURES,
        repr_dim: int = REPR_DIM,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.n_aux_features = n_aux_features
        self.repr_dim = repr_dim

        # Input projection (continuous mode)
        self.input_proj = nn.Linear(1, embed_dim)
        if n_aux_features > 0:
            self.aux_proj = nn.Linear(n_aux_features, embed_dim)
        self.pos_embed = nn.Embedding(context_length, embed_dim)
        self.level_embed = nn.Embedding(6, embed_dim)
        self.asset_class_embed = nn.Embedding(7, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.repr_head = nn.Linear(embed_dim, repr_dim)

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

        mask = torch.triu(
            torch.ones(s, s, device=ctx.device, dtype=torch.bool), diagonal=1,
        )
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.ln_f(x)

        last_hidden = x[:, -1, :]  # (B, embed_dim)
        return self.repr_head(last_hidden)  # (B, repr_dim)


# ---------------------------------------------------------------------------
# TNC Contrastive Loss
# ---------------------------------------------------------------------------


class TNCLoss(nn.Module):
    """Temporal Neighborhood Coding contrastive objective.

    Pulls anchor and positive (same regime neighbor) representations together
    while pushing anchor and negative (different regime) apart.
    """

    def forward(
        self,
        z_anchor: torch.Tensor,
        z_pos: torch.Tensor,
        z_neg: torch.Tensor,
    ) -> torch.Tensor:
        # Dot-product scores
        pos_score = torch.sum(z_anchor * z_pos, dim=1)
        neg_score = torch.sum(z_anchor * z_neg, dim=1)
        loss = -torch.log(torch.sigmoid(pos_score) + 1e-7).mean()
        loss = loss + (-torch.log(1.0 - torch.sigmoid(neg_score) + 1e-7)).mean()
        return loss


# ---------------------------------------------------------------------------
# Data parsing helper
# ---------------------------------------------------------------------------


def parse_x(
    X: np.ndarray,
    ctx_len: int = CONTEXT_LENGTH,
    n_aux: int = N_AUX_FEATURES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse flat X into (ctx, lvl, ac, aux) numpy arrays."""
    ctx = X[:, :ctx_len].astype(np.float32)
    aux_end = ctx_len + ctx_len * n_aux
    aux = X[:, ctx_len:aux_end].astype(np.float32).reshape(-1, ctx_len, n_aux)
    lvl = X[:, aux_end].astype(np.int64)
    ac = X[:, aux_end + 1].astype(np.int64)
    return ctx, lvl, ac, aux


# ---------------------------------------------------------------------------
# Triplet mining
# ---------------------------------------------------------------------------


def mine_triplets(
    returns: np.ndarray,
    valid: np.ndarray,
    n_samples: int,
    W: int = NEIGHBORHOOD_W,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mine (anchor, positive, negative) index triplets from return directions.

    Positive: a sample within W positions of the anchor with the same return
    sign (same regime proxy).  Negative: a sample with opposite return sign or
    more than W positions away.

    Returns three arrays of shape (n_triplets,) with valid indices into X.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    directions = np.sign(returns)
    valid_idx = np.where(valid & ~np.isnan(returns) & (directions != 0))[0]

    if len(valid_idx) < 10:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    pos_idx = valid_idx[directions[valid_idx] > 0]
    neg_idx = valid_idx[directions[valid_idx] < 0]

    anchors, positives, negatives = [], [], []

    for _ in range(n_samples):
        # Pick a random anchor
        a = rng.choice(valid_idx)
        a_dir = directions[a]

        # Positive: same direction, within W positions
        same_dir = pos_idx if a_dir > 0 else neg_idx
        nearby = same_dir[(same_dir >= a - W) & (same_dir <= a + W) & (same_dir != a)]
        if len(nearby) == 0:
            # Relax: any same-direction sample
            nearby = same_dir[same_dir != a]
        if len(nearby) == 0:
            continue
        p = rng.choice(nearby)

        # Negative: opposite direction
        opp_dir = neg_idx if a_dir > 0 else pos_idx
        if len(opp_dir) == 0:
            continue
        n = rng.choice(opp_dir)

        anchors.append(a)
        positives.append(p)
        negatives.append(n)

    return (
        np.array(anchors, dtype=np.int64),
        np.array(positives, dtype=np.int64),
        np.array(negatives, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------


def train_tnc_phase1(
    encoder: TemporalEncoder,
    X_train: np.ndarray,
    returns: np.ndarray,
    valid: np.ndarray,
    device: torch.device,
) -> float:
    """Phase 1: TNC contrastive pre-training. Returns final epoch loss."""
    ctx, lvl, ac, aux = parse_x(X_train)
    criterion = TNCLoss()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=MODEL_KWARGS["learning_rate"])
    batch_size = MODEL_KWARGS["batch_size"]
    rng = np.random.default_rng(42)

    avg_loss = 0.0
    for epoch in range(TNC_EPOCHS):
        encoder.train()

        # Re-mine triplets each epoch with different randomness
        a_idx, p_idx, n_idx = mine_triplets(
            returns, valid, n_samples=len(X_train), W=NEIGHBORHOOD_W, rng=rng,
        )
        if len(a_idx) < batch_size:
            logger.warning("Too few triplets mined (%d), skipping epoch", len(a_idx))
            continue

        # Build epoch tensors
        ctx_a = torch.tensor(ctx[a_idx], dtype=torch.float32)
        lvl_a = torch.tensor(lvl[a_idx], dtype=torch.long)
        ac_a = torch.tensor(ac[a_idx], dtype=torch.long)
        aux_a = torch.tensor(aux[a_idx], dtype=torch.float32)

        ctx_p = torch.tensor(ctx[p_idx], dtype=torch.float32)
        lvl_p = torch.tensor(lvl[p_idx], dtype=torch.long)
        ac_p = torch.tensor(ac[p_idx], dtype=torch.long)
        aux_p = torch.tensor(aux[p_idx], dtype=torch.float32)

        ctx_n = torch.tensor(ctx[n_idx], dtype=torch.float32)
        lvl_n = torch.tensor(lvl[n_idx], dtype=torch.long)
        ac_n = torch.tensor(ac[n_idx], dtype=torch.long)
        aux_n = torch.tensor(aux[n_idx], dtype=torch.float32)

        dataset = TensorDataset(
            ctx_a, lvl_a, ac_a, aux_a,
            ctx_p, lvl_p, ac_p, aux_p,
            ctx_n, lvl_n, ac_n, aux_n,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total, n_batches = 0.0, 0
        for batch in loader:
            (ca, la, aa, xa, cp, lp, ap, xp, cn, ln, an, xn) = [t.to(device) for t in batch]

            xa_3d = xa.reshape(-1, CONTEXT_LENGTH, N_AUX_FEATURES)
            xp_3d = xp.reshape(-1, CONTEXT_LENGTH, N_AUX_FEATURES)
            xn_3d = xn.reshape(-1, CONTEXT_LENGTH, N_AUX_FEATURES)

            optimizer.zero_grad()
            z_a = encoder(ca, la, aa, aux=xa_3d)
            z_p = encoder(cp, lp, ap, aux=xp_3d)
            z_n = encoder(cn, ln, an, aux=xn_3d)

            loss = criterion(z_a, z_p, z_n)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            total += loss.item()
            n_batches += 1

        avg_loss = total / max(n_batches, 1)
        if (epoch + 1) % 10 == 0:
            logger.info("  TNC Phase 1 epoch %d/%d  loss=%.4f", epoch + 1, TNC_EPOCHS, avg_loss)

    return avg_loss


def train_tnc_phase2(
    encoder: TemporalEncoder,
    cls_head: nn.Linear,
    X_train: np.ndarray,
    y_train: np.ndarray,
    valid: np.ndarray,
    class_weights: list[float],
    device: torch.device,
) -> float:
    """Phase 2: freeze encoder, train linear classification head with CE."""
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    ctx, lvl, ac, aux = parse_x(X_train)

    # Filter to valid samples
    keep = valid & ~np.isnan(y_train.astype(np.float64))
    ctx_k, lvl_k, ac_k, aux_k = ctx[keep], lvl[keep], ac[keep], aux[keep]
    y_k = y_train[keep].astype(np.int64)

    dataset = TensorDataset(
        torch.tensor(ctx_k, dtype=torch.float32),
        torch.tensor(lvl_k, dtype=torch.long),
        torch.tensor(ac_k, dtype=torch.long),
        torch.tensor(aux_k.reshape(len(aux_k), -1), dtype=torch.float32),
        torch.tensor(y_k, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=MODEL_KWARGS["batch_size"], shuffle=True)

    cw_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)
    optimizer = torch.optim.AdamW(cls_head.parameters(), lr=MODEL_KWARGS["learning_rate"])

    avg_loss = 0.0
    for epoch in range(FINETUNE_EPOCHS):
        cls_head.train()
        total, n_b = 0.0, 0
        for batch in loader:
            ctx_b = batch[0].to(device)
            lvl_b = batch[1].to(device)
            ac_b = batch[2].to(device)
            aux_b = batch[3].to(device).reshape(-1, CONTEXT_LENGTH, N_AUX_FEATURES)
            tgt = batch[4].to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                z = encoder(ctx_b, lvl_b, ac_b, aux=aux_b)
            logits = cls_head(z)
            loss = criterion(logits, tgt)
            loss.backward()
            optimizer.step()

            total += loss.item()
            n_b += 1
        avg_loss = total / max(n_b, 1)
        if (epoch + 1) % 5 == 0:
            logger.info("  TNC Phase 2 epoch %d/%d  loss=%.4f", epoch + 1, FINETUNE_EPOCHS, avg_loss)

    return avg_loss


def predict_tnc(
    encoder: TemporalEncoder,
    cls_head: nn.Linear,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Get argmax class predictions from encoder + classification head."""
    ctx, lvl, ac, aux = parse_x(X)
    encoder.eval()
    cls_head.eval()
    with torch.no_grad():
        z = encoder(
            torch.tensor(ctx, dtype=torch.float32).to(device),
            torch.tensor(lvl, dtype=torch.long).to(device),
            torch.tensor(ac, dtype=torch.long).to(device),
            aux=torch.tensor(aux, dtype=torch.float32).to(device),
        )
        logits = cls_head(z)
    return logits.argmax(dim=-1).cpu().numpy()


# ---------------------------------------------------------------------------
# CE baseline helper
# ---------------------------------------------------------------------------


def train_ce_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weights: list[float],
) -> tuple[WaveletGPT, float, float]:
    """Train standard CE baseline. Returns (model, train_loss, train_time)."""
    model = WaveletGPT(
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
    metrics = model.fit(X_train, y_train.astype(np.float64))
    elapsed = time.time() - t0
    return model, metrics["train_loss"], elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("TNC experiment: contrastive pre-training with temporal neighborhood coding")

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

    # Class weights
    valid_labels = y_train[tr_valid]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- CE Baseline ---
    logger.info("Training CE baseline...")
    ce_model, ce_loss, ce_time = train_ce_baseline(X_train, y_train, class_weights)
    pred_ce = ce_model.predict(X_test).astype(np.int64)
    m_ce = evaluate_5_metrics(pred_ce, te_rets, y_test, te_valid)
    print_5_metrics("CE Baseline", m_ce)

    # --- TNC ---
    logger.info("Training TNC encoder (Phase 1: contrastive, %d epochs)...", TNC_EPOCHS)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    encoder = TemporalEncoder(repr_dim=REPR_DIM, **NET_KWARGS).to(device)

    t0 = time.time()
    tnc_loss = train_tnc_phase1(encoder, X_train, tr_rets, tr_valid, device)
    phase1_time = time.time() - t0
    logger.info("TNC Phase 1 done in %.1fs (final loss=%.4f)", phase1_time, tnc_loss)

    # Phase 2: linear probe
    logger.info("Training TNC classifier (Phase 2: linear probe, %d epochs)...", FINETUNE_EPOCHS)
    cls_head = nn.Linear(REPR_DIM, N_CLASSES).to(device)

    t0 = time.time()
    ft_loss = train_tnc_phase2(encoder, cls_head, X_train, y_train, tr_valid, class_weights, device)
    phase2_time = time.time() - t0
    tnc_total_time = phase1_time + phase2_time
    logger.info("TNC Phase 2 done in %.1fs (final loss=%.4f)", phase2_time, ft_loss)

    pred_tnc = predict_tnc(encoder, cls_head, X_test, device)
    m_tnc = evaluate_5_metrics(pred_tnc, te_rets, y_test, te_valid)
    print_5_metrics("TNC (contrastive pre-train + linear probe)", m_tnc)

    # --- Comparison ---
    verdict = verdict_from_metrics(m_ce, m_tnc)

    w = 100
    print("\n" + "=" * w)
    print("TNC vs CE Comparison")
    print("=" * w)
    print(f"  CE  econ_dir={m_ce['econ_dir']:.1%}  transition={m_ce['transition_acc']:.1%}"
          f"  large_move={m_ce['large_move_acc']:.1%}  sharpe={m_ce['sharpe_costs']:+.3f}")
    print(f"  TNC econ_dir={m_tnc['econ_dir']:.1%}  transition={m_tnc['transition_acc']:.1%}"
          f"  large_move={m_tnc['large_move_acc']:.1%}  sharpe={m_tnc['sharpe_costs']:+.3f}")
    delta_econ = m_tnc["econ_dir"] - m_ce["econ_dir"]
    delta_trans = m_tnc["transition_acc"] - m_ce["transition_acc"]
    delta_large = m_tnc["large_move_acc"] - m_ce["large_move_acc"]
    delta_sharpe = m_tnc["sharpe_costs"] - m_ce["sharpe_costs"]
    print(f"  Delta: econ_dir={delta_econ:+.1%}  transition={delta_trans:+.1%}"
          f"  large_move={delta_large:+.1%}  sharpe={delta_sharpe:+.3f}")

    # TNC-specific: check flat rate
    flat_ce = m_ce["pred_dist"]["flat"]
    flat_tnc = m_tnc["pred_dist"]["flat"]
    print(f"\n  CE flat%:  {flat_ce:.1%}")
    print(f"  TNC flat%: {flat_tnc:.1%}")

    # Key thresholds
    print(f"\n  Key targets: transition >= 55%: {'PASS' if m_tnc['transition_acc'] >= 0.55 else 'MISS'}")
    print(f"               econ_dir  >= 64%: {'PASS' if m_tnc['econ_dir'] >= 0.64 else 'MISS'}")
    print(f"               large_move>= 70%: {'PASS' if m_tnc['large_move_acc'] >= 0.70 else 'MISS'}")

    print(f"\n  VERDICT: {verdict}")
    print("=" * w)

    save_results("tnc", {
        "ce_baseline": m_ce,
        "tnc": m_tnc,
        "ce_train_loss": ce_loss,
        "ce_train_time": ce_time,
        "tnc_phase1_loss": tnc_loss,
        "tnc_phase2_loss": ft_loss,
        "tnc_total_time": tnc_total_time,
        "tnc_phase1_epochs": TNC_EPOCHS,
        "tnc_phase2_epochs": FINETUNE_EPOCHS,
        "repr_dim": REPR_DIM,
        "neighborhood_W": NEIGHBORHOOD_W,
        "verdict": verdict,
        "delta": {
            "econ_dir": delta_econ,
            "transition_acc": delta_trans,
            "large_move_acc": delta_large,
            "sharpe_costs": delta_sharpe,
        },
    })


if __name__ == "__main__":
    main()
