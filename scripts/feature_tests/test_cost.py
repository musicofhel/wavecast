#!/usr/bin/env python3
"""CoST — Contrastive learning in the frequency domain.

Applies DFT to context windows, projects frequency magnitudes alongside the
time-domain transformer representation, and trains with InfoNCE contrastive
loss.  Positive pairs are created via amplitude-scaling augmentation (preserves
frequency structure); different samples serve as negatives.

Phase 1 (30 epochs): contrastive pre-training with InfoNCE.
Phase 2 (20 epochs): freeze encoder, train linear classifier with CE.

Usage:
    python -m scripts.feature_tests.test_cost
"""

from __future__ import annotations  # noqa: I001

import logging
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
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.targets.returns import assign_quantile_labels, compute_quantile_boundaries

logger = logging.getLogger(__name__)

EMBED_DIM = MODEL_KWARGS["embed_dim"]
NET_KWARGS = {k: MODEL_KWARGS[k] for k in ("embed_dim", "num_heads", "num_layers", "dropout")}
REPR_DIM = 32
COST_EPOCHS = 30
FINETUNE_EPOCHS = 20
TAU = 0.1  # InfoNCE temperature
AUG_SCALE = 0.1  # amplitude scaling noise magnitude


# ---------------------------------------------------------------------------
# Frequency Encoder
# ---------------------------------------------------------------------------


class FrequencyEncoder(nn.Module):
    """Transformer backbone with frequency-domain branch.

    The time-domain branch is a standard causal transformer (same as
    WaveletGPTNet in continuous mode).  The frequency branch takes the FFT
    magnitude of the raw context window and projects it to ``embed_dim``.
    Both branches are concatenated and projected to ``repr_dim``.
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

        # Time-domain branch
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

        # Frequency-domain branch
        freq_bins = context_length // 2 + 1  # rfft output length
        self.freq_proj = nn.Linear(freq_bins, embed_dim)

        # Combine time + freq
        self.combine = nn.Linear(embed_dim * 2, repr_dim)

    def forward(
        self,
        ctx: torch.Tensor,
        lvl: torch.Tensor,
        ac: torch.Tensor,
        aux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s = ctx.shape
        pos = torch.arange(s, device=ctx.device).unsqueeze(0).expand(b, -1)

        # Time-domain
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
        z_time = x[:, -1, :]  # (B, embed_dim)

        # Frequency-domain
        fft_mag = torch.abs(torch.fft.rfft(ctx, dim=1))  # (B, freq_bins)
        z_freq = self.freq_proj(fft_mag)  # (B, embed_dim)

        return self.combine(torch.cat([z_time, z_freq], dim=1))  # (B, repr_dim)


# ---------------------------------------------------------------------------
# InfoNCE Loss
# ---------------------------------------------------------------------------


class InfoNCELoss(nn.Module):
    """InfoNCE / NT-Xent contrastive loss with temperature scaling."""

    def __init__(self, temperature: float = TAU) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_anchor: torch.Tensor, z_pos: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss.

        z_anchor, z_pos: (B, D) L2-normalized representations.
        Other samples in the batch serve as negatives.
        """
        z_a = f_nn.normalize(z_anchor, dim=1)
        z_p = f_nn.normalize(z_pos, dim=1)

        # Similarity matrix: (B, B)
        logits = z_a @ z_p.T / self.temperature  # (B, B)
        labels = torch.arange(len(z_a), device=z_a.device)
        return f_nn.cross_entropy(logits, labels)


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
# Augmentation
# ---------------------------------------------------------------------------


def amplitude_augment(ctx: torch.Tensor, scale: float = AUG_SCALE) -> torch.Tensor:
    """Amplitude scaling augmentation: x * (1 + scale * noise).

    Preserves frequency structure while creating a positive view.
    """
    noise = torch.randn_like(ctx) * scale
    return ctx * (1.0 + noise)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------


def train_cost_phase1(
    encoder: FrequencyEncoder,
    X_train: np.ndarray,
    device: torch.device,
) -> float:
    """Phase 1: CoST contrastive pre-training. Returns final epoch loss."""
    ctx, lvl, ac, aux = parse_x(X_train)

    dataset = TensorDataset(
        torch.tensor(ctx, dtype=torch.float32),
        torch.tensor(lvl, dtype=torch.long),
        torch.tensor(ac, dtype=torch.long),
        torch.tensor(aux.reshape(len(aux), -1), dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=MODEL_KWARGS["batch_size"], shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=MODEL_KWARGS["learning_rate"])
    criterion = InfoNCELoss(temperature=TAU)

    avg_loss = 0.0
    for epoch in range(COST_EPOCHS):
        encoder.train()
        total, n_b = 0.0, 0
        for batch in loader:
            ctx_b = batch[0].to(device)
            lvl_b = batch[1].to(device)
            ac_b = batch[2].to(device)
            aux_b = batch[3].to(device).reshape(-1, CONTEXT_LENGTH, N_AUX_FEATURES)

            # Create augmented view
            ctx_aug = amplitude_augment(ctx_b)

            optimizer.zero_grad()
            z_anchor = encoder(ctx_b, lvl_b, ac_b, aux=aux_b)
            z_pos = encoder(ctx_aug, lvl_b, ac_b, aux=aux_b)

            loss = criterion(z_anchor, z_pos)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            total += loss.item()
            n_b += 1

        avg_loss = total / max(n_b, 1)
        if (epoch + 1) % 10 == 0:
            logger.info("  CoST Phase 1 epoch %d/%d  loss=%.4f", epoch + 1, COST_EPOCHS, avg_loss)

    return avg_loss


def train_cost_phase2(
    encoder: FrequencyEncoder,
    cls_head: nn.Linear,
    X_train: np.ndarray,
    y_train: np.ndarray,
    valid: np.ndarray,
    class_weights: list[float],
    device: torch.device,
) -> float:
    """Phase 2: freeze encoder, train linear classifier with CE."""
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    ctx, lvl, ac, aux = parse_x(X_train)

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
            logger.info("  CoST Phase 2 epoch %d/%d  loss=%.4f", epoch + 1, FINETUNE_EPOCHS, avg_loss)

    return avg_loss


def predict_cost(
    encoder: FrequencyEncoder,
    cls_head: nn.Linear,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Get argmax predictions from encoder + classification head."""
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
    logger.info("CoST experiment: frequency-domain contrastive learning")

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

    # --- CoST ---
    logger.info("Training CoST encoder (Phase 1: InfoNCE, %d epochs, tau=%.2f)...", COST_EPOCHS, TAU)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    encoder = FrequencyEncoder(repr_dim=REPR_DIM, **NET_KWARGS).to(device)

    t0 = time.time()
    cost_loss = train_cost_phase1(encoder, X_train, device)
    phase1_time = time.time() - t0
    logger.info("CoST Phase 1 done in %.1fs (final loss=%.4f)", phase1_time, cost_loss)

    # Phase 2: linear probe
    logger.info("Training CoST classifier (Phase 2: linear probe, %d epochs)...", FINETUNE_EPOCHS)
    cls_head = nn.Linear(REPR_DIM, N_CLASSES).to(device)

    t0 = time.time()
    ft_loss = train_cost_phase2(encoder, cls_head, X_train, y_train, tr_valid, class_weights, device)
    phase2_time = time.time() - t0
    cost_total_time = phase1_time + phase2_time
    logger.info("CoST Phase 2 done in %.1fs (final loss=%.4f)", phase2_time, ft_loss)

    pred_cost = predict_cost(encoder, cls_head, X_test, device)
    m_cost = evaluate_5_metrics(pred_cost, te_rets, y_test, te_valid)
    print_5_metrics("CoST (freq-domain contrastive + linear probe)", m_cost)

    # --- Comparison ---
    verdict = verdict_from_metrics(m_ce, m_cost)

    w = 100
    print("\n" + "=" * w)
    print("CoST vs CE Comparison")
    print("=" * w)
    print(f"  CE   econ_dir={m_ce['econ_dir']:.1%}  transition={m_ce['transition_acc']:.1%}"
          f"  large_move={m_ce['large_move_acc']:.1%}  sharpe={m_ce['sharpe_costs']:+.3f}")
    print(f"  CoST econ_dir={m_cost['econ_dir']:.1%}  transition={m_cost['transition_acc']:.1%}"
          f"  large_move={m_cost['large_move_acc']:.1%}  sharpe={m_cost['sharpe_costs']:+.3f}")
    delta_econ = m_cost["econ_dir"] - m_ce["econ_dir"]
    delta_trans = m_cost["transition_acc"] - m_ce["transition_acc"]
    delta_large = m_cost["large_move_acc"] - m_ce["large_move_acc"]
    delta_sharpe = m_cost["sharpe_costs"] - m_ce["sharpe_costs"]
    print(f"  Delta: econ_dir={delta_econ:+.1%}  transition={delta_trans:+.1%}"
          f"  large_move={delta_large:+.1%}  sharpe={delta_sharpe:+.3f}")

    # Flat-rate check
    flat_ce = m_ce["pred_dist"]["flat"]
    flat_cost = m_cost["pred_dist"]["flat"]
    print(f"\n  CE flat%:   {flat_ce:.1%}")
    print(f"  CoST flat%: {flat_cost:.1%}")

    # Key thresholds
    print(f"\n  Key targets: econ_dir  >= 65%: {'PASS' if m_cost['econ_dir'] >= 0.65 else 'MISS'}")
    print(f"               transition >= 55%: {'PASS' if m_cost['transition_acc'] >= 0.55 else 'MISS'}")

    print(f"\n  VERDICT: {verdict}")
    print("=" * w)

    save_results("cost", {
        "ce_baseline": m_ce,
        "cost": m_cost,
        "ce_train_loss": ce_loss,
        "ce_train_time": ce_time,
        "cost_phase1_loss": cost_loss,
        "cost_phase2_loss": ft_loss,
        "cost_total_time": cost_total_time,
        "cost_phase1_epochs": COST_EPOCHS,
        "cost_phase2_epochs": FINETUNE_EPOCHS,
        "temperature": TAU,
        "aug_scale": AUG_SCALE,
        "repr_dim": REPR_DIM,
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
