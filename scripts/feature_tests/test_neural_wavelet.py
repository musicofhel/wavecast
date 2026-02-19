#!/usr/bin/env python3
"""Neural Wavelet — trainable wavelet decomposition layer.

Replaces the fixed db4 wavelet filter bank (applied during data preprocessing)
with a learnable low-pass / high-pass pair parameterized as nn.Parameters.  The
QMF (quadrature mirror filter) constraint is enforced so the learned pair
remains a valid wavelet decomposition.

The NeuralWaveletLayer produces multi-scale detail coefficients + approximation
which are concatenated per-position and fed into the standard transformer
backbone.  Everything is trained end-to-end with cross-entropy.

Usage:
    python -m scripts.feature_tests.test_neural_wavelet
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
FILTER_LENGTH = 8
N_SCALES = 3


# ---------------------------------------------------------------------------
# Neural Wavelet Layer
# ---------------------------------------------------------------------------


class NeuralWaveletLayer(nn.Module):
    """Learnable wavelet decomposition via parameterized low-/high-pass filters.

    The low-pass filter is parameterized as raw weights passed through softmax
    for normalization.  The high-pass (detail) filter is derived via the QMF
    relation: g[n] = (-1)^n * h[L-1-n].
    """

    def __init__(self, filter_length: int = FILTER_LENGTH, n_scales: int = N_SCALES) -> None:
        super().__init__()
        self.n_scales = n_scales
        self.filter_length = filter_length
        # Learnable low-pass filter (raw logits — normalized via softmax)
        self.low_pass = nn.Parameter(torch.randn(filter_length) * 0.1)

    def get_filters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized (low-pass h, high-pass g) filter pair."""
        h = f_nn.softmax(self.low_pass, dim=0)
        signs = torch.tensor(
            [(-1.0) ** i for i in range(self.filter_length)],
            device=h.device,
            dtype=h.dtype,
        )
        g = signs * h.flip(0)
        return h, g

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-scale wavelet decomposition.

        Args:
            x: (batch, seq_len) raw context coefficients.

        Returns:
            details: (batch, n_scales, seq_len) detail coefficients per scale.
            approx:  (batch, seq_len) final approximation coefficients.
        """
        h, g = self.get_filters()
        details: list[torch.Tensor] = []
        approx = x  # (batch, seq_len)

        for _ in range(self.n_scales):
            h_k = h.unsqueeze(0).unsqueeze(0)  # (1, 1, filter_len)
            g_k = g.unsqueeze(0).unsqueeze(0)  # (1, 1, filter_len)
            a_in = approx.unsqueeze(1)  # (batch, 1, seq_len)
            detail = f_nn.conv1d(a_in, g_k, padding="same").squeeze(1)  # (batch, seq_len)
            approx = f_nn.conv1d(a_in, h_k, padding="same").squeeze(1)  # (batch, seq_len)
            details.append(detail)

        return torch.stack(details, dim=1), approx  # (B, n_scales, S), (B, S)


# ---------------------------------------------------------------------------
# Full model: Neural Wavelet + Transformer
# ---------------------------------------------------------------------------


class NeuralWaveletModel(nn.Module):
    """End-to-end model with a learnable wavelet front-end.

    1. NeuralWaveletLayer decomposes the context into multi-scale details +
       approximation.
    2. Per-position features from all scales + approx are projected to
       ``embed_dim`` via a linear layer.
    3. Standard causal transformer encoder produces the final hidden state.
    4. Classification head outputs N_CLASSES logits.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        context_length: int = CONTEXT_LENGTH,
        n_aux_features: int = N_AUX_FEATURES,
        n_classes: int = N_CLASSES,
        filter_length: int = FILTER_LENGTH,
        n_scales: int = N_SCALES,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.n_aux_features = n_aux_features
        self.n_scales = n_scales

        self.wavelet = NeuralWaveletLayer(filter_length=filter_length, n_scales=n_scales)

        # Project multi-scale features (n_scales detail + 1 approx) per position
        self.scale_proj = nn.Linear(n_scales + 1, embed_dim)

        # Auxiliary features
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
        self.cls_head = nn.Linear(embed_dim, n_classes)

    def forward(
        self,
        ctx: torch.Tensor,
        lvl: torch.Tensor,
        ac: torch.Tensor,
        aux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s = ctx.shape
        pos = torch.arange(s, device=ctx.device).unsqueeze(0).expand(b, -1)

        # Multi-scale wavelet decomposition
        details, approx = self.wavelet(ctx)  # (B, n_scales, S), (B, S)

        # Stack: (B, S, n_scales + 1) — details transposed + approx
        scale_features = torch.cat(
            [details.permute(0, 2, 1), approx.unsqueeze(-1)],  # (B, S, n_scales), (B, S, 1)
            dim=-1,
        )  # (B, S, n_scales + 1)

        x = self.scale_proj(scale_features) + self.pos_embed(pos)  # (B, S, embed_dim)

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
        return self.cls_head(last_hidden)  # (B, n_classes)


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
# Training
# ---------------------------------------------------------------------------


def train_neural_wavelet(
    model: NeuralWaveletModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    valid: np.ndarray,
    class_weights: list[float],
    device: torch.device,
) -> float:
    """Train end-to-end with CE. Returns final epoch loss."""
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=MODEL_KWARGS["learning_rate"])

    avg_loss = 0.0
    for epoch in range(MODEL_KWARGS["epochs"]):
        model.train()
        total, n_b = 0.0, 0
        for batch in loader:
            ctx_b = batch[0].to(device)
            lvl_b = batch[1].to(device)
            ac_b = batch[2].to(device)
            aux_b = batch[3].to(device).reshape(-1, CONTEXT_LENGTH, N_AUX_FEATURES)
            tgt = batch[4].to(device)

            optimizer.zero_grad()
            logits = model(ctx_b, lvl_b, ac_b, aux=aux_b)
            loss = criterion(logits, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total += loss.item()
            n_b += 1
        avg_loss = total / max(n_b, 1)
        if (epoch + 1) % 5 == 0:
            logger.info("  NeuralWavelet epoch %d/%d  loss=%.4f", epoch + 1, MODEL_KWARGS["epochs"], avg_loss)

    return avg_loss


def predict_neural_wavelet(
    model: NeuralWaveletModel,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Get argmax predictions."""
    ctx, lvl, ac, aux = parse_x(X)
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(ctx, dtype=torch.float32).to(device),
            torch.tensor(lvl, dtype=torch.long).to(device),
            torch.tensor(ac, dtype=torch.long).to(device),
            aux=torch.tensor(aux, dtype=torch.float32).to(device),
        )
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
    logger.info(
        "Neural Wavelet experiment: learnable filter bank (filter_len=%d, scales=%d)",
        FILTER_LENGTH, N_SCALES,
    )

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

    # --- CE Baseline (fixed db4 wavelet in preprocessing) ---
    logger.info("Training CE baseline (fixed db4 wavelet)...")
    ce_model, ce_loss, ce_time = train_ce_baseline(X_train, y_train, class_weights)
    pred_ce = ce_model.predict(X_test).astype(np.int64)
    m_ce = evaluate_5_metrics(pred_ce, te_rets, y_test, te_valid)
    print_5_metrics("CE Baseline (fixed db4)", m_ce)

    # --- Neural Wavelet ---
    logger.info("Training Neural Wavelet model (end-to-end)...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    nw_model = NeuralWaveletModel(
        filter_length=FILTER_LENGTH, n_scales=N_SCALES, **NET_KWARGS,
    ).to(device)

    t0 = time.time()
    nw_loss = train_neural_wavelet(nw_model, X_train, y_train, tr_valid, class_weights, device)
    nw_time = time.time() - t0
    logger.info("Neural Wavelet training done in %.1fs (final loss=%.4f)", nw_time, nw_loss)

    pred_nw = predict_neural_wavelet(nw_model, X_test, device)
    m_nw = evaluate_5_metrics(pred_nw, te_rets, y_test, te_valid)
    print_5_metrics("Neural Wavelet (learned filter bank)", m_nw)

    # --- Inspect learned filters ---
    h, g = nw_model.wavelet.get_filters()
    h_np = h.detach().cpu().numpy()
    g_np = g.detach().cpu().numpy()

    # --- Comparison ---
    verdict = verdict_from_metrics(m_ce, m_nw)

    w = 100
    print("\n" + "=" * w)
    print("Neural Wavelet vs CE Baseline (fixed db4)")
    print("=" * w)
    print(f"  CE  econ_dir={m_ce['econ_dir']:.1%}  transition={m_ce['transition_acc']:.1%}"
          f"  large_move={m_ce['large_move_acc']:.1%}  sharpe={m_ce['sharpe_costs']:+.3f}")
    print(f"  NW  econ_dir={m_nw['econ_dir']:.1%}  transition={m_nw['transition_acc']:.1%}"
          f"  large_move={m_nw['large_move_acc']:.1%}  sharpe={m_nw['sharpe_costs']:+.3f}")
    delta_econ = m_nw["econ_dir"] - m_ce["econ_dir"]
    delta_trans = m_nw["transition_acc"] - m_ce["transition_acc"]
    delta_large = m_nw["large_move_acc"] - m_ce["large_move_acc"]
    delta_sharpe = m_nw["sharpe_costs"] - m_ce["sharpe_costs"]
    print(f"  Delta: econ_dir={delta_econ:+.1%}  transition={delta_trans:+.1%}"
          f"  large_move={delta_large:+.1%}  sharpe={delta_sharpe:+.3f}")

    print(f"\n  Learned low-pass filter (h): {np.array2string(h_np, precision=4)}")
    print(f"  Learned high-pass filter (g): {np.array2string(g_np, precision=4)}")

    # Key thresholds
    print(f"\n  Key targets: econ_dir  >= 65%: {'PASS' if m_nw['econ_dir'] >= 0.65 else 'MISS'}")
    print(f"               transition >= 55%: {'PASS' if m_nw['transition_acc'] >= 0.55 else 'MISS'}")

    print(f"\n  VERDICT: {verdict}")
    print("=" * w)

    save_results("neural_wavelet", {
        "ce_baseline": m_ce,
        "neural_wavelet": m_nw,
        "ce_train_loss": ce_loss,
        "ce_train_time": ce_time,
        "nw_train_loss": nw_loss,
        "nw_train_time": nw_time,
        "filter_length": FILTER_LENGTH,
        "n_scales": N_SCALES,
        "learned_low_pass": h_np.tolist(),
        "learned_high_pass": g_np.tolist(),
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
