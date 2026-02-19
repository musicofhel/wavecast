#!/usr/bin/env python3
"""MDN Head Experiment: Mixture Density Network for continuous return distribution.

Replaces the 5-class softmax classification head with an MDN that outputs a
mixture of K=3 Gaussians. Direction comes from the distribution's mass above/
below zero (CDF). Eliminates the flat class by construction — every prediction
is either up or down based on P(return > 0).

Key idea: Instead of discretizing returns into 5 quantile bins (strong_down,
down, flat, up, strong_up), the MDN directly models P(return | context) as a
continuous density. Direction is read from the mixture CDF at zero.

Metrics:
  - Compare MDN vs CE baseline on the standard 5-metric evaluation
  - MDN predictions are mapped to class labels for fair comparison:
    mixture_mean > 0 → class 3 (up), < 0 → class 1 (down)
    High confidence (|P(up) - 0.5| > 0.2) → class 4 or 0 (strong)

Usage:
    python -m scripts.feature_tests.test_mdn_head
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

# --- MDN architecture constants ---
N_COMPONENTS = 3
EMBED_DIM = MODEL_KWARGS["embed_dim"]
NET_KWARGS = {k: MODEL_KWARGS[k] for k in ("embed_dim", "num_heads", "num_layers", "dropout")}


# ---------------------------------------------------------------------------
# MDN modules
# ---------------------------------------------------------------------------


class MDNHead(nn.Module):
    """Mixture Density Network head: K Gaussian components.

    Outputs (pi, mu, sigma) for a K-component mixture:
      - pi:    (batch, K) mixing coefficients via softmax
      - mu:    (batch, K) component means (unbounded)
      - sigma: (batch, K) component stds via softplus + floor
    """

    def __init__(self, input_dim: int, n_components: int = 3) -> None:
        super().__init__()
        self.n_components = n_components
        self.pi_head = nn.Linear(input_dim, n_components)
        self.mu_head = nn.Linear(input_dim, n_components)
        self.sigma_head = nn.Linear(input_dim, n_components)

    def forward(
        self, h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi = nn.functional.softmax(self.pi_head(h), dim=-1)   # (B, K)
        mu = self.mu_head(h)                                    # (B, K)
        sigma = nn.functional.softplus(self.sigma_head(h)) + 1e-4  # (B, K), > 0
        return pi, mu, sigma


class MDNLoss(nn.Module):
    """Negative log-likelihood of a Gaussian mixture, with log-sum-exp trick."""

    def forward(
        self,
        pi: torch.Tensor,    # (B, K)
        mu: torch.Tensor,    # (B, K)
        sigma: torch.Tensor, # (B, K)
        target: torch.Tensor, # (B,)
    ) -> torch.Tensor:
        y = target.unsqueeze(-1)  # (B, 1)
        # Log of each Gaussian component: log N(y | mu_k, sigma_k)
        log_normal = (
            -0.5 * math.log(2 * math.pi)
            - torch.log(sigma)
            - 0.5 * ((y - mu) / sigma) ** 2
        )  # (B, K)
        # log(pi_k * N_k) = log(pi_k) + log(N_k)
        log_mix = torch.log(pi + 1e-10) + log_normal  # (B, K)
        # log-sum-exp over components
        log_prob = torch.logsumexp(log_mix, dim=-1)  # (B,)
        return -log_prob.mean()


class MDNWaveletModel(nn.Module):
    """Standalone transformer encoder + MDN head.

    Replicates WaveletGPTNet's encoder architecture (input projection,
    positional + level + asset-class embeddings, causal transformer, layer
    norm) but replaces the classification head with an MDNHead.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        context_length: int = CONTEXT_LENGTH,
        n_aux_features: int = N_AUX_FEATURES,
        n_components: int = N_COMPONENTS,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.n_aux_features = n_aux_features

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
        self.mdn = MDNHead(embed_dim, n_components)

    def forward(
        self,
        ctx: torch.Tensor,       # (B, S) float32
        lvl: torch.Tensor,       # (B,)   int64
        ac: torch.Tensor,        # (B,)   int64
        aux: torch.Tensor | None = None,  # (B, S, n_aux) float32
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return self.mdn(last_hidden)


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
# Training loops
# ---------------------------------------------------------------------------


def train_mdn(
    model: MDNWaveletModel,
    X_train: np.ndarray,
    y_returns: np.ndarray,
    valid_mask: np.ndarray,
    device: torch.device,
) -> float:
    """Train MDN model on continuous returns. Returns final epoch loss."""
    ctx, lvl, ac, aux = parse_x(X_train)

    # Only train on valid samples with non-NaN returns
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
    criterion = MDNLoss()

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
            pi, mu, sigma = model(ctx_b, lvl_b, ac_b, aux=aux_b)
            loss = criterion(pi, mu, sigma, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            n += 1
        avg_loss = total / max(n, 1)
        if (epoch + 1) % 5 == 0:
            logger.info("  MDN epoch %d/%d  loss=%.4f", epoch + 1, MODEL_KWARGS["epochs"], avg_loss)

    return avg_loss


def predict_mdn_labels(
    model: MDNWaveletModel,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Convert MDN output to 5-class labels via mixture CDF at zero.

    Mapping:
      P(up) > 0.7 → class 4 (strong up)
      P(up) > 0.5 → class 3 (up)
      P(up) < 0.3 → class 0 (strong down)
      P(up) < 0.5 → class 1 (down)
      else         → class 2 (flat)  [should be rare]
    """
    ctx, lvl, ac, aux = parse_x(X)

    model.eval()
    with torch.no_grad():
        pi, mu, sigma = model(
            torch.tensor(ctx, dtype=torch.float32).to(device),
            torch.tensor(lvl, dtype=torch.long).to(device),
            torch.tensor(ac, dtype=torch.long).to(device),
            aux=torch.tensor(aux, dtype=torch.float32).to(device),
        )

    # P(return > 0) = sum_k pi_k * (1 - Phi(-mu_k / sigma_k))
    # where Phi is the standard normal CDF
    pi_np = pi.cpu().numpy()
    mu_np = mu.cpu().numpy()
    sigma_np = sigma.cpu().numpy()

    # Standard normal CDF via scipy-free computation: 0.5 * erfc(-x / sqrt(2))
    # Phi((0 - mu) / sigma) = Phi(-mu / sigma) = 0.5 * erfc(mu / (sigma * sqrt(2)))
    z = mu_np / (sigma_np * math.sqrt(2.0))
    # erfc approximation not needed — use the relation:
    # P(X > 0) for N(mu, sigma) = Phi(mu / sigma)
    # = 0.5 * (1 + erf(mu / (sigma * sqrt(2))))
    from scipy.special import erf as _erf  # noqa: PLC0415

    p_above_zero_per_component = 0.5 * (1.0 + _erf(z))  # (N, K)
    p_up = np.sum(pi_np * p_above_zero_per_component, axis=1)  # (N,)

    # Map to 5-class labels
    labels = np.full(len(p_up), MID, dtype=np.int64)  # default: flat (class 2)
    labels[p_up > 0.7] = 4   # strong up
    labels[(p_up > 0.5) & (p_up <= 0.7)] = 3   # up
    labels[(p_up < 0.3)] = 0  # strong down
    labels[(p_up >= 0.3) & (p_up < 0.5)] = 1   # down

    return labels


# ---------------------------------------------------------------------------
# CE baseline helper (reuses WaveletGPT wrapper)
# ---------------------------------------------------------------------------


def train_ce_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weights: list[float],
) -> tuple[WaveletGPT, float, float]:
    """Train a standard CE baseline. Returns (model, train_loss, train_time)."""
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
    logger.info("MDN head experiment: continuous return distribution via mixture density")

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

    # --- MDN ---
    logger.info("Training MDN model (K=%d components)...", N_COMPONENTS)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mdn_model = MDNWaveletModel(
        n_components=N_COMPONENTS, **NET_KWARGS,
    ).to(device)

    t0 = time.time()
    mdn_loss = train_mdn(mdn_model, X_train, tr_rets, tr_valid, device)
    mdn_time = time.time() - t0
    logger.info("MDN training done in %.1fs (final loss=%.4f)", mdn_time, mdn_loss)

    pred_mdn = predict_mdn_labels(mdn_model, X_test, device)
    m_mdn = evaluate_5_metrics(pred_mdn, te_rets, y_test, te_valid)
    print_5_metrics("MDN Head (K=3)", m_mdn)

    # --- Comparison ---
    verdict = verdict_from_metrics(m_ce, m_mdn)

    w = 100
    print("\n" + "=" * w)
    print("MDN vs CE Comparison")
    print("=" * w)
    print(f"  CE  econ_dir={m_ce['econ_dir']:.1%}  transition={m_ce['transition_acc']:.1%}"
          f"  large_move={m_ce['large_move_acc']:.1%}  sharpe={m_ce['sharpe_costs']:+.3f}")
    print(f"  MDN econ_dir={m_mdn['econ_dir']:.1%}  transition={m_mdn['transition_acc']:.1%}"
          f"  large_move={m_mdn['large_move_acc']:.1%}  sharpe={m_mdn['sharpe_costs']:+.3f}")
    delta_econ = m_mdn["econ_dir"] - m_ce["econ_dir"]
    delta_trans = m_mdn["transition_acc"] - m_ce["transition_acc"]
    delta_large = m_mdn["large_move_acc"] - m_ce["large_move_acc"]
    delta_sharpe = m_mdn["sharpe_costs"] - m_ce["sharpe_costs"]
    print(f"  Delta: econ_dir={delta_econ:+.1%}  transition={delta_trans:+.1%}"
          f"  large_move={delta_large:+.1%}  sharpe={delta_sharpe:+.3f}")

    # MDN-specific: check flat rate — should be very low by construction
    flat_ce = m_ce["pred_dist"]["flat"]
    flat_mdn = m_mdn["pred_dist"]["flat"]
    print(f"\n  CE flat%:  {flat_ce:.1%}")
    print(f"  MDN flat%: {flat_mdn:.1%}")
    if flat_mdn < flat_ce * 0.5:
        print("  MDN successfully reduced flat predictions (by construction)")
    elif flat_mdn < 0.05:
        print("  MDN has near-zero flat predictions (expected)")

    print(f"\n  VERDICT: {verdict}")
    print("=" * w)

    save_results("mdn_head", {
        "ce_baseline": m_ce,
        "mdn_head": m_mdn,
        "ce_train_loss": ce_loss,
        "ce_train_time": ce_time,
        "mdn_train_loss": mdn_loss,
        "mdn_train_time": mdn_time,
        "n_components": N_COMPONENTS,
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
