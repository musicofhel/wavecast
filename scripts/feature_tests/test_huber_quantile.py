#!/usr/bin/env python3
"""Huber Quantile Regression: Quantile loss with Huber robustness to outliers.

Combines quantile regression with Huber loss to be robust to return outliers.
Predicts 3 quantile values at tau = {0.25, 0.5, 0.75} using the same
transformer backbone as other Tier 1 experiments.

For each (quantile_pred, target, tau):
  err = target - quantile_pred
  huber = where(|err| < kappa, 0.5 * err^2, kappa * (|err| - 0.5 * kappa))
  quantile_huber = |tau - (err < 0).float()| * huber / kappa

Summed across quantile levels, averaged over batch. kappa=0.02 controls
the Huber threshold — returns beyond kappa are treated linearly.

Direction from median sign, same conversion to class labels as arctan_pinball.

Usage:
    python -m scripts.feature_tests.test_huber_quantile
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
TAU_LEVELS = [0.25, 0.5, 0.75]
N_QUANTILES = len(TAU_LEVELS)
KAPPA = 0.02


class HuberQuantileLoss(nn.Module):
    """Huber-smoothed quantile loss at multiple tau levels.

    For each (pred, target, tau):
      err = target - pred
      huber = where(|err| < kappa, 0.5*err^2, kappa*(|err| - 0.5*kappa))
      quantile_huber = |tau - (err < 0).float()| * huber / kappa

    Args:
        tau_levels: List of quantile levels to predict.
        kappa: Huber threshold controlling outlier robustness.
    """

    def __init__(self, tau_levels: list[float], kappa: float = KAPPA) -> None:
        super().__init__()
        self.register_buffer("tau", torch.tensor(tau_levels, dtype=torch.float32))
        self.kappa = kappa

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Huber quantile loss.

        Args:
            preds: (B, Q) predicted quantile values.
            targets: (B,) actual return values.

        Returns:
            Scalar loss.
        """
        y = targets.unsqueeze(-1)  # (B, 1)
        err = y - preds  # (B, Q)
        abs_err = torch.abs(err)

        # Huber loss
        huber = torch.where(
            abs_err < self.kappa,
            0.5 * err ** 2,
            self.kappa * (abs_err - 0.5 * self.kappa),
        )

        # Quantile weighting: |tau - I(err < 0)|
        indicator = (err < 0).float()
        quantile_weight = torch.abs(self.tau.unsqueeze(0) - indicator)

        loss = quantile_weight * huber / self.kappa  # (B, Q)
        return loss.sum(dim=-1).mean()


class HuberQuantileWaveletModel(nn.Module):
    """Transformer encoder + quantile regression head.

    Same architecture as QuantileWaveletModel: input_proj, pos_embed,
    level_embed, asset_class_embed, causal transformer, ln_f,
    quantile_head outputting Q values.
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
    """Convert median quantile predictions to 5-class labels."""
    labels = np.full(len(median_preds), MID, dtype=np.int64)
    labels[median_preds > threshold] = 3
    labels[median_preds < -threshold] = 1
    return labels


def train_huber_quantile(
    model: HuberQuantileWaveletModel,
    X_train: np.ndarray,
    y_returns: np.ndarray,
    valid_mask: np.ndarray,
    device: torch.device,
    kappa: float = KAPPA,
) -> float:
    """Train Huber quantile model on continuous returns. Returns final loss."""
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
    criterion = HuberQuantileLoss(TAU_LEVELS, kappa=kappa).to(device)

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
    model: HuberQuantileWaveletModel, X: np.ndarray, device: torch.device,
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
    logger.info("Huber quantile regression experiment (kappa=%.3f)", KAPPA)

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

    # --- Huber Quantile ---
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Training Huber quantile model (tau=%s, kappa=%.3f)...", TAU_LEVELS, KAPPA)

    hq_model = HuberQuantileWaveletModel(**NET_KWARGS).to(device)
    t0 = time.time()
    hq_loss = train_huber_quantile(hq_model, X_train, tr_rets, tr_valid, device, kappa=KAPPA)
    hq_time = time.time() - t0
    logger.info("Huber quantile training done in %.1fs (final loss=%.6f)", hq_time, hq_loss)

    q_preds = predict_quantiles(hq_model, X_test, device)
    median_preds = q_preds[:, 1]  # tau=0.5 is index 1
    pred_labels = quantile_to_labels(median_preds)
    m_hq = evaluate_5_metrics(pred_labels, te_rets, y_test, te_valid)
    print_5_metrics("Huber Quantile Regression", m_hq)

    # --- Comparison ---
    verdict = verdict_from_metrics(m_ce, m_hq)

    w = 100
    print("\n" + "=" * w)
    print("Huber Quantile Regression Results")
    print("=" * w)
    print(f"  CE  econ_dir={m_ce['econ_dir']:.1%}  transition={m_ce['transition_acc']:.1%}"
          f"  large_move={m_ce['large_move_acc']:.1%}  sharpe={m_ce['sharpe_costs']:+.3f}")
    print(f"  HQR econ_dir={m_hq['econ_dir']:.1%}  transition={m_hq['transition_acc']:.1%}"
          f"  large_move={m_hq['large_move_acc']:.1%}  sharpe={m_hq['sharpe_costs']:+.3f}")

    # Quantile spread and monotonicity analysis
    valid_q = q_preds[te_valid]
    q25_mean = np.mean(valid_q[:, 0])
    q50_mean = np.mean(valid_q[:, 1])
    q75_mean = np.mean(valid_q[:, 2])
    iqr_mean = np.mean(valid_q[:, 2] - valid_q[:, 0])
    monotonic_pct = float(np.mean(
        (valid_q[:, 0] <= valid_q[:, 1]) & (valid_q[:, 1] <= valid_q[:, 2]),
    ))

    print(f"\n  Quantile spread: Q25={q25_mean:.5f}  Q50={q50_mean:.5f}  Q75={q75_mean:.5f}")
    print(f"  Mean IQR: {iqr_mean:.5f}")
    print(f"  Monotonic (Q25 <= Q50 <= Q75): {monotonic_pct:.1%}")

    # Outlier analysis: how many test returns exceed kappa
    valid_rets = te_rets[te_valid & ~np.isnan(te_rets)]
    outlier_pct = float(np.mean(np.abs(valid_rets) > KAPPA))
    print(f"  Returns exceeding kappa ({KAPPA}): {outlier_pct:.1%}")

    print(f"\n  VERDICT: {verdict}")
    print("=" * w)

    save_results("huber_quantile", {
        "ce_baseline": m_ce, "ce_train_time": ce_time,
        "huber_quantile": m_hq, "hq_train_loss": hq_loss, "hq_train_time": hq_time,
        "tau_levels": TAU_LEVELS, "kappa": KAPPA,
        "quantile_spread": {
            "q25_mean": float(q25_mean), "q50_mean": float(q50_mean),
            "q75_mean": float(q75_mean), "iqr_mean": float(iqr_mean),
            "monotonic_pct": monotonic_pct,
        },
        "outlier_pct_beyond_kappa": outlier_pct,
        "verdict": verdict,
    })


if __name__ == "__main__":
    main()
