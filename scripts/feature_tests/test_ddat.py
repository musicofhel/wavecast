#!/usr/bin/env python3
"""DDAT: Data-Driven Adaptive Training via autoencoder reconstruction error.

Uses a small autoencoder to learn the manifold of D1 input features. Samples
with high reconstruction error are "unusual" — likely near regime boundaries
or transitions. The per-sample reconstruction error becomes a difficulty signal
that up-weights hard/unusual samples in the classification loss.

Two-phase approach:
  Phase 1: Train a lightweight autoencoder on the D1 feature vectors (20 epochs).
  Phase 2: Compute per-sample reconstruction error, normalize to [0, 1], and
           train WaveletGPTNet with weighted CE:
             loss_i = (1 + alpha * norm_recon_error_i) * CE_i

Alpha = 1.0 (tunable). Samples near the manifold center get weight ~1.0;
unusual samples get weight up to 2.0.

Key metric targets: transition >= 55%, econ_dir >= 64%.

Usage:
    python -m scripts.feature_tests.test_ddat
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
AE_EPOCHS = 20
AE_HIDDEN_DIM = 32
ALPHA = 1.0  # reconstruction error weight multiplier


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------


class SimpleAutoencoder(nn.Module):
    """Symmetric autoencoder for input feature reconstruction."""

    def __init__(self, input_dim: int, hidden_dim: int = AE_HIDDEN_DIM) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


# ---------------------------------------------------------------------------
# parse_x helper
# ---------------------------------------------------------------------------


def parse_x(
    X: np.ndarray,
    ctx_len: int = CONTEXT_LENGTH,
    n_aux: int = N_AUX_FEATURES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split flat X array into (ctx, lvl, ac, aux) numpy arrays."""
    ctx = X[:, :ctx_len].astype(np.float32)
    aux_end = ctx_len + ctx_len * n_aux
    aux = X[:, ctx_len:aux_end].astype(np.float32).reshape(-1, ctx_len, n_aux)
    lvl = X[:, aux_end].astype(np.int64)
    ac = X[:, aux_end + 1].astype(np.int64)
    return ctx, lvl, ac, aux


def extract_ae_features(X: np.ndarray) -> np.ndarray:
    """Extract the continuous features for the autoencoder (context + aux, no ids)."""
    ctx_len = CONTEXT_LENGTH
    n_aux = N_AUX_FEATURES
    aux_end = ctx_len + ctx_len * n_aux
    # Context values + flattened aux features
    return X[:, :aux_end].astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 1: Autoencoder training + reconstruction error
# ---------------------------------------------------------------------------


def train_autoencoder(
    features: np.ndarray,
    device: torch.device,
) -> tuple[SimpleAutoencoder, np.ndarray]:
    """Train autoencoder and return (model, per-sample reconstruction error)."""
    input_dim = features.shape[1]
    ae = SimpleAutoencoder(input_dim).to(device)

    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=MODEL_KWARGS["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(AE_EPOCHS):
        ae.train()
        total, n = 0.0, 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = ae(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        if (epoch + 1) % 5 == 0:
            logger.info("  AE epoch %d/%d  loss=%.6f", epoch + 1, AE_EPOCHS, total / max(n, 1))

    # Compute per-sample reconstruction error
    ae.eval()
    all_features = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon = ae(all_features)
        per_sample_err = ((recon - all_features) ** 2).mean(dim=1).cpu().numpy()

    return ae, per_sample_err


# ---------------------------------------------------------------------------
# Phase 2: Weighted CE training
# ---------------------------------------------------------------------------


def train_weighted_ce(
    net: WaveletGPTNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: np.ndarray,
    class_weights: list[float],
    device: torch.device,
) -> float:
    """Train WaveletGPTNet with per-sample weighted CE. Returns final loss."""
    ctx, lvl, ac, aux = parse_x(X_train)
    y = y_train.astype(np.int64)
    n = len(y)

    cw_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    ctx_t = torch.tensor(ctx, dtype=torch.float32)
    lvl_t = torch.tensor(lvl, dtype=torch.long)
    ac_t = torch.tensor(ac, dtype=torch.long)
    aux_t = torch.tensor(aux.reshape(n, -1), dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    w_t = torch.tensor(sample_weights, dtype=torch.float32)
    dataset = TensorDataset(ctx_t, lvl_t, ac_t, aux_t, y_t, w_t)

    loader = DataLoader(
        dataset, batch_size=MODEL_KWARGS["batch_size"], shuffle=True,
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=MODEL_KWARGS["learning_rate"])

    avg_loss = 0.0
    for epoch in range(MODEL_KWARGS["epochs"]):
        net.train()
        total, count = 0.0, 0
        for batch in loader:
            ctx_b = batch[0].to(device)
            lvl_b = batch[1].to(device)
            ac_b = batch[2].to(device)
            aux_b = batch[3].to(device).reshape(-1, CONTEXT_LENGTH, N_AUX_FEATURES)
            tgt = batch[4].to(device)
            w_b = batch[5].to(device)

            optimizer.zero_grad()
            logits_dict = net(ctx_b, lvl_b, ac_b, aux_features=aux_b)
            # Per-sample CE with class weights
            ce_per_sample = nn.functional.cross_entropy(
                logits_dict[1], tgt, weight=cw_tensor, reduction="none",
            )
            # Apply per-sample difficulty weights
            loss = (w_b * ce_per_sample).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            count += 1

        avg_loss = total / max(count, 1)
        if (epoch + 1) % 5 == 0:
            logger.info(
                "  DDAT epoch %d/%d  loss=%.4f", epoch + 1, MODEL_KWARGS["epochs"], avg_loss,
            )

    return avg_loss


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------


def predict_from_net(
    net: WaveletGPTNet, X: np.ndarray, device: torch.device,
) -> np.ndarray:
    """Argmax predictions from a raw WaveletGPTNet."""
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
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("DDAT: Data-Driven Adaptive Training via autoencoder difficulty")

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

    # --- Phase 1: Autoencoder ---
    logger.info("Phase 1: Training autoencoder on D1 features...")
    ae_features = extract_ae_features(X_train)
    logger.info("AE input dim: %d", ae_features.shape[1])

    t0 = time.time()
    _ae_model, recon_error = train_autoencoder(ae_features, device)
    ae_time = time.time() - t0
    logger.info(
        "AE done in %.1fs. Recon error: mean=%.6f std=%.6f max=%.6f",
        ae_time, recon_error.mean(), recon_error.std(), recon_error.max(),
    )

    # Normalize reconstruction error to [0, 1]
    err_min = recon_error.min()
    err_max = recon_error.max()
    if err_max > err_min:
        norm_error = (recon_error - err_min) / (err_max - err_min)
    else:
        norm_error = np.zeros_like(recon_error)

    # Per-sample weights: 1 + alpha * normalized_error
    sample_weights = 1.0 + ALPHA * norm_error

    # --- CE Baseline (uniform weights) ---
    logger.info("Training CE baseline (uniform weights)...")
    model_ce = WaveletGPT(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=N_CLASSES,
        class_weights=class_weights, input_mode="continuous",
        n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
    )
    t0 = time.time()
    metrics_ce = model_ce.fit(X_train, y_train.astype(np.float64))
    ce_time = time.time() - t0
    pred_ce = model_ce.predict(X_test).astype(np.int64)
    m_ce = evaluate_5_metrics(pred_ce, te_rets, y_test, te_valid)
    print_5_metrics("CE Baseline", m_ce)

    # --- Phase 2: DDAT (weighted CE) ---
    logger.info("Phase 2: Training DDAT model (alpha=%.1f)...", ALPHA)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    net_ddat = WaveletGPTNet(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=N_CLASSES,
        input_mode="continuous", n_aux_features=N_AUX_FEATURES,
        **NET_KWARGS,
    ).to(device)

    t0 = time.time()
    ddat_loss = train_weighted_ce(
        net_ddat, X_train, y_train, sample_weights, class_weights, device,
    )
    ddat_time = time.time() - t0
    logger.info("DDAT training done in %.1fs (final loss=%.4f)", ddat_time, ddat_loss)

    pred_ddat = predict_from_net(net_ddat, X_test, device)
    m_ddat = evaluate_5_metrics(pred_ddat, te_rets, y_test, te_valid)
    print_5_metrics("DDAT (alpha=1.0)", m_ddat)

    # --- Comparison ---
    verdict = verdict_from_metrics(m_ce, m_ddat)

    w = 100
    print("\n" + "=" * w)
    print("DDAT vs CE Comparison")
    print("=" * w)
    print(f"  CE   econ_dir={m_ce['econ_dir']:.1%}  transition={m_ce['transition_acc']:.1%}"
          f"  large_move={m_ce['large_move_acc']:.1%}  sharpe={m_ce['sharpe_costs']:+.3f}")
    print(f"  DDAT econ_dir={m_ddat['econ_dir']:.1%}  transition={m_ddat['transition_acc']:.1%}"
          f"  large_move={m_ddat['large_move_acc']:.1%}  sharpe={m_ddat['sharpe_costs']:+.3f}")
    delta_econ = m_ddat["econ_dir"] - m_ce["econ_dir"]
    delta_trans = m_ddat["transition_acc"] - m_ce["transition_acc"]
    delta_large = m_ddat["large_move_acc"] - m_ce["large_move_acc"]
    delta_sharpe = m_ddat["sharpe_costs"] - m_ce["sharpe_costs"]
    print(f"  Delta: econ_dir={delta_econ:+.1%}  transition={delta_trans:+.1%}"
          f"  large_move={delta_large:+.1%}  sharpe={delta_sharpe:+.3f}")

    # DDAT-specific: reconstruction error distribution vs outcome
    flat_ce = m_ce["pred_dist"]["flat"]
    flat_ddat = m_ddat["pred_dist"]["flat"]
    print(f"\n  CE flat%:   {flat_ce:.1%}")
    print(f"  DDAT flat%: {flat_ddat:.1%}")

    # Weight distribution stats
    print("\n  Sample weight stats:")
    print(f"    mean={sample_weights.mean():.3f}  std={sample_weights.std():.3f}"
          f"  min={sample_weights.min():.3f}  max={sample_weights.max():.3f}")

    print(f"\n  VERDICT: {verdict}")
    print("=" * w)

    save_results("ddat", {
        "ce_baseline": m_ce,
        "ddat": m_ddat,
        "ce_train_loss": metrics_ce["train_loss"],
        "ce_train_time": ce_time,
        "ddat_train_loss": ddat_loss,
        "ddat_train_time": ddat_time,
        "ae_train_time": ae_time,
        "alpha": ALPHA,
        "ae_hidden_dim": AE_HIDDEN_DIM,
        "ae_epochs": AE_EPOCHS,
        "recon_error_stats": {
            "mean": float(recon_error.mean()),
            "std": float(recon_error.std()),
            "min": float(recon_error.min()),
            "max": float(recon_error.max()),
        },
        "sample_weight_stats": {
            "mean": float(sample_weights.mean()),
            "std": float(sample_weights.std()),
            "min": float(sample_weights.min()),
            "max": float(sample_weights.max()),
        },
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
