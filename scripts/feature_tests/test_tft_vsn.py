#!/usr/bin/env python3
"""TFT-style Variable Selection Network (VSN) via Gated Residual Networks.

Inserts a Variable Selection Network before the transformer encoder.  The 4 aux
features + the raw context are treated as 5 independent "variables" per time
position.  Each variable passes through its own GRN, then a selection GRN
produces softmax weights over variables.  The weighted combination feeds into
the standard transformer.

Key benefit: the learned variable importance weights are interpretable — we can
see which input features the model relies on most for the classification task.

Usage:
    python -m scripts.feature_tests.test_tft_vsn
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
GRN_HIDDEN = 32
# Total variables = 1 (context) + N_AUX_FEATURES (4) = 5
N_VARIABLES = 1 + N_AUX_FEATURES

# Human-readable variable names for reporting
VARIABLE_NAMES = [
    "context (wavelet coeff)",
    "aux_0 (energy ratio)",
    "aux_1 (zero-crossing rate)",
    "aux_2 (local volatility)",
    "aux_3 (approx trend)",
]


# ---------------------------------------------------------------------------
# GRN: Gated Residual Network
# ---------------------------------------------------------------------------


class GRN(nn.Module):
    """Gated Residual Network from Temporal Fusion Transformer (Lim et al. 2021).

    Applies: LayerNorm(skip(x) + sigmoid(gate(h)) * fc2(h))
    where h = ELU(fc1(x)) with dropout.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip: nn.Linear | None = None
        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = f_nn.elu(self.fc1(x))
        h = self.dropout(h)
        out = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))
        skip = self.skip(x) if self.skip is not None else x
        return self.ln(skip + gate * out)


# ---------------------------------------------------------------------------
# VSN: Variable Selection Network
# ---------------------------------------------------------------------------


class VSN(nn.Module):
    """Variable Selection Network.

    Each of ``n_variables`` scalar inputs per time step is independently
    processed by its own GRN, producing an ``embed_dim``-sized representation.
    A selection GRN then produces softmax weights over variables from the
    concatenated representation (mean-pooled over time).  The final output is
    the weighted combination across variables.
    """

    def __init__(
        self,
        n_variables: int,
        embed_dim: int,
        hidden_dim: int = GRN_HIDDEN,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_variables = n_variables
        self.grns = nn.ModuleList([
            GRN(1, hidden_dim, embed_dim, dropout) for _ in range(n_variables)
        ])
        self.selection = GRN(
            n_variables * embed_dim, hidden_dim, n_variables, dropout,
        )

    def forward(
        self, inputs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select and combine variables.

        Args:
            inputs: list of (batch, seq_len, 1) tensors, one per variable.

        Returns:
            selected: (batch, seq_len, embed_dim) weighted variable combination.
            weights:  (batch, n_variables) learned importance weights.
        """
        # Per-variable GRN transform: each (B, S, 1) -> (B, S, embed_dim)
        transformed = [grn(inp) for grn, inp in zip(self.grns, inputs, strict=True)]

        # Concatenate for selection: (B, S, n_vars * embed_dim)
        combined = torch.cat(transformed, dim=-1)

        # Selection weights from time-averaged representation
        weights = f_nn.softmax(
            self.selection(combined.mean(dim=1)),  # (B, n_vars)
            dim=-1,
        )

        # Weighted combination: stack → (B, S, embed_dim, n_vars), weight, sum
        stacked = torch.stack(transformed, dim=-1)  # (B, S, embed_dim, n_vars)
        selected = (
            stacked * weights.unsqueeze(1).unsqueeze(2)
        ).sum(dim=-1)  # (B, S, embed_dim)

        return selected, weights


# ---------------------------------------------------------------------------
# Full model: VSN + Transformer
# ---------------------------------------------------------------------------


class VSNModel(nn.Module):
    """Model with TFT-style Variable Selection before transformer encoder.

    The 5 input variables (1 context coeff + 4 aux features) are first processed
    by the VSN, producing a weighted ``embed_dim`` representation per position.
    This feeds into the standard causal transformer, followed by a classification
    head.
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
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.n_aux_features = n_aux_features
        n_variables = 1 + n_aux_features  # context + aux

        self.vsn = VSN(n_variables, embed_dim, hidden_dim=GRN_HIDDEN, dropout=dropout)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            logits:  (B, n_classes) classification logits.
            weights: (B, n_variables) variable importance weights.
        """
        b, s = ctx.shape
        pos = torch.arange(s, device=ctx.device).unsqueeze(0).expand(b, -1)

        # Build per-variable inputs: list of (B, S, 1)
        variables: list[torch.Tensor] = [ctx.unsqueeze(-1)]  # context coefficients
        if self.n_aux_features > 0 and aux is not None:
            for f in range(self.n_aux_features):
                variables.append(aux[:, :, f : f + 1])  # each aux feature

        # Variable selection
        x, weights = self.vsn(variables)  # (B, S, embed_dim), (B, n_vars)

        x = x + self.pos_embed(pos)
        x = x + self.level_embed(lvl).unsqueeze(1)
        x = x + self.asset_class_embed(ac).unsqueeze(1)

        mask = torch.triu(
            torch.ones(s, s, device=ctx.device, dtype=torch.bool), diagonal=1,
        )
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.ln_f(x)

        last_hidden = x[:, -1, :]
        logits = self.cls_head(last_hidden)  # (B, n_classes)

        return logits, weights


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


def train_vsn(
    model: VSNModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    valid: np.ndarray,
    class_weights: list[float],
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """Train VSN model end-to-end. Returns (final_loss, mean_weights)."""
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
    all_weights: list[np.ndarray] = []

    for epoch in range(MODEL_KWARGS["epochs"]):
        model.train()
        total, n_b = 0.0, 0
        epoch_weights: list[np.ndarray] = []
        for batch in loader:
            ctx_b = batch[0].to(device)
            lvl_b = batch[1].to(device)
            ac_b = batch[2].to(device)
            aux_b = batch[3].to(device).reshape(-1, CONTEXT_LENGTH, N_AUX_FEATURES)
            tgt = batch[4].to(device)

            optimizer.zero_grad()
            logits, weights = model(ctx_b, lvl_b, ac_b, aux=aux_b)
            loss = criterion(logits, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total += loss.item()
            n_b += 1
            epoch_weights.append(weights.detach().cpu().numpy())

        avg_loss = total / max(n_b, 1)
        if epoch_weights:
            all_weights = epoch_weights  # keep last epoch's weights
        if (epoch + 1) % 5 == 0:
            logger.info("  VSN epoch %d/%d  loss=%.4f", epoch + 1, MODEL_KWARGS["epochs"], avg_loss)

    # Mean weights from last epoch
    if all_weights:
        mean_weights = np.concatenate(all_weights, axis=0).mean(axis=0)
    else:
        mean_weights = np.ones(N_VARIABLES) / N_VARIABLES

    return avg_loss, mean_weights


def predict_vsn(
    model: VSNModel,
    X: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions and variable weights from VSN model."""
    ctx, lvl, ac, aux = parse_x(X)
    model.eval()
    with torch.no_grad():
        logits, weights = model(
            torch.tensor(ctx, dtype=torch.float32).to(device),
            torch.tensor(lvl, dtype=torch.long).to(device),
            torch.tensor(ac, dtype=torch.long).to(device),
            aux=torch.tensor(aux, dtype=torch.float32).to(device),
        )
    return logits.argmax(dim=-1).cpu().numpy(), weights.cpu().numpy()


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
    logger.info("TFT VSN experiment: variable selection via Gated Residual Networks")

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

    # --- VSN ---
    logger.info("Training VSN model (n_variables=%d, GRN hidden=%d)...", N_VARIABLES, GRN_HIDDEN)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    vsn_model = VSNModel(**NET_KWARGS).to(device)

    t0 = time.time()
    vsn_loss, train_weights = train_vsn(
        vsn_model, X_train, y_train, tr_valid, class_weights, device,
    )
    vsn_time = time.time() - t0
    logger.info("VSN training done in %.1fs (final loss=%.4f)", vsn_time, vsn_loss)

    pred_vsn, test_weights = predict_vsn(vsn_model, X_test, device)
    m_vsn = evaluate_5_metrics(pred_vsn, te_rets, y_test, te_valid)
    print_5_metrics("VSN (TFT-style variable selection)", m_vsn)

    # Mean test weights
    mean_test_weights = test_weights.mean(axis=0)

    # --- Comparison ---
    verdict = verdict_from_metrics(m_ce, m_vsn)

    w = 100
    print("\n" + "=" * w)
    print("VSN vs CE Comparison")
    print("=" * w)
    print(f"  CE  econ_dir={m_ce['econ_dir']:.1%}  transition={m_ce['transition_acc']:.1%}"
          f"  large_move={m_ce['large_move_acc']:.1%}  sharpe={m_ce['sharpe_costs']:+.3f}")
    print(f"  VSN econ_dir={m_vsn['econ_dir']:.1%}  transition={m_vsn['transition_acc']:.1%}"
          f"  large_move={m_vsn['large_move_acc']:.1%}  sharpe={m_vsn['sharpe_costs']:+.3f}")
    delta_econ = m_vsn["econ_dir"] - m_ce["econ_dir"]
    delta_trans = m_vsn["transition_acc"] - m_ce["transition_acc"]
    delta_large = m_vsn["large_move_acc"] - m_ce["large_move_acc"]
    delta_sharpe = m_vsn["sharpe_costs"] - m_ce["sharpe_costs"]
    print(f"  Delta: econ_dir={delta_econ:+.1%}  transition={delta_trans:+.1%}"
          f"  large_move={delta_large:+.1%}  sharpe={delta_sharpe:+.3f}")

    # Feature importance
    print("\n  Learned Variable Importance (train mean):")
    sorted_idx = np.argsort(-train_weights)
    for rank, idx in enumerate(sorted_idx):
        name = VARIABLE_NAMES[idx] if idx < len(VARIABLE_NAMES) else f"var_{idx}"
        bar = "#" * int(train_weights[idx] * 50)
        print(f"    {rank + 1}. {name:<30s}  {train_weights[idx]:.3f}  {bar}")

    print("\n  Learned Variable Importance (test mean):")
    sorted_idx_test = np.argsort(-mean_test_weights)
    for rank, idx in enumerate(sorted_idx_test):
        name = VARIABLE_NAMES[idx] if idx < len(VARIABLE_NAMES) else f"var_{idx}"
        bar = "#" * int(mean_test_weights[idx] * 50)
        print(f"    {rank + 1}. {name:<30s}  {mean_test_weights[idx]:.3f}  {bar}")

    # Check if top feature is direction-related
    top_var_idx = sorted_idx[0]
    top_var_name = VARIABLE_NAMES[top_var_idx] if top_var_idx < len(VARIABLE_NAMES) else f"var_{top_var_idx}"
    direction_related = top_var_idx in (0, 3)  # context or approx trend
    print(f"\n  Top feature: {top_var_name} (direction-related: {direction_related})")

    # Key thresholds
    print(f"\n  Key targets: econ_dir  >= 65%: {'PASS' if m_vsn['econ_dir'] >= 0.65 else 'MISS'}")
    print(f"               transition >= 55%: {'PASS' if m_vsn['transition_acc'] >= 0.55 else 'MISS'}")

    print(f"\n  VERDICT: {verdict}")
    print("=" * w)

    save_results("tft_vsn", {
        "ce_baseline": m_ce,
        "vsn": m_vsn,
        "ce_train_loss": ce_loss,
        "ce_train_time": ce_time,
        "vsn_train_loss": vsn_loss,
        "vsn_train_time": vsn_time,
        "n_variables": N_VARIABLES,
        "grn_hidden": GRN_HIDDEN,
        "variable_names": VARIABLE_NAMES,
        "train_weights": train_weights.tolist(),
        "test_weights": mean_test_weights.tolist(),
        "top_feature": top_var_name,
        "top_feature_direction_related": direction_related,
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
