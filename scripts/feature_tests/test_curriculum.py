#!/usr/bin/env python3
"""Curriculum Learning by Transition Density.

Orders training windows by difficulty: easy (pure trend) first, hard (many
direction changes) later. Difficulty is measured as the fraction of consecutive
label transitions within each context window.

Curriculum schedule:
  - Epochs  1-5:  easiest 50% only (lowest transition density)
  - Epochs  6-10: all samples uniformly
  - Epochs 11-20: oversample hard 50% (2x weight on highest density)

Hypothesis: gradually introducing harder examples improves transition accuracy
because the model builds a stable directional representation before learning
to detect reversals.

Key metric target: transition >= 55%.

Usage:
    python -m scripts.feature_tests.test_curriculum
"""

from __future__ import annotations  # noqa: I001

import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

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


# ---------------------------------------------------------------------------
# Difficulty scoring
# ---------------------------------------------------------------------------


def compute_difficulty(y: np.ndarray, ctx_len: int = CONTEXT_LENGTH) -> np.ndarray:
    """Per-sample transition density: fraction of label changes in trailing window.

    For sample i the difficulty is the number of class changes in
    y[i - ctx_len : i] divided by (ctx_len - 1).  Samples with fewer than
    ctx_len predecessors get difficulty 0.
    """
    difficulty = np.zeros(len(y), dtype=np.float64)
    for i in range(ctx_len, len(y)):
        window = y[i - ctx_len:i]
        changes = np.sum(np.diff(window) != 0)
        difficulty[i] = changes / (ctx_len - 1)
    return difficulty


# ---------------------------------------------------------------------------
# parse_x helper (same layout used by harness / other exp2 scripts)
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


# ---------------------------------------------------------------------------
# Curriculum training loop
# ---------------------------------------------------------------------------


def train_curriculum(
    net: WaveletGPTNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
    difficulty: np.ndarray,
    class_weights: list[float],
    device: torch.device,
) -> float:
    """Train with curriculum scheduling. Returns final epoch loss."""
    ctx, lvl, ac, aux = parse_x(X_train)
    y = y_train.astype(np.int64)
    n = len(y)

    cw_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)

    ctx_t = torch.tensor(ctx, dtype=torch.float32)
    lvl_t = torch.tensor(lvl, dtype=torch.long)
    ac_t = torch.tensor(ac, dtype=torch.long)
    aux_t = torch.tensor(aux.reshape(n, -1), dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(ctx_t, lvl_t, ac_t, aux_t, y_t)

    optimizer = torch.optim.AdamW(net.parameters(), lr=MODEL_KWARGS["learning_rate"])

    # Difficulty percentiles for curriculum phases
    median_diff = float(np.median(difficulty))

    total_epochs = MODEL_KWARGS["epochs"]
    avg_loss = 0.0

    for epoch in range(total_epochs):
        # --- Curriculum phase ---
        if epoch < 5:
            # Phase 1: easy 50% only (uniform weight on easy, zero on hard)
            weights = np.where(difficulty <= median_diff, 1.0, 0.0)
            # Ensure at least some samples are included
            if weights.sum() == 0:
                weights[:] = 1.0
        elif epoch < 10:
            # Phase 2: all samples, uniform
            weights = np.ones(n, dtype=np.float64)
        else:
            # Phase 3: oversample hard 50% (2x weight)
            weights = np.where(difficulty > median_diff, 2.0, 1.0)

        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=n,
            replacement=True,
        )
        loader = DataLoader(
            dataset, batch_size=MODEL_KWARGS["batch_size"], sampler=sampler,
        )

        net.train()
        total, count = 0.0, 0
        for batch in loader:
            ctx_b = batch[0].to(device)
            lvl_b = batch[1].to(device)
            ac_b = batch[2].to(device)
            aux_b = batch[3].to(device).reshape(-1, CONTEXT_LENGTH, N_AUX_FEATURES)
            tgt = batch[4].to(device)

            optimizer.zero_grad()
            logits_dict = net(ctx_b, lvl_b, ac_b, aux_features=aux_b)
            loss = criterion(logits_dict[1], tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            count += 1

        avg_loss = total / max(count, 1)
        if (epoch + 1) % 5 == 0:
            phase = "easy-only" if epoch < 5 else ("uniform" if epoch < 10 else "hard-oversample")
            logger.info(
                "  Curriculum epoch %d/%d [%s]  loss=%.4f",
                epoch + 1, total_epochs, phase, avg_loss,
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
    logger.info("Curriculum learning by transition density")

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

    # Difficulty scores
    difficulty = compute_difficulty(y_train)
    easy_pct = float(np.mean(difficulty == 0.0))
    hard_pct = float(np.mean(difficulty > 0.5))
    logger.info(
        "Difficulty: mean=%.3f, zero=%.1f%%, >0.5=%.1f%%",
        difficulty.mean(), easy_pct * 100, hard_pct * 100,
    )

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- CE Baseline (standard random order) ---
    logger.info("Training CE baseline (random order)...")
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

    # --- Curriculum ---
    logger.info("Training curriculum model (easy->all->hard)...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    net_cur = WaveletGPTNet(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        task="return_quantile", n_output_classes=N_CLASSES,
        input_mode="continuous", n_aux_features=N_AUX_FEATURES,
        **NET_KWARGS,
    ).to(device)

    t0 = time.time()
    cur_loss = train_curriculum(net_cur, X_train, y_train, difficulty, class_weights, device)
    cur_time = time.time() - t0
    logger.info("Curriculum training done in %.1fs (final loss=%.4f)", cur_time, cur_loss)

    pred_cur = predict_from_net(net_cur, X_test, device)
    m_cur = evaluate_5_metrics(pred_cur, te_rets, y_test, te_valid)
    print_5_metrics("Curriculum (easy->all->hard)", m_cur)

    # --- Comparison ---
    verdict = verdict_from_metrics(m_ce, m_cur)

    w = 100
    print("\n" + "=" * w)
    print("Curriculum vs CE Comparison")
    print("=" * w)
    print(f"  CE   econ_dir={m_ce['econ_dir']:.1%}  transition={m_ce['transition_acc']:.1%}"
          f"  large_move={m_ce['large_move_acc']:.1%}  sharpe={m_ce['sharpe_costs']:+.3f}")
    print(f"  CUR  econ_dir={m_cur['econ_dir']:.1%}  transition={m_cur['transition_acc']:.1%}"
          f"  large_move={m_cur['large_move_acc']:.1%}  sharpe={m_cur['sharpe_costs']:+.3f}")
    delta_econ = m_cur["econ_dir"] - m_ce["econ_dir"]
    delta_trans = m_cur["transition_acc"] - m_ce["transition_acc"]
    delta_large = m_cur["large_move_acc"] - m_ce["large_move_acc"]
    delta_sharpe = m_cur["sharpe_costs"] - m_ce["sharpe_costs"]
    print(f"  Delta: econ_dir={delta_econ:+.1%}  transition={delta_trans:+.1%}"
          f"  large_move={delta_large:+.1%}  sharpe={delta_sharpe:+.3f}")

    # Curriculum-specific diagnostics
    flat_ce = m_ce["pred_dist"]["flat"]
    flat_cur = m_cur["pred_dist"]["flat"]
    print(f"\n  CE flat%:         {flat_ce:.1%}")
    print(f"  Curriculum flat%: {flat_cur:.1%}")

    print(f"\n  VERDICT: {verdict}")
    print("=" * w)

    save_results("curriculum", {
        "ce_baseline": m_ce,
        "curriculum": m_cur,
        "ce_train_loss": metrics_ce["train_loss"],
        "ce_train_time": ce_time,
        "curriculum_train_loss": cur_loss,
        "curriculum_train_time": cur_time,
        "difficulty_stats": {
            "mean": float(difficulty.mean()),
            "std": float(difficulty.std()),
            "zero_pct": easy_pct,
            "above_half_pct": hard_pct,
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
