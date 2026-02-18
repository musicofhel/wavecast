#!/usr/bin/env python3
"""Experiment 9: g-Layers Calibration.

Source: Gupta et al. 2020 "Post-hoc Calibration by g-Layers"
Hypothesis: Append a small trainable calibration network after the frozen
model. Theoretically proven to produce calibrated predictions.

Pass criteria: ECE < 0.05. Selective econ_dir > 67% at coverage > 30%.

Usage:
    python -m scripts.feature_tests.test_g_layers
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scripts.feature_tests.harness import (
    CONTEXT_LENGTH,
    MODEL_KWARGS,
    N_AUX_FEATURES,
    N_CLASSES,
    PERCENTILES,
    _build_d1_pipeline,
    _evaluate,
    _load_ohlcv,
    _ohlcv_to_timeseries,
    _split_ohlcv,
)

from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
)

logger = logging.getLogger(__name__)


class GLayer(nn.Module):
    """Small calibration network appended to frozen base model logits."""

    def __init__(self, n_classes: int = 5, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_classes, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.net(logits)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = correct[mask].mean()
            ece += mask.sum() / len(labels) * abs(avg_acc - avg_conf)
    return float(ece)


def run_g_layers_test(n_seeds: int = 3) -> dict:
    """Train base model, freeze, add g-layers, compare calibration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger.info("g-Layers calibration experiment, %d seeds", n_seeds)

    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))

    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)

    X_all, tr_rets, tr_valid, tr_lvls, _, _ = _build_d1_pipeline(
        train_prices, None, tickers, None, 0
    )
    X_test, te_rets, te_valid, te_lvls, _, _ = _build_d1_pipeline(
        test_prices, None, tickers, None, 0
    )

    boundaries = compute_quantile_boundaries(
        tr_rets, tr_lvls, tr_valid, PERCENTILES, per_level=True
    )
    y_all = assign_quantile_labels(tr_rets, tr_lvls, boundaries)
    y_test = assign_quantile_labels(te_rets, te_lvls, boundaries)

    # Split train into 80% train, 20% calibration
    n_total = len(X_all)
    n_train = int(0.8 * n_total)
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_cal = X_all[n_train:]
    y_cal = y_all[n_train:]

    valid_labels = y_train[y_train >= 0]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    y_tr = y_train.astype(np.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_eces = []
    glayer_eces = []
    base_sel_econs = []
    glayer_sel_econs = []
    base_evals = []
    glayer_evals = []

    for seed in range(n_seeds):
        logger.info("--- Seed %d/%d ---", seed + 1, n_seeds)

        # Train base model
        np.random.seed(seed * 42 + 7)
        torch.manual_seed(seed * 42 + 7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 42 + 7)

        model_base = WaveletGPT(
            vocab_size=1, context_length=CONTEXT_LENGTH,
            task="return_quantile", n_output_classes=N_CLASSES,
            class_weights=class_weights, input_mode="continuous",
            n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
        )
        t0 = time.time()
        metrics = model_base.fit(X_train, y_tr)
        base_time = time.time() - t0

        # Get base model probabilities on test set
        base_probs_test = model_base.predict_proba(X_test)
        base_preds_test = base_probs_test.argmax(axis=1).astype(np.int64)

        base_eval = _evaluate(
            base_preds_test, te_rets, y_test, te_valid,
            metrics["train_loss"], base_time,
        )
        base_evals.append(base_eval)

        # Base ECE
        valid_mask_test = te_valid.astype(bool)
        base_ece = compute_ece(
            base_probs_test[valid_mask_test], y_test[valid_mask_test]
        )
        base_eces.append(base_ece)

        # Base selective performance (max softmax threshold)
        base_conf = base_probs_test[valid_mask_test].max(axis=1)
        base_preds_v = base_preds_test[valid_mask_test]
        base_rets_v = te_rets[valid_mask_test]
        threshold_30 = np.quantile(base_conf, 0.7)  # top 30% confidence
        keep = base_conf >= threshold_30
        if keep.sum() > 0:
            mid = N_CLASSES // 2
            pd = np.where(base_preds_v[keep] > mid, 1, np.where(base_preds_v[keep] < mid, -1, 0))
            ad = np.sign(base_rets_v[keep])
            dm = (pd != 0) & (ad != 0)
            base_sel_econ = float((pd[dm] == ad[dm]).mean()) if dm.sum() > 0 else 0.0
        else:
            base_sel_econ = 0.0
        base_sel_econs.append(base_sel_econ)

        # --- g-Layers calibration ---
        # Get logits on calibration set
        base_probs_cal = model_base.predict_proba(X_cal)
        cal_logits = torch.tensor(
            np.log(base_probs_cal + 1e-10), dtype=torch.float32
        ).to(device)
        cal_labels = torch.tensor(y_cal.astype(np.int64), dtype=torch.long).to(device)

        # Train g-layer on calibration set
        g_layer = GLayer(n_classes=N_CLASSES, hidden=32).to(device)
        g_opt = torch.optim.Adam(g_layer.parameters(), lr=0.001)
        g_criterion = nn.CrossEntropyLoss()

        g_layer.train()
        for _ in range(200):
            g_opt.zero_grad()
            out = g_layer(cal_logits)
            loss = g_criterion(out, cal_labels)
            loss.backward()
            g_opt.step()

        # Apply g-layer to test set
        test_logits = torch.tensor(
            np.log(base_probs_test + 1e-10), dtype=torch.float32
        ).to(device)
        g_layer.eval()
        with torch.no_grad():
            g_out = g_layer(test_logits)
            g_probs = torch.softmax(g_out, dim=-1).cpu().numpy()

        g_preds = g_probs.argmax(axis=1).astype(np.int64)

        g_eval = _evaluate(
            g_preds, te_rets, y_test, te_valid,
            metrics["train_loss"], base_time,
        )
        glayer_evals.append(g_eval)

        # g-layer ECE
        g_ece = compute_ece(g_probs[valid_mask_test], y_test[valid_mask_test])
        glayer_eces.append(g_ece)

        # g-layer selective
        g_conf = g_probs[valid_mask_test].max(axis=1)
        g_preds_v = g_preds[valid_mask_test]
        g_rets_v = te_rets[valid_mask_test]
        g_threshold_30 = np.quantile(g_conf, 0.7)
        keep_g = g_conf >= g_threshold_30
        if keep_g.sum() > 0:
            pd_g = np.where(g_preds_v[keep_g] > mid, 1, np.where(g_preds_v[keep_g] < mid, -1, 0))
            ad_g = np.sign(g_rets_v[keep_g])
            dm_g = (pd_g != 0) & (ad_g != 0)
            g_sel_econ = float((pd_g[dm_g] == ad_g[dm_g]).mean()) if dm_g.sum() > 0 else 0.0
        else:
            g_sel_econ = 0.0
        glayer_sel_econs.append(g_sel_econ)

        logger.info(
            "  Base: econ=%.1f%% ECE=%.4f sel_econ=%.1f%%",
            base_eval.econ_dir_accuracy * 100, base_ece, base_sel_econ * 100,
        )
        logger.info(
            "  g-Layer: econ=%.1f%% ECE=%.4f sel_econ=%.1f%%",
            g_eval.econ_dir_accuracy * 100, g_ece, g_sel_econ * 100,
        )

    # Aggregate
    avg_base_ece = float(np.mean(base_eces))
    avg_g_ece = float(np.mean(glayer_eces))
    avg_base_sel = float(np.mean(base_sel_econs))
    avg_g_sel = float(np.mean(glayer_sel_econs))
    avg_base_econ = float(np.mean([e.econ_dir_accuracy for e in base_evals]))
    avg_g_econ = float(np.mean([e.econ_dir_accuracy for e in glayer_evals]))
    avg_base_sharpe = float(np.mean([e.sharpe_with_costs for e in base_evals]))
    avg_g_sharpe = float(np.mean([e.sharpe_with_costs for e in glayer_evals]))

    verdict = "PASS" if avg_g_ece < 0.05 and avg_g_sel > 0.67 else "FAIL"

    w = 100
    print()
    print("=" * w)
    print("EXPERIMENT: g-Layers Post-Hoc Calibration")
    print("=" * w)
    print(f"\n{'Metric':<20} {'Baseline':>10} {'g-Layer':>10} {'Delta':>10}")
    print("-" * w)
    print(f"{'Econ Dir Acc':<20} {avg_base_econ:>9.1%} {avg_g_econ:>9.1%} {avg_g_econ - avg_base_econ:>+9.1%}")
    print(f"{'Sharpe (+costs)':<20} {avg_base_sharpe:>+9.3f} {avg_g_sharpe:>+9.3f} {avg_g_sharpe - avg_base_sharpe:>+9.3f}")
    print(f"{'ECE':<20} {avg_base_ece:>9.4f} {avg_g_ece:>9.4f} {avg_g_ece - avg_base_ece:>+9.4f}")
    print(f"{'Sel Econ @30%':<20} {avg_base_sel:>9.1%} {avg_g_sel:>9.1%} {avg_g_sel - avg_base_sel:>+9.1%}")
    print("-" * w)
    print(f"\nVERDICT: {verdict}")
    print("=" * w)

    results_dir = Path.home() / ".wavecast" / "audit" / "feature_tests"
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "name": "g_layers",
        "experiment_type": "calibration",
        "n_seeds": n_seeds,
        "verdict": verdict,
        "baseline": {
            "econ_dir": avg_base_econ,
            "sharpe_costs": avg_base_sharpe,
            "ece": avg_base_ece,
            "selective_econ_30": avg_base_sel,
        },
        "g_layer": {
            "econ_dir": avg_g_econ,
            "sharpe_costs": avg_g_sharpe,
            "ece": avg_g_ece,
            "selective_econ_30": avg_g_sel,
        },
    }
    out_path = results_dir / "g_layers_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)
    return output


if __name__ == "__main__":
    run_g_layers_test()
