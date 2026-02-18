#!/usr/bin/env python3
"""Feature Test 19: N-BEATS Baseline — Pure Deep Learning Diagnostic.

Hypothesis: If N-BEATS (pure FC network, no wavelets) matches wavecast's
~64% econ_dir, wavelets aren't helping. If N-BEATS underperforms, wavelets
are critical. This answers whether to pursue architecture changes or stick
with wavelet preprocessing.

N-BEATS architecture: backward/forward residual stacks with FC blocks.
Input: raw D1 coefficient deltas (same as WaveletGPT).
Output: 5-class quantile prediction.

Usage:
    python -m scripts.feature_tests.test_nbeats
"""

from __future__ import annotations

import json
import logging
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
from torch.utils.data import DataLoader, TensorDataset

from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
)

logger = logging.getLogger(__name__)


class NBeatsBlock(nn.Module):
    """Single N-BEATS block with fully connected layers."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.backward_proj = nn.Linear(hidden_dim, input_dim)
        self.forward_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        backward = self.backward_proj(h)
        forward = self.forward_proj(h)
        return backward, forward


class NBeatsNet(nn.Module):
    """N-BEATS: Neural Basis Expansion Analysis.

    2 stacks of 3 blocks each, classification head on top.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_stacks: int = 2,
        n_blocks: int = 3,
        n_classes: int = 5,
    ) -> None:
        super().__init__()
        self.stacks = nn.ModuleList()
        for _ in range(n_stacks):
            stack = nn.ModuleList()
            for _ in range(n_blocks):
                stack.append(NBeatsBlock(input_dim, hidden_dim, hidden_dim))
            self.stacks.append(stack)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * n_stacks, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stack_outputs = []
        for stack in self.stacks:
            residual = x
            block_sum = torch.zeros(x.shape[0], stack[0].forward_proj.out_features, device=x.device)
            for block in stack:
                backward, forward = block(residual)
                residual = residual - backward
                block_sum = block_sum + forward
            stack_outputs.append(block_sum)

        combined = torch.cat(stack_outputs, dim=-1)
        return self.classifier(combined)


def run_nbeats_test():
    """Run N-BEATS diagnostic experiment."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("N-BEATS diagnostic experiment")

    # Load data (same as harness)
    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))
    logger.info("Loaded %d tickers", len(tickers))

    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)

    logger.info("Building D1 dataset...")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # N-BEATS input: flatten the context + aux features (same X as WaveletGPT minus level/ac columns)
    # X layout: [ctx_0..ctx_15, aux_00..aux_15_3, level, ac]
    ctx_len = CONTEXT_LENGTH
    n_aux = N_AUX_FEATURES
    input_dim = ctx_len + ctx_len * n_aux  # coefficients + aux features

    n_seeds = 3
    baseline_results = []
    nbeats_results = []

    for seed in range(n_seeds):
        logger.info("--- Seed %d/%d ---", seed + 1, n_seeds)
        np.random.seed(seed * 42 + 7)
        torch.manual_seed(seed * 42 + 7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 42 + 7)

        # --- Baseline: WaveletGPT ---
        logger.info("  Training WaveletGPT baseline...")
        model_base = WaveletGPT(
            vocab_size=1, context_length=CONTEXT_LENGTH,
            task="return_quantile", n_output_classes=N_CLASSES,
            class_weights=class_weights, input_mode="continuous",
            n_aux_features=N_AUX_FEATURES, **MODEL_KWARGS,
        )
        y_tr = y_train.astype(np.float64)
        metrics = model_base.fit(X_train, y_tr)
        pred_base = model_base.predict(X_test).astype(np.int64)
        base_eval = _evaluate(pred_base, te_rets, y_test, te_valid, metrics["train_loss"], 0.0)
        baseline_results.append(base_eval)
        logger.info("  Baseline: econ_dir=%.1f%% sharpe=%.3f",
                     base_eval.econ_dir_accuracy * 100, base_eval.sharpe_with_costs)

        # --- N-BEATS ---
        logger.info("  Training N-BEATS...")
        # Extract input features (drop level and ac columns)
        X_train_nb = X_train[:, :input_dim].astype(np.float32)
        X_test_nb = X_test[:, :input_dim].astype(np.float32)

        model_nb = NBeatsNet(
            input_dim=input_dim,
            hidden_dim=256,
            n_stacks=2,
            n_blocks=3,
            n_classes=N_CLASSES,
        ).to(device)

        optimizer = torch.optim.AdamW(model_nb.parameters(), lr=MODEL_KWARGS["learning_rate"])
        weight_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_t)

        train_dataset = TensorDataset(
            torch.tensor(X_train_nb, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=MODEL_KWARGS["batch_size"], shuffle=True
        )

        best_loss = float("inf")
        patience_counter = 0

        for _epoch in range(MODEL_KWARGS["epochs"]):
            model_nb.train()
            total_loss = 0.0
            n_batches = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                logits = model_nb(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_nb.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model_nb.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= MODEL_KWARGS["patience"]:
                    break

        model_nb.load_state_dict(best_state)
        model_nb.eval()
        with torch.no_grad():
            test_tensor = torch.tensor(X_test_nb, dtype=torch.float32).to(device)
            nb_logits = model_nb(test_tensor)
            nb_pred = nb_logits.argmax(dim=-1).cpu().numpy().astype(np.int64)

        nb_eval = _evaluate(nb_pred, te_rets, y_test, te_valid, best_loss, 0.0)
        nbeats_results.append(nb_eval)
        logger.info("  N-BEATS: econ_dir=%.1f%% sharpe=%.3f",
                     nb_eval.econ_dir_accuracy * 100, nb_eval.sharpe_with_costs)

    # Summary
    base_econ = np.mean([r.econ_dir_accuracy for r in baseline_results])
    nb_econ = np.mean([r.econ_dir_accuracy for r in nbeats_results])
    base_sharpe = np.mean([r.sharpe_with_costs for r in baseline_results])
    nb_sharpe = np.mean([r.sharpe_with_costs for r in nbeats_results])
    base_trans = np.mean([r.transition_accuracy for r in baseline_results])
    nb_trans = np.mean([r.transition_accuracy for r in nbeats_results])

    d_econ = nb_econ - base_econ
    d_sharpe = nb_sharpe - base_sharpe
    d_trans = nb_trans - base_trans

    # Diagnostic verdict
    if abs(d_econ) < 0.01:
        diag = "EQUIVALENT — wavelets aren't adding signal beyond FC extraction"
    elif d_econ > 0.01:
        diag = "N-BEATS BETTER — transformer architecture may be hurting"
    else:
        diag = "WAVELET-GPT BETTER — wavelet+transformer synergy is real"

    w = 100
    print()
    print("=" * w)
    print("DIAGNOSTIC TEST: N-BEATS vs WaveletGPT")
    print(f"N-BEATS: 2 stacks × 3 blocks, hidden=256, Seeds: {n_seeds}")
    print("=" * w)
    print(f"\n{'Metric':<20} {'WaveletGPT':>12} {'N-BEATS':>12} {'Delta':>10}")
    print("-" * w)
    print(f"{'Econ Dir Acc':<20} {base_econ:>11.1%} {nb_econ:>11.1%} {d_econ:>+9.1%}")
    print(f"{'Sharpe (+costs)':<20} {base_sharpe:>+11.3f} {nb_sharpe:>+11.3f} {d_sharpe:>+9.3f}")
    print(f"{'Transition Acc':<20} {base_trans:>11.1%} {nb_trans:>11.1%} {d_trans:>+9.1%}")
    print("-" * w)
    print(f"\nDIAGNOSTIC: {diag}")
    print("=" * w)

    # Save
    results_dir = Path.home() / ".wavecast" / "audit" / "feature_tests"
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "name": "nbeats_baseline",
        "n_seeds": n_seeds,
        "diagnostic": diag,
        "baseline_mean": {"econ_dir": base_econ, "sharpe_costs": base_sharpe, "transition": base_trans},
        "nbeats_mean": {"econ_dir": nb_econ, "sharpe_costs": nb_sharpe, "transition": nb_trans},
        "delta": {"econ_dir": d_econ, "sharpe_costs": d_sharpe, "transition": d_trans},
        "per_seed": {
            "baseline": [
                {"econ_dir": r.econ_dir_accuracy, "sharpe_costs": r.sharpe_with_costs, "transition": r.transition_accuracy}
                for r in baseline_results
            ],
            "nbeats": [
                {"econ_dir": r.econ_dir_accuracy, "sharpe_costs": r.sharpe_with_costs, "transition": r.transition_accuracy}
                for r in nbeats_results
            ],
        },
    }
    out_path = results_dir / "nbeats_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)

    return output


if __name__ == "__main__":
    run_nbeats_test()
