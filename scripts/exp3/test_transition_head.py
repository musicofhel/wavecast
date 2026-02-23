"""Experiment 3.8: Auxiliary Transition Head.

Add a second prediction head that predicts "will direction change?"
as binary classification. Joint training with combined loss:

    total_loss = (1 - lambda) * CE_direction + lambda * BCE_transition

Tests lambda = [0.1, 0.2, 0.3, 0.5].

Branch: exp3/wave3-transition-head
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from scripts.exp3.evaluate_representation import (
    BASELINE,
    RESULTS_DIR,
    _serialize_metrics,
)
from scripts.feature_tests.exp2_helpers import evaluate_5_metrics
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

from wavecast.models.wavelet_gpt import WaveletGPTNet
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
)

logger = logging.getLogger(__name__)


class WaveletGPTWithTransitionHead(nn.Module):
    """WaveletGPT with auxiliary binary transition prediction head."""

    def __init__(self, base_net: WaveletGPTNet, embed_dim: int = 64):
        super().__init__()
        self.base = base_net
        self.transition_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, token_ids, level_ids, asset_class_ids, aux_features=None):
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        if self.base.input_mode == "continuous":
            x = self.base.input_proj(token_ids.unsqueeze(-1)) + self.base.pos_embed(positions)
        else:
            x = self.base.token_embed(token_ids) + self.base.pos_embed(positions)

        if self.base.n_aux_features > 0 and aux_features is not None:
            x = x + self.base.aux_proj(aux_features)

        x = x + self.base.level_embed(level_ids).unsqueeze(1)
        x = x + self.base.asset_class_embed(asset_class_ids).unsqueeze(1)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.base.transformer(x, mask=causal_mask, is_causal=True)
        x = self.base.ln_f(x)

        last_hidden = x[:, -1, :]

        # Direction logits from base heads
        direction_logits = {}
        for h in self.base.prediction_horizons:
            direction_logits[h] = self.base.heads[str(h)](last_hidden)

        # Transition logit
        transition_logit = self.transition_head(last_hidden).squeeze(-1)

        return direction_logits, transition_logit


def _compute_transition_labels(returns: NDArray, valid: NDArray) -> NDArray:
    """Compute binary transition labels.

    transition[i] = 1 if sign(return[i]) != sign(return[i-1]), else 0.
    First valid sample defaults to 0 (no prior to compare).
    """
    labels = np.zeros(len(returns), dtype=np.float32)
    directions = np.sign(returns)
    vi = np.where(valid)[0]
    for j in range(1, len(vi)):
        i = vi[j]
        i_prev = vi[j - 1]
        if directions[i] != directions[i_prev]:
            labels[i] = 1.0
    return labels


def _train_with_transition_head(
    X_train, y_train_cls, transition_labels, class_weights,
    X_test, te_rets, te_valid, y_test_cls,
    lam: float, seed: int,
) -> dict:
    """Train model with joint direction + transition loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(seed * 42 + 7)
    torch.manual_seed(seed * 42 + 7)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed * 42 + 7)

    # Build base network
    base_net = WaveletGPTNet(
        vocab_size=1, context_length=CONTEXT_LENGTH,
        embed_dim=MODEL_KWARGS["embed_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_layers=MODEL_KWARGS["num_layers"],
        dropout=MODEL_KWARGS["dropout"],
        task="return_quantile",
        n_output_classes=N_CLASSES,
        input_mode="continuous",
        n_aux_features=N_AUX_FEATURES,
    )
    model = WaveletGPTWithTransitionHead(base_net, MODEL_KWARGS["embed_dim"]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=MODEL_KWARGS["learning_rate"])

    # Loss functions
    cw = torch.tensor(class_weights, dtype=torch.float32).to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=cw)

    # Weight transition BCE for class imbalance
    trans_pos_weight = torch.tensor([2.3], dtype=torch.float32).to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=trans_pos_weight)

    # Parse X
    ctx_len = CONTEXT_LENGTH
    n_aux = N_AUX_FEATURES
    ctx_train = torch.tensor(X_train[:, :ctx_len], dtype=torch.float32).to(device)
    aux_end = ctx_len + ctx_len * n_aux
    aux_train = torch.tensor(
        X_train[:, ctx_len:aux_end].reshape(-1, ctx_len, n_aux),
        dtype=torch.float32,
    ).to(device)
    lvl_train = torch.tensor(X_train[:, aux_end], dtype=torch.long).to(device)
    ac_train = torch.tensor(X_train[:, aux_end + 1], dtype=torch.long).to(device)

    y_dir = torch.tensor(y_train_cls, dtype=torch.long).to(device)
    y_trans = torch.tensor(transition_labels, dtype=torch.float32).to(device)

    # Training loop
    batch_size = MODEL_KWARGS["batch_size"]
    n_samples = len(X_train)
    indices = np.arange(n_samples)

    t0 = time.time()
    for _epoch in range(MODEL_KWARGS["epochs"]):
        model.train()
        np.random.shuffle(indices)
        total_loss = 0.0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = indices[start:end]

            direction_logits, trans_logit = model(
                ctx_train[idx], lvl_train[idx], ac_train[idx],
                aux_features=aux_train[idx],
            )

            dir_loss = ce_loss_fn(direction_logits[1], y_dir[idx])
            trans_loss = bce_loss_fn(trans_logit, y_trans[idx])
            loss = (1 - lam) * dir_loss + lam * trans_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

    train_time = time.time() - t0

    # Evaluate
    model.eval()
    ctx_test = torch.tensor(X_test[:, :ctx_len], dtype=torch.float32).to(device)
    aux_test = torch.tensor(
        X_test[:, ctx_len:aux_end].reshape(-1, ctx_len, n_aux),
        dtype=torch.float32,
    ).to(device)
    lvl_test = torch.tensor(X_test[:, aux_end], dtype=torch.long).to(device)
    ac_test = torch.tensor(X_test[:, aux_end + 1], dtype=torch.long).to(device)

    with torch.no_grad():
        direction_logits, trans_logit = model(
            ctx_test, lvl_test, ac_test, aux_features=aux_test,
        )
    pred_labels = direction_logits[1].argmax(dim=-1).cpu().numpy().astype(np.int64)
    trans_pred = (torch.sigmoid(trans_logit) > 0.5).cpu().numpy()

    metrics = evaluate_5_metrics(pred_labels, te_rets, y_test_cls, te_valid)
    metrics["train_time"] = train_time

    # Transition head standalone accuracy
    vi = np.where(te_valid)[0]
    if len(vi) > 10:
        actual_trans = np.zeros(len(te_rets))
        ad = np.sign(te_rets)
        for j in range(1, len(vi)):
            if ad[vi[j]] != ad[vi[j - 1]]:
                actual_trans[vi[j]] = 1.0
        trans_head_acc = float(np.mean(trans_pred[te_valid] == actual_trans[te_valid]))
        metrics["transition_head_acc"] = trans_head_acc
    else:
        metrics["transition_head_acc"] = 0.5

    return metrics


def run_transition_head_experiment():
    """Run the transition head experiment with lambda sweep."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info("=" * 80)
    logger.info("Experiment 3.8: Auxiliary Transition Head")
    logger.info("=" * 80)

    # Load data
    ohlcv_all = _load_ohlcv()
    train_ohlcv, test_ohlcv = _split_ohlcv(ohlcv_all)
    train_prices = _ohlcv_to_timeseries(train_ohlcv)
    test_prices = _ohlcv_to_timeseries(test_ohlcv)
    tickers = sorted(set(train_ohlcv) & set(test_ohlcv))

    X_train, tr_rets, tr_valid, tr_lvls, _, _ = _build_d1_pipeline(
        train_prices, None, tickers, None, 0,
    )
    X_test, te_rets, te_valid, te_lvls, _, _ = _build_d1_pipeline(
        test_prices, None, tickers, None, 0,
    )

    boundaries = compute_quantile_boundaries(
        tr_rets, tr_lvls, tr_valid, list(PERCENTILES), per_level=True,
    )
    y_train_cls = assign_quantile_labels(tr_rets, tr_lvls, boundaries)
    y_test_cls = assign_quantile_labels(te_rets, te_lvls, boundaries)

    valid_labels = y_train_cls[tr_valid]
    counts = np.bincount(valid_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_weights = (inv_freq / inv_freq.sum() * N_CLASSES).tolist()

    # Transition labels
    transition_labels = _compute_transition_labels(tr_rets, tr_valid)
    trans_rate = float(transition_labels[tr_valid].mean())
    logger.info("Transition rate: %.1f%% (%d transitions in %d valid)",
                trans_rate * 100, int(transition_labels[tr_valid].sum()), int(tr_valid.sum()))

    lambdas = [0.1, 0.2, 0.3, 0.5]
    all_results = {}

    for lam in lambdas:
        logger.info("\n--- Lambda = %.1f ---", lam)
        seed_metrics = []
        for seed in range(3):
            logger.info("  Seed %d/3...", seed + 1)
            m = _train_with_transition_head(
                X_train, y_train_cls, transition_labels, class_weights,
                X_test, te_rets, te_valid, y_test_cls,
                lam=lam, seed=seed,
            )
            seed_metrics.append(m)
            logger.info("    econ_dir=%.1f%% trans=%.1f%% large=%.1f%% trans_head=%.1f%%",
                        m["econ_dir"] * 100, m["transition_acc"] * 100,
                        m["large_move_acc"] * 100, m.get("transition_head_acc", 0) * 100)

        # Average
        avg = {}
        for k in ["econ_dir", "transition_acc", "large_move_acc", "sharpe_costs", "transition_head_acc"]:
            avg[k] = float(np.mean([m.get(k, 0) for m in seed_metrics]))
        avg["pred_dist"] = {
            "up": float(np.mean([m["pred_dist"]["up"] for m in seed_metrics])),
            "flat": float(np.mean([m["pred_dist"]["flat"] for m in seed_metrics])),
            "down": float(np.mean([m["pred_dist"]["down"] for m in seed_metrics])),
        }
        avg["n_valid"] = float(np.mean([m["n_valid"] for m in seed_metrics]))

        all_results[f"lambda_{lam}"] = {
            "avg": avg,
            "per_seed": [_serialize_metrics(m) for m in seed_metrics],
        }

    # Summary
    print("\n" + "=" * 80)
    print("TRANSITION HEAD LAMBDA SWEEP SUMMARY")
    print("=" * 80)
    best_name = None
    best_trans = 0.0
    for name, r in all_results.items():
        a = r["avg"]
        print(f"  {name}: econ_dir={a['econ_dir']:.1%} trans={a['transition_acc']:.1%} "
              f"large={a['large_move_acc']:.1%} flat={a['pred_dist']['flat']:.1%} "
              f"sharpe={a['sharpe_costs']:+.3f} trans_head={a.get('transition_head_acc', 0):.1%}")
        if a["transition_acc"] > best_trans and a["econ_dir"] >= BASELINE["econ_dir"] - 0.02:
            best_trans = a["transition_acc"]
            best_name = name

    # Verdict for best lambda
    if best_name:
        best = all_results[best_name]["avg"]
        if best["econ_dir"] >= 0.65 and best["transition_acc"] >= 0.523:
            verdict = f"PASS: best={best_name} econ_dir={best['econ_dir']:.1%} trans={best['transition_acc']:.1%}"
        elif best["transition_acc"] > BASELINE["transition"] + 0.02:
            verdict = f"INTERESTING: best={best_name} trans={best['transition_acc']:.1%} but econ_dir={best['econ_dir']:.1%}"
        else:
            verdict = f"FAIL: best={best_name} econ_dir={best['econ_dir']:.1%} trans={best['transition_acc']:.1%}"
    else:
        verdict = "FAIL: all lambdas degraded econ_dir"

    logger.info("\nVERDICT: %s", verdict)

    # Save
    import json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "name": "transition_head",
        "verdict": verdict,
        "n_seeds": 3,
        "lambdas": lambdas,
        "results_per_lambda": {k: _serialize_metrics(v) for k, v in all_results.items()},
        "transition_rate": trans_rate,
    }
    out_path = RESULTS_DIR / "transition_head_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)

    return results


if __name__ == "__main__":
    results = run_transition_head_experiment()
    print(f"\nFinal verdict: {results['verdict']}")
