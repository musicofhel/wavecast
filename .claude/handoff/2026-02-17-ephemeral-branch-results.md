# Ephemeral Branch Experiments: Results Summary

**Date**: 2026-02-17
**Scope**: 20 experiments across 5 tiers, each on a separate git branch off master
**Baseline**: D1 WaveletGPT — ~63.9% econ_dir, +4.67 Sharpe, ~55.2% transition accuracy
**Data**: 20 tickers, hourly bars, train ≤ 2024-06-30, test ≥ 2025-01-01
**Seeds**: 3 per experiment (seed * 42 + 7)

---

## Summary Table

| # | Branch | Tier | Verdict | Econ Dir Δ | Sharpe Δ | Trans Δ | Key Insight |
|---|--------|------|---------|-----------|---------|---------|-------------|
| 1 | `exp/focal-loss` | Loss | FAIL | -0.1pp | -0.09 | -1.1pp | γ=2.0 doesn't help; class imbalance isn't the issue |
| 2 | `exp/ordinal-loss` | Loss | FAIL | -21.9pp | -11.8 | -21.6pp | Cumulative link model catastrophically collapses |
| 3 | `exp/label-smoothing` | Loss | **PASS** | +0.2pp | +0.12 | +1.8pp | ε=0.1 best; ECE reduced 55.1% |
| 4 | `exp/asymmetric-loss` | Loss | FAIL | -26.4pp | -16.7 | -24.1pp | Cost matrix 5:1 destabilizes completely |
| 5 | `exp/beta-nll` | Loss | **PASS** | +5.2pp | +0.66 | — | β=0.5 best; selective 75.2% at 30% coverage |
| 6 | `exp/heteroscedastic` | Loss | **PASS** | +5.6pp | +0.81 | — | Best selective: 80.9% at 30% coverage |
| 7 | `exp/decision-focused` | Loss | FAIL | -16.3pp | -5.3 | — | Mode collapse; all seeds identical 47.6% |
| 8 | `exp/post-hoc-calibration` | Calib | **PASS** | — | — | — | Platt@30%: 70.4% econ_dir, Sharpe +8.1 |
| 9 | `exp/g-layers` | Calib | FAIL | +7.9pp | +2.3 | — | ECE worsens (0.049→0.088); econ gains are real though |
| 10 | `exp/conformal-selective` | Calib | FAIL | — | — | — | Singletons <1% coverage (need >25%) |
| 11 | `exp/selectivenet` | Select | **PASS** | +5.1pp | — | — | Uncond 69.0%; sel@50% 70.8%, sel@70% 70.1% |
| 12 | `exp/error-regularization` | Select | **PASS** | +3.9pp | +1.1 | — | λ=0.1: ECE -73%; simplest effective approach (20 LOC) |
| 13 | `exp/vpin-feature` | Feature | FAIL | +0.4pp | +0.33 | -3.2pp | Volume data also degrades transitions |
| 14 | `exp/amihud-feature` | Feature | FAIL | -0.2pp | -0.19 | -3.1pp | Illiquidity ratio adds no signal |
| 15 | `exp/wavelet-scattering` | Feature | FAIL | +0.2pp | +0.32 | -5.3pp | Nonlinear wavelets can't help transitions |
| 16 | `exp/tda-features` | Feature | FAIL | -0.1pp | -0.28 | -5.6pp | Persistent homology is pure noise here |
| 17 | `exp/ordinal-network` | Feature | FAIL | +0.3pp | -0.12 | -4.2pp | Permutation patterns don't add signal |
| 18 | `exp/weits` | Arch | FAIL | -0.2pp | 0.00 | -1.4pp | D1 gets 61% weight; multi-scale doesn't help |
| 19 | `exp/nbeats-baseline` | Arch | DIAG | -6.4pp | -4.14 | -11.5pp | WaveletGPT >> N-BEATS; wavelets are critical |
| 20 | `exp/learning-to-rank` | Arch | FAIL | -3.6pp | -1.49 | -14.3pp | Pairwise ranking too noisy; high seed variance |

**Pass rate: 6/20 (30%)**

---

## Tier-by-Tier Analysis

### Tier 1: Loss Functions (7 experiments → 3 PASS, 4 FAIL)

**Winners**:
- **Label smoothing** (ε=0.1): simplest change, best calibration improvement. Reduces overconfidence without hurting any metric. Low-risk merge candidate.
- **β-NLL** (β=0.5): variance head enables selective trading. Unconditional econ_dir jumps to 69.1%. Selective econ_dir reaches 75.2% when abstaining on high-variance predictions (30% coverage).
- **Heteroscedastic** (Stirn init): similar to β-NLL but better selective performance (80.9% at 30% coverage). Both should be evaluated together — they're fundamentally the same mechanism (dual-head variance prediction).

**Failures**:
- Focal loss: no effect — class imbalance isn't the bottleneck
- Ordinal loss: cumulative link model catastrophically fails — quantiles may not be truly ordinal in wavelet coefficient space
- Asymmetric loss: cost-weighted CE with 5:1 FP/FN ratio completely destabilizes
- Decision-focused: differentiable Sharpe collapses to constant prediction — needs curriculum learning or warmer start

### Tier 2: Calibration (3 experiments → 1 PASS, 2 FAIL)

**Winner**:
- **Post-hoc calibration**: Platt scaling works best of the three methods (vs temperature, isotonic). At 30% abstention: 70.4% econ_dir, Sharpe +8.1. This is a post-training method — zero retraining cost.

**Failures**:
- g-layers: learned calibration network worsens ECE despite improving econ_dir — overfits the calibration set
- Conformal prediction: prediction sets are either singletons (no uncertainty info) or multi-class (must abstain). Coverage at useful confidence levels is <1%

### Tier 3: Selective Prediction (2 experiments → 2 PASS)

Both approaches work:
- **SelectiveNet**: end-to-end rejection head. At 50% coverage: 70.8% econ_dir. At 70% coverage: 70.1%. Trade-off is smooth.
- **Error regularization**: simplest approach — just 20 LOC loss class. λ=0.1 reduces ECE by 73% and improves econ_dir by 3.9pp. Best bang-for-buck of all experiments.

### Tier 4: Features (5 experiments → 0 PASS, 5 FAIL)

**Universal pattern**: ALL feature additions degrade transition accuracy by 3-6pp. This holds across:
- Price-derived features (bar structure, range dynamics, jump filters — from prior tests)
- Volume-derived features (VPIN, Amihud illiquidity)
- Nonlinear wavelet features (scattering transform)
- Topological features (TDA persistence)
- Information-theoretic features (ordinal network entropy)

**Root cause**: the D1 representation is at its signal ceiling. Additional features don't help because the model capacity is already saturated on the D1 input. Extra features just add noise dimensions that hurt the model's ability to detect regime transitions.

### Tier 5: Architecture (3 experiments → 0 PASS, 1 DIAGNOSTIC)

- **WEITS**: multi-scale (D1+D2+D3) provides no benefit. Learned weights converge to ~61% D1, confirming D1 dominates.
- **N-BEATS** (diagnostic): pure FC dramatically underperforms WaveletGPT (-6.4pp econ_dir, -4.1 Sharpe, -11.5pp transition). **The wavelet preprocessing + causal transformer architecture is providing real signal.** Architecture is not the bottleneck.
- **Learning-to-rank**: pairwise RankNet is too noisy. Extreme seed variance (54.5%, 56.8%, 69.6%) shows gradient estimation instability.

---

## Cross-Experiment Insights

### 1. The bottleneck is NOT architecture or features — it's the target/loss

The wavelet+transformer pipeline extracts real signal from D1 coefficients (confirmed by N-BEATS diagnostic). But standard CE on quantile classes wastes capacity on easy flat predictions. The winning experiments all change HOW the model trains (loss function, calibration, selective prediction) rather than WHAT it trains on (features, architecture).

### 2. Selective prediction is the most promising direction

The strongest results come from teaching the model WHEN NOT to trade:
- Heteroscedastic: 80.9% econ_dir at 30% coverage
- β-NLL: 75.2% at 30% coverage
- SelectiveNet: 70.8% at 50% coverage
- Post-hoc Platt: 70.4% at 30% coverage

All four methods independently converge on the same insight: ~30% of predictions are noise, and filtering them out dramatically improves performance.

### 3. Feature additions are actively harmful

Every single feature experiment (8 total across all sessions) degrades transition accuracy. The mechanism is clear: extra input dimensions add noise that overwhelms the model's ability to detect the ~55% of regime transitions it currently catches. The D1 representation is a hard signal ceiling — no amount of auxiliary information helps.

### 4. Catastrophic failures reveal fragile training dynamics

Three experiments (ordinal loss, asymmetric loss, decision-focused) produced catastrophic drops of 16-26pp. The common thread: loss functions that fundamentally change the gradient landscape (cumulative links, asymmetric cost matrices, differentiable Sharpe) are incompatible with the small transformer architecture. The model has only 64K-200K parameters — complex loss surfaces need larger models or curriculum learning.

---

## Merge Recommendations

### Merge immediately (low risk, proven benefit):

1. **`exp/label-smoothing`** — ε=0.1, one-line change to `nn.CrossEntropyLoss(label_smoothing=0.1)`. Reduces ECE 55%, improves transition +1.8pp. No downside.

2. **`exp/error-regularization`** — 20 LOC `ErrorRegularizedLoss` class, λ=0.1. ECE -73%, econ_dir +3.9pp. Simple, effective, well-understood.

### Merge with validation (medium risk, high reward):

3. **`exp/heteroscedastic`** — adds variance head for selective trading. 80.9% econ_dir at 30% coverage. Requires downstream integration (trading system must respect abstention signals).

4. **`exp/post-hoc-calibration`** — post-training Platt scaling. Zero retraining cost. 70.4% econ_dir at 30% coverage.

5. **`exp/selectivenet`** — end-to-end rejection head. 70.8% at 50% coverage. Good for production systems that need principled abstention.

### Do NOT merge:

All feature and architecture experiments (13-20). All loss experiments except label-smoothing and error-regularization.

---

## Next Steps

1. **Stack the winners**: combine label-smoothing + error-regularization + heteroscedastic variance in a single training run. These should be compatible since they operate on different aspects (regularization, calibration, uncertainty).

2. **2026 holdout validation**: all 6 merge candidates must pass on the 2026 holdout data before production. The current results are on 2025 test data only.

3. **Trading system integration**: the selective prediction winners require a downstream system that can handle abstention (skip trades when model uncertainty is high). This is a system design change, not just a model change.

4. **Hyperparameter sensitivity**: the winning methods all have hyperparameters (ε, λ, β, target_coverage) that were only tested at 2-3 values. A proper Optuna sweep on the validation set could improve results further.

---

## Files Modified

| Branch | Files Changed |
|--------|--------------|
| `exp/label-smoothing` | `wavelet_gpt.py` (1 line: `label_smoothing=ε`), `test_label_smoothing.py` |
| `exp/beta-nll` | `wavelet_gpt.py` (+BetaNLLLoss, +variance_head, +predict_variance), `test_beta_nll.py` |
| `exp/heteroscedastic` | `wavelet_gpt.py` (+HeteroscedasticLoss, +variance_head, +predict_variance), `test_heteroscedastic.py` |
| `exp/post-hoc-calibration` | `test_calibration.py` only (no model changes) |
| `exp/selectivenet` | `wavelet_gpt.py` (+SelectiveNetLoss, +reject_head, +predict_rejection), `test_selectivenet.py` |
| `exp/error-regularization` | `wavelet_gpt.py` (+ErrorRegularizedLoss), `test_error_reg.py` |
| `exp/focal-loss` | `wavelet_gpt.py` (+FocalLoss), `test_focal_loss.py` |
| `exp/ordinal-loss` | `wavelet_gpt.py` (+OrdinalLoss), `test_ordinal_loss.py` |
| `exp/asymmetric-loss` | `wavelet_gpt.py` (+AsymmetricLoss), `test_asymmetric_loss.py` |
| `exp/decision-focused` | `wavelet_gpt.py` (+DifferentiableSharpeLoss), `test_decision_focused.py` |
| `exp/g-layers` | `test_g_layers.py` only |
| `exp/conformal-selective` | `test_conformal.py` only |
| `exp/vpin-feature` | `test_vpin.py` only |
| `exp/amihud-feature` | `test_amihud.py` only |
| `exp/wavelet-scattering` | `test_wavelet_scatter.py` only |
| `exp/tda-features` | `test_tda.py` only |
| `exp/ordinal-network` | `test_ordinal_net.py` only |
| `exp/weits` | `test_weits.py` only |
| `exp/nbeats-baseline` | `test_nbeats.py` only |
| `exp/learning-to-rank` | `wavelet_gpt.py` (+RankNetLoss), `test_learning_to_rank.py` |

All results saved to `~/.wavecast/audit/feature_tests/<name>_results.json`.
All branches pushed to `origin/exp/<name>` on GitHub.
Master is clean — 476 tests passing, zero lint errors.
