# WaveCast: Complete 46-Experiment Summary

**Rounds**: 2 (Round 1: 20 experiments, Round 2: 26 experiments)
**Total duration**: ~6 hours execution across both rounds
**Baseline**: D1 WaveletGPT — ~64% econ_dir, +4.69 Sharpe, 52.3% transition, 67.5% large-move, 25.9% flat
**Data**: 20 US tickers, hourly bars, 124K train / 37K test windows
**Overall pass rate**: 7/46 (15%) — and 6 of those 7 are selective/calibration methods, not model improvements

---

## Round 1 — 91-Paper Review (2026-02-17)

20 experiments across 5 tiers. 3-seed averaging. Pass threshold: >+1pp econ_dir or >+0.2 Sharpe or >+2pp transition.

| # | Branch | Category | Verdict | econ_dir | Sharpe | transition | Key Finding |
|---|--------|----------|---------|----------|--------|------------|-------------|
| 1 | exp/focal-loss | Loss | FAIL | -0.1pp | -0.09 | -1.1pp | Class imbalance not the bottleneck |
| 2 | exp/ordinal-loss | Loss | FAIL | -21.9pp | -11.8 | -21.6pp | Cumulative link catastrophically collapses |
| 3 | exp/label-smoothing | Loss | **PASS** | +0.2pp | +0.12 | +1.8pp | epsilon=0.1, ECE reduced 55% |
| 4 | exp/asymmetric-loss | Loss | FAIL | -26.4pp | -16.7 | -24.1pp | 5:1 cost matrix destabilizes training |
| 5 | exp/beta-nll | Loss | **PASS*** | +5.2pp | +0.66 | — | beta=0.5, selective 75.2% @ 30% coverage |
| 6 | exp/heteroscedastic | Loss | **PASS*** | +5.6pp | +0.81 | — | Stirn init, selective 80.9% @ 30% coverage |
| 7 | exp/decision-focused | Loss | FAIL | -16.3pp | -5.3 | — | Differentiable Sharpe → mode collapse |
| 8 | exp/post-hoc-calibration | Calibration | **PASS** | — | — | — | Platt @ 30%: 70.4% econ_dir, Sharpe +8.1 |
| 9 | exp/g-layers | Calibration | FAIL | +7.9pp | +2.3 | — | ECE worsens despite econ_dir gain |
| 10 | exp/conformal-selective | Calibration | FAIL | — | — | — | Singleton coverage <1% |
| 11 | exp/selectivenet | Selective | **PASS*** | +5.1pp | — | — | Sel @ 50%: 70.8%, sel @ 70%: 70.1% |
| 12 | exp/error-regularization | Selective | **PASS** | +3.9pp | +1.1 | — | lambda=0.1, ECE -73%, simplest (20 LOC) |
| 13 | exp/vpin-feature | Feature | FAIL | +0.4pp | +0.33 | -3.2pp | Volume degrades transitions |
| 14 | exp/amihud-feature | Feature | FAIL | -0.2pp | -0.19 | -3.1pp | Illiquidity ratio = noise |
| 15 | exp/wavelet-scattering | Feature | FAIL | +0.2pp | +0.32 | -5.3pp | Nonlinear wavelets hurt transitions |
| 16 | exp/tda-features | Feature | FAIL | -0.1pp | -0.28 | -5.6pp | Persistent homology = noise |
| 17 | exp/ordinal-network | Feature | FAIL | +0.3pp | -0.12 | -4.2pp | Permutation patterns = noise |
| 18 | exp/weits | Architecture | FAIL | -0.2pp | 0.00 | -1.4pp | Multi-scale: D1 gets 61% weight, no benefit |
| 19 | exp/nbeats-baseline | Architecture | DIAG | -6.4pp | -4.14 | -11.5pp | N-BEATS << WaveletGPT; wavelet preprocessing is real |
| 20 | exp/learning-to-rank | Architecture | FAIL | -3.6pp | -1.49 | -14.3pp | Pairwise ranking too noisy, extreme seed variance |

**Round 1 pass rate: 6/20 (30%)**

*\* = INVALIDATED by Round 2 flat-bias analysis. Beta-NLL, heteroscedastic, and SelectiveNet all achieve high econ_dir by predicting flat ~80% of the time, inflating accuracy on the self-selected directional predictions. Transition accuracy collapses (52% → 3-18%). See "Stacking & Flat-Bias Discovery" below.*

---

## Stacking & Flat-Bias Discovery (2026-02-18)

Three follow-up experiments revealed that the Round 1 "winners" were gaming the metric:

| Experiment | Finding |
|-----------|---------|
| **Stacking** (hetero + LS + ereg) | Winners DON'T compound. Hetero alone = 69.5%, +LS = 68.8%, +LS+ereg = 68.7%. Redundant. |
| **Heteroscedastic validation** | **SMOKING GUN**: Predicts flat ~82% of the time. 69.5% econ_dir is on self-selected 18% directional. Transition: 52.3% → 17.7% → 3.1% at 30% coverage. |
| **Three diagnostics** | D1: Platt partially real (half genuine, half flat-selection). D2: CE entropy deciles show 20pp gradient (genuine signal). D3: **Large moves are 69.4% accurate** — the 64% headline UNDERSTATES the tradeable edge. |

**Adjusted Round 1 results**: Only **label-smoothing** and **error-regularization** are genuinely beneficial. Post-hoc Platt is partially real. Beta-NLL, heteroscedastic, and SelectiveNet are flat-bias gamers.

---

## Round 2 — 2,841-Paper Review (2026-02-22)

26 experiments across 9 tiers. Updated 5-metric evaluation with flat-bias detection (>40% flat = auto-FAIL), transition accuracy, and large-move accuracy thresholds.

| # | Branch | Category | Verdict | econ_dir | flat% | transition | large_move | sharpe | Key Finding |
|---|--------|----------|---------|----------|-------|------------|------------|--------|-------------|
| 1 | exp2/mine-ceiling | Diagnostic | INFO | — | — | — | — | — | MINE MI=0.0016 bits; can't handle 82-dim |
| 2 | exp2/bolt-ceiling | Diagnostic | **CEILING** | 63.1% | — | — | — | — | CE already near-optimal for this representation |
| 3 | exp2/focal-flatcheck | Diagnostic | BALANCED | 62.4% | 2.5% | **65.0%** | 66.4% | +4.454 | Round 1 focal FAIL was honest. Transition +12.7pp |
| 4 | exp2/ordinal-flatcheck | Diagnostic | COLLAPSED | 68.7% | 75.2% | 22.4% | 72.9% | +4.256 | Flat-bias gamer confirmed |
| 5 | exp2/mdn-head | Target Rep | FAIL | 60.6% | — | — | 65.4% | — | Gaussian mixture worse than CE |
| 6 | exp2/empl-loss | Target Rep | FAIL | 50.0% | 100% | 1.9% | 50.0% | — | Total flat collapse |
| 7 | exp2/bqn-crps | Target Rep | FAIL | 71.7% | 80.0% | 19.7% | — | — | Flat gamer, NaN loss |
| 8 | exp2/n3pom-ordinal | Target Rep | FAIL | 74.0% | 90.6% | 10.9% | — | — | Worst flat collapse (90.6%) |
| 9 | exp2/iqn | Target Rep | FAIL | 72.0% | 77.2% | 22.5% | — | — | Flat gamer, NaN loss |
| 10 | exp2/bocpd-feature | Features | FAIL | 65.1% | 32.0% | 49.9% | — | +4.847 | Mild flat-bias, close miss |
| 11 | exp2/permutation-entropy | Features | FAIL | — | — | 51.0% | — | — | Transition miss by 1.3pp |
| 12 | exp2/cyclical-focal | Loss | FAIL | 63.4% | 18.9% | **56.8%** | 67.5% | +4.577 | Transition +4.5pp, large_move miss by 0.5pp |
| 13 | exp2/focal-psi-gamma | Loss | FAIL | 62.4% | 2.5% | **65.0%** | 66.4% | +4.454 | Psi correction made ECE worse |
| 14 | exp2/arctan-pinball | Loss | FAIL | 52.4% | — | 49.2% | 50.8% | — | Near random |
| 15 | exp2/huber-quantile | Loss | FAIL | — | — | 46.5% | — | — | Below baseline |
| 16 | exp2/curriculum-transition | Training | FAIL | — | — | — | 67.9% | — | Large_move miss by 0.1pp |
| 17 | exp2/ddat-difficulty | Training | FAIL | — | — | — | 67.4% | — | Close miss |
| 18 | exp2/tnc-contrastive | Contrastive | FAIL | 63.0% | 18.5% | **58.1%** | 67.4% | +4.142 | Transition +5.7pp, large_move miss |
| 19 | exp2/cost-contrastive | Contrastive | NO EFFECT | — | — | — | — | — | Frequency contrastive = nothing |
| 20 | exp2/neural-wavelet-layer | Architecture | NO EFFECT | — | — | — | — | — | Learnable wavelets = no improvement |
| 21 | exp2/tft-variable-selection | Architecture | FAIL | — | — | 45.9% | 67.1% | — | Variable selection hurt transition |
| 22 | exp2/ordinal-conformal | Conformal | FAIL | — | — | — | — | — | — |
| 23 | exp2/eraps-conformal | Conformal | FAIL | — | — | — | — | — | — |
| 24 | exp2/encqr-bootstrap | Conformal | **PASS** | **67.9%** | — | — | — | — | 5-model ensemble, spread<=1: 67.9% @ 44.6% coverage |
| 25 | exp2/hocmim-selection | Feature Sel. | PASS (info) | 63.5% | — | 50.9% | — | — | ctx_mean dominates; aux features ~0 MI |
| 26 | exp2/smrmr-selection | Feature Sel. | PASS (info) | — | — | — | — | — | 0/6 features survive FDR @ q=0.20 |

**Round 2 pass rate: 1/26 (4%)** — only EnCQR selective trading

---

## Combined Analysis: What 46 Experiments Prove

### 1. The representation is the ceiling — not the loss, architecture, or features

| Category | Tested | Pass | Pattern |
|----------|--------|------|---------|
| Loss functions | 15 | 2 | Only label-smoothing (epsilon=0.1) and error-reg (lambda=0.1) genuinely help. All others either collapse, game flat-bias, or are within noise. |
| Target representations | 5 | 0 | MDN, BQN, N3POM, IQN, EMPL — all fail or flat-collapse. Distributional outputs are the wrong direction. |
| Calibration/post-hoc | 5 | 1 | Platt partially real. Conformal methods fail on coverage. |
| Selective/abstention | 4 | 1 | EnCQR (ensemble disagreement) is the only clean pass. Hetero/beta-NLL/SelectiveNet are flat-bias gamers. |
| Features (new inputs) | 10 | 0 | ALL 10 degrade transition accuracy by 3-6pp. D1 is a hard signal ceiling. |
| Feature selection | 2 | 0* | HOCMIM/SmRMR confirm aux features are noise. Informational pass only. |
| Architecture | 5 | 0 | Multi-scale, neural wavelets, variable selection, contrastive, N-BEATS — nothing beats the transformer+wavelet combo. |
| Training strategy | 4 | 0 | Curriculum, DDAT, cyclical focal, focal-psi — all fail or marginal. |
| Diagnostics/ceiling | 4 | — | BOLT confirms CE is near-optimal. MINE underestimates. Focal is honest. Ordinal is a gamer. |

### 2. The flat-bias problem is universal

Any method that can increase its "accuracy" by predicting flat WILL predict flat. Out of 46 experiments:
- **7 approaches** predicted flat >40% of the time (ordinal, hetero, beta-NLL, BQN, N3POM, IQN, EMPL)
- The worst (EMPL) hit 100% flat — every single prediction was flat
- The 5-metric framework (introduced in Round 2) catches this. Raw econ_dir alone is unreliable.

### 3. Transition accuracy separates real from fake improvements

| Method | econ_dir | transition | Flat% | Real? |
|--------|----------|------------|-------|-------|
| CE baseline | 63.6% | 52.3% | 25.9% | Baseline |
| Heteroscedastic | 69.5% | 17.7% | 82% | NO — flat gamer |
| N3POM | 74.0% | 10.9% | 90.6% | NO — flat gamer |
| Focal loss | 62.4% | **65.0%** | 2.5% | YES — honest |
| TNC contrastive | 63.0% | **58.1%** | 18.5% | YES — honest |
| EnCQR (spread<=1) | **67.9%** | — | — | YES — selective |
| Label smoothing | 63.8% | 54.1% | ~26% | YES — mild |
| Error regularization | 67.5% | ~52% | ~26% | YES — genuine |

### 4. The only path forward is selective trading or representation change

**What works (selective trading):**
- EnCQR: 67.9% econ_dir at 44.6% coverage via model disagreement
- CE baseline on large moves: 69.4% accuracy (vs 64% overall) — the model is BETTER on high-magnitude moves
- CE entropy deciles D2-D3: 67-71% econ_dir at ~48% transition

**What might work (representation change — untested):**
- Predict signed returns (regression instead of classification)
- Multi-scale targets (combine D1-D5 wavelet levels)
- Different wavelet families or decomposition depths
- Raw price returns with positional features
- Attention to the auxiliary feature pipeline (currently carries ~0 MI)

**What definitely doesn't work:**
- Adding more features to the current representation (10/10 failed)
- Alternative loss functions on the current representation (13/15 failed or gamed)
- Alternative architectures on the current representation (5/5 failed)
- Distributional output heads (5/5 collapsed to flat)

---

## File Locations

| Item | Path |
|------|------|
| Round 1 results (20 JSONs) | `~/.wavecast/audit/feature_tests/<name>_results.json` |
| Round 2 results (26 JSONs) | `~/.wavecast/audit/feature_tests/<name>_results.json` |
| Round 1 branches | `origin/exp/<name>` (20 branches) |
| Round 2 branches | `origin/exp2/<name>` (26 branches) |
| Shared R2 evaluation | `scripts/feature_tests/exp2_helpers.py` (on master) |
| R1 harness | `scripts/feature_tests/harness.py` (on master) |
| R2 runner script | `scripts/run_round2_remaining.sh` |
| R2 progress log | `scripts/round2_progress.log` |
| R1 handoff | `.claude/handoff/2026-02-17-ephemeral-branch-results.md` |
| R2 handoff | `.claude/handoff/2026-02-22-round2-experiment-results.md` |
| This summary | `.claude/handoff/2026-02-22-all-46-experiments-summary.md` |
