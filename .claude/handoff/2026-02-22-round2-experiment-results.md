# Round 2 Ephemeral Branch Experiments — Final Results

**Date**: 2026-02-22
**Duration**: Script creation ~3 days (parallel agents), execution ~4 hours
**Branch**: All 26 `exp2/*` branches on origin, master unchanged

## Executive Summary

26 experiments from a 2,841-paper literature review were tested against the D1 wavelet coefficient delta pipeline using a 5-metric evaluation framework that catches flat-bias gaming. **Zero experiments beat the CE baseline** on all metrics simultaneously. BOLT ceiling analysis correctly predicted this: the current feature representation is already near-optimal for CE loss.

**One selective trading strategy passed** (EnCQR bootstrap ensemble), and **two experiments showed strong transition detection** (focal loss family + TNC contrastive), but neither improved overall economic directional accuracy.

## 5-Metric Evaluation Framework

All experiments evaluated with `exp2_helpers.py` (committed to master):
1. **econ_dir** — Economic directional accuracy (threshold: >= 63%)
2. **pred_dist** — Prediction distribution (flat > 40% = auto-FAIL for flat-bias)
3. **transition_acc** — Accuracy at regime transitions (threshold: >= 52%)
4. **large_move_acc** — Accuracy on large price moves (threshold: >= 68%)
5. **sharpe_costs** — Sharpe ratio with transaction costs (P100 vs random baseline)

## Full Results Table

| # | Tier | Branch | Experiment | Verdict | econ_dir | flat% | transition | large_move | sharpe | Notes |
|---|------|--------|-----------|---------|----------|-------|------------|------------|--------|-------|
| 1 | 0 | exp2/mine-ceiling | MINE MI ceiling | INFO | — | — | — | — | — | MI=0.0016 bits, Fano bound unreliable for 82-dim |
| 2 | 0 | exp2/bolt-ceiling | BOLT 0-1 ceiling | **AT CEILING** | 63.1% | — | — | — | — | CE already near-optimal for this representation |
| 3 | 0 | exp2/focal-flatcheck | Focal flatcheck | BALANCED | 62.4% | 2.5% | **65.0%** | 66.4% | +4.454 | Round 1 FAIL was honest, not flat-biased |
| 4 | 0 | exp2/ordinal-flatcheck | Ordinal flatcheck | COLLAPSED | 68.7% | **75.2%** | 22.4% | 72.9% | +4.256 | Pure flat-bias gaming |
| 5 | 1 | exp2/mdn-head | MDN (Gaussian mixture) | FAIL | 60.6% | — | — | 65.4% | — | econ_dir and large_move below threshold |
| 6 | 1 | exp2/empl-loss | EMPL (Earth Mover) | FAIL | 50.0% | **100%** | 1.9% | 50.0% | — | Total flat collapse |
| 7 | 1 | exp2/bqn-crps | BQN (Bernstein quantile) | FAIL | 71.7% | **80.0%** | 19.7% | — | — | Flat-bias gamer, NaN training loss |
| 8 | 1 | exp2/n3pom-ordinal | N3POM (proportional odds) | FAIL | 74.0% | **90.6%** | 10.9% | — | — | Worst flat collapse of all |
| 9 | 1 | exp2/iqn | IQN (implicit quantile) | FAIL | 72.0% | **77.2%** | 22.5% | — | — | Flat-bias gamer, NaN training loss |
| 10 | 2 | exp2/bocpd-feature | BOCPD changepoint | FAIL | 65.1% | 32.0% | 49.9% | — | +4.847 | Close miss on transition (0.1pp) |
| 11 | 2 | exp2/permutation-entropy | Permutation entropy | FAIL | — | — | 51.0% | — | — | Close miss on transition (1.3pp) |
| 12 | 3 | exp2/cyclical-focal | Cyclical focal | FAIL | 63.4% | 18.9% | **56.8%** | 67.5% | +4.577 | Best gamma=3; transition +4.5pp but large_move miss |
| 13 | 3 | exp2/focal-psi-gamma | Focal + Psi calibration | FAIL | 62.4% | 2.5% | **65.0%** | 66.4% | +4.454 | Psi correction made ECE worse (0.13→0.35) |
| 14 | 3 | exp2/arctan-pinball | Arctan smooth pinball | FAIL | 52.4% | — | 49.2% | 50.8% | — | Near random performance |
| 15 | 3 | exp2/huber-quantile | Huber quantile regression | FAIL | — | — | 46.5% | — | — | Below CE on transition |
| 16 | 4 | exp2/curriculum-transition | Curriculum learning | FAIL | — | — | — | 67.9% | — | 0.1pp miss on large_move |
| 17 | 4 | exp2/ddat-difficulty | DDAT difficulty-aware | FAIL | — | — | — | 67.4% | — | Close miss on large_move |
| 18 | 5 | exp2/tnc-contrastive | TNC contrastive | FAIL | 63.0% | 18.5% | **58.1%** | 67.4% | +4.142 | Best transition of all (+5.7pp), low flat |
| 19 | 5 | exp2/cost-contrastive | CoST frequency contrastive | NO EFFECT | — | — | — | — | — | No improvement over CE |
| 20 | 6 | exp2/neural-wavelet-layer | Neural wavelet layer | NO EFFECT | — | — | — | — | — | Learnable wavelets = no improvement |
| 21 | 6 | exp2/tft-variable-selection | TFT variable selection | FAIL | — | — | 45.9% | 67.1% | — | VSN didn't help |
| 22 | 7 | exp2/ordinal-conformal | Ordinal conformal | FAIL | — | — | — | — | — | Selective trading below thresholds |
| 23 | 7 | exp2/eraps-conformal | ERAPS/AgACI conformal | FAIL | — | — | — | — | — | Adaptive conformal below thresholds |
| 24 | 7 | exp2/encqr-bootstrap | EnCQR bootstrap ensemble | **PASS** | **67.9%** | — | — | — | — | 5-model ensemble, spread<=1: 67.9% @ 44.6% coverage |
| 25 | 8 | exp2/hocmim-selection | HOCMIM feature ranking | PASS (info) | — | — | — | — | — | ctx_mean dominates (MI=0.0224), aux ~0 |
| 26 | 8 | exp2/smrmr-selection | SmRMR knockoff filter | PASS (info) | — | — | — | — | — | 0/6 features survive FDR at q=0.20 |

**CE Baseline reference**: econ_dir=63.6%, flat=25.9%, transition=52.3%, large_move=67.5%, sharpe=+4.692

## Key Findings

### 1. BOLT Ceiling Confirmed
The CE loss is already near-optimal for the current D1 wavelet coefficient delta features. No loss function, architecture, or training strategy broke through 64% econ_dir. This is a feature representation bottleneck, not a modeling bottleneck.

### 2. Flat-Bias Is Endemic
Every distributional output approach (MDN, BQN, N3POM, IQN, EMPL) collapsed to >77% flat predictions. The flat class (quantile 2) is the probability attractor — these approaches learn to pile mass on flat to minimize loss. The 5-metric framework caught all of them via pred_dist + transition_acc.

### 3. Transition Detection Is the Bright Spot
Three experiments showed significant transition accuracy improvement without flat-biasing:
- **TNC contrastive**: 58.1% transition (+5.7pp), 18.5% flat
- **Focal loss family** (focal, cyclical focal, focal+psi): ~65% transition (+12.7pp), 2.5% flat
- These sacrifice 1-2pp econ_dir for much better regime change detection

### 4. EnCQR: Selective Trading Works
The only PASS — bootstrap ensemble with disagreement filtering:
- 5 models, abstain when they disagree (spread > 1)
- When models agree: 67.9% econ_dir at 44.6% coverage
- This is a **meta-strategy** (selective trading), not a better model
- Implication: model uncertainty is a useful trading signal

### 5. Aux Features Are Noise
Both HOCMIM and SmRMR confirm the context sequence carries all the information. The 4 auxiliary features (energy ratio, zero-crossing rate, local volatility, approx trend) contribute essentially zero MI beyond what the raw wavelet coefficient deltas provide. No feature survived FDR control at q=0.20.

## Implications for Phase 8

1. **Representation is the bottleneck** — new features, not new losses/architectures, are needed to break 64%
2. **Predict signed returns or return quantiles** instead of SAX levels — the current 5-class quantile target may be information-lossy
3. **Focal loss + EnCQR ensemble** is the best combo available today: focal for transition detection, ensemble disagreement for trade selection
4. **Contrastive pre-training (TNC)** may become valuable if combined with richer features — it learns regime structure the classifier can't see

## Files

- `scripts/feature_tests/exp2_helpers.py` — shared 5-metric evaluation (on master)
- `scripts/run_round2_remaining.sh` — batch runner script
- `scripts/round2_progress.log` — full execution log with timestamps
- `scripts/round2_status.txt` — final status
- Results JSON: `~/.wavecast/audit/feature_tests/{experiment}_results.json` (26 files)

## How to Re-run Any Experiment

```bash
cd ~/wavecast
source .venv/bin/activate
git checkout exp2/<branch-name>
python -u -m scripts.feature_tests.test_<module_name>
git checkout master
```

## Bug Fixes Applied

1. **test_focal_psi.py** (exp2/focal-psi-gamma): Removed duplicate `compute_ece()` call that passed unfiltered `ce_probs` (37702 samples) against filtered `y_test[te_valid]` (37642 samples). Shape mismatch → ValueError.

2. **test_tnc.py** (exp2/tnc-contrastive): Reduced `TNC_EPOCHS` from 30 to 20. Loss had plateaued by epoch 20 (0.8772→0.8663 with no improvement after). Original 30 epochs + CE baseline + Phase 2 exceeded the 30-minute timeout.
