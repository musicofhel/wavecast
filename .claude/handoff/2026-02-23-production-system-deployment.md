# Handoff: Production System Deployment

**Date**: 2026-02-23
**From**: Strategic planning session (Claude chat)
**To**: Claude Code
**Priority**: Ship it — research phase is over

---

## Situation Summary

66 total experiments across 5 rounds proved the D1 representation (wavelet coefficient deltas + 4 aux features) is at its Bayes-optimal ceiling of ~64% economic directional accuracy. The production system (A2i) has been validated on 2026 out-of-sample data with strong results. No further model research is warranted without new data sources.

### Validated Production System: A2i

```
Model:    CE baseline WaveletGPT (D1 representation)
Location: ~/.wavecast/models/d1_augmented_v1/
Filter:   Large-move only (top tercile by predicted magnitude, Signal B: expected absolute return)
Sizing:   Flat (1x per trade)
Costs:    7 bps round-trip
```

### 2026 OOS Results (Jan 1 – Feb 22, 2026)

| Metric | A2i (production) | Unfiltered baseline |
|--------|-------------------|---------------------|
| Trades | 1,644 | 3,922 |
| Accuracy | **66.7%** | 62.2% |
| Sharpe | **+10.40** | +6.06 |
| Max drawdown | **-13.6%** | -19.5% |
| Expectancy/trade | **+0.339%** | +0.158% |
| Win rate | **60.9%** | 54.6% |
| Win/loss ratio | **1.48** | 1.40 |

### Key Findings That Shape the System

1. **Large-move filter is the primary edge**: +4.5pp accuracy, nearly 2x Sharpe vs unfiltered.
2. **Reversal filter does NOT generalize**: 78.9% reversal accuracy on 2025 test data collapsed to 63.8% on 2026 OOS. Reversal vs continuation accuracy is identical OOS (63.8% vs 64.2%). The reversal structure was overfit to the test set.
3. **B2i (reversal + large) loses to A2i**: 905 trades at 65.1% accuracy vs 1,644 trades at 66.7%. Reversal filter cuts trades in half while slightly hurting accuracy.
4. **The 64% ceiling is confirmed**: 66 experiments across loss functions, architectures, features, training strategies, calibration, decomposition variants, target representations, context lengths. Nothing breaks through.

---

## Task 1: Write `docs/PRODUCTION_SYSTEM.md`

Document the complete production system. This is the reference document for anyone running the system.

### Contents

```markdown
# WaveCast Production System

## System Definition

- **Model**: WaveletGPT with D1 representation (wavelet coefficient deltas + 4 auxiliary features)
- **Model path**: `~/.wavecast/models/d1_augmented_v1/`
- **Training period**: 2021-01-01 to 2025-12-31
- **Architecture**: embed_dim=64, num_layers=3, dropout=0.1, context_length=16
- **Input**: DWT level-1 detail coefficient deltas + [coeff_sign, magnitude_zscore, volatility_ratio, approx_direction]
- **Output**: 5-class quantile prediction (softmax probabilities over return bins)
- **Trading rule**: A2i — trade only when predicted magnitude is in top tercile
- **Position sizing**: Flat (1x)
- **Cost assumption**: 7 bps round-trip

## How the Magnitude Filter Works

1. Model outputs softmax probabilities over 5 quantile bins
2. Compute predicted magnitude: Signal B = Σ(prob_i × |bin_midpoint_i|)
3. Rank all predictions by Signal B
4. Trade only the top tercile (largest predicted moves)
5. Direction = argmax of softmax → map to up/flat/down → trade direction
6. Flat predictions are always skipped (no trade)

## Performance

### 2026 Out-of-Sample (Jan 1 – Feb 22)
- Trades: 1,644 (20 tickers × 8 weeks)
- Accuracy on trades taken: 66.7%
- Annualized Sharpe (after 7bps costs): +10.40
- Max drawdown: -13.6%
- Per-trade expectancy: +0.339%
- Win rate: 60.9%
- Avg win / avg loss: 1.48

### 2025 Test (reference)
- Trades: 12,451
- Accuracy: 63.2%
- Sharpe: +8.40
- Max drawdown: -18.3%
- Expectancy/trade: +0.193%

## What Doesn't Work (Proven by 66 Experiments)

- Reversal filtering (overfit to test set, doesn't generalize)
- Any aux feature additions (10/10 degraded transition accuracy)
- Alternative loss functions (13/15 either gamed metrics or failed)
- Architecture changes (5/5 failed)
- Distributional outputs (5/5 collapsed)
- Longer context windows (hurt transition accuracy)
- Alternative decompositions (fracdiff, range-DWT, crossscale, MODWT — all failed or marginal)
- Fine-grained targets (7/11 class — interesting for transition but lower per-trade expectancy)
- Regression targets (best transition accuracy but lower per-trade expectancy)
- Ensemble P1+P2 combination (P1 too weak, corrupts P2 signal)

## Running the System

See forward test scripts for live execution.
```

---

## Task 2: Update Forward Test to Use A2i Rules

The existing forward test (`scripts/forward_test_d1.py` and `src/wavecast/forward/runner.py`) logs raw predictions. It needs to be updated to:

1. **Save softmax probabilities** alongside each prediction
2. **Compute predicted magnitude** (Signal B) for each prediction
3. **Flag whether A2i would trade** this prediction (is it in the top tercile by magnitude?)
4. **Track separate metrics** for A2i-filtered vs unfiltered predictions

### Implementation

#### Modify `ForwardPrediction` or extend it

The forward prediction dataclass (in `src/wavecast/forward/types.py`) needs new fields:

```python
@dataclass
class ForwardPrediction:
    # ... existing fields ...

    # New fields for production system
    softmax_probs: list[float] | None = None      # 5 probabilities
    predicted_magnitude: float | None = None        # Signal B value
    magnitude_tercile: str | None = None            # "large", "medium", "small"
    a2i_trade: bool = False                         # Would A2i take this trade?
```

#### Modify `ForwardTestRunner.run_once()`

After generating predictions, add:

```python
# After getting softmax probs from model
magnitude = compute_signal_b(probs, bin_midpoints)  # Signal B
# Determine tercile thresholds from historical predictions or training data
# Flag prediction as a2i_trade if magnitude >= large_tercile_threshold
```

**Critical**: The magnitude tercile threshold must come from training data, NOT from the current batch of predictions. Load the threshold from a file saved during the PnL simulation (it should be in `~/.wavecast/audit/production/pnl_simulation.json` under the magnitude signal section).

#### Modify `ForwardTestTracker` metrics

Track two sets of rolling metrics:
- `metrics_all`: accuracy, directional accuracy, PnL on ALL predictions (existing)
- `metrics_a2i`: accuracy, directional accuracy, PnL on A2i-filtered predictions only (new)

#### Modify `ForwardTestSummary`

Add `a2i_trades`, `a2i_accuracy`, `a2i_pnl`, `a2i_sharpe` fields.

#### Modify the report generator

`generate_forward_report()` should print both unfiltered and A2i-filtered metrics side by side, matching the format from the 2026 validation.

### Where to find existing code

- Forward test types: `src/wavecast/forward/types.py`
- Forward test runner: `src/wavecast/forward/runner.py` — has `_build_d1_pipeline_context()` for D1 models
- Forward test tracker: `src/wavecast/forward/tracker.py`
- Forward test report: `src/wavecast/forward/report.py`
- Forward test config: `src/wavecast/forward/config.py`
- Forward test CLI: `src/wavecast/cli/commands/forward.py`
- Convenience script: `scripts/forward_test_d1.py`

### Where to find magnitude signal implementation

The PnL simulation script (`scripts/production/pnl_simulation.py`) computes Signal B. Reuse that logic:

```python
def compute_signal_b(probs, bin_midpoints):
    """Expected absolute return from softmax distribution."""
    return sum(p * abs(m) for p, m in zip(probs, bin_midpoints))
```

The `bin_midpoints` come from `quantile_boundaries.json` in the model directory. They're the midpoints of each quantile bin computed from training data.

---

## Task 3: Create Scheduled Forward Test Script

### `scripts/run_forward_test.sh`

A simple bash script that runs the forward test, suitable for cron or manual execution after market hours.

```bash
#!/bin/bash
# Run WaveCast forward test - execute after market close
# Cron: 0 21 * * 1-5 /home/musicofhel/wavecast/scripts/run_forward_test.sh

cd /home/musicofhel/wavecast
source .venv/bin/activate

echo "$(date): Starting forward test cycle"

# Run one cycle: resolve pending, generate new predictions
python scripts/forward_test_d1.py --run-once 2>&1 | tee -a ~/.wavecast/forward_tests/d1_forward_v1/run.log

# Print current status
python scripts/forward_test_d1.py --report 2>&1 | tee -a ~/.wavecast/forward_tests/d1_forward_v1/run.log

echo "$(date): Forward test cycle complete"
```

---

## Task 4: Update Project Documentation

### `ROADMAP.md`

Add Phase 12 and Production Deployment as completed:

```markdown
### Phase 12: Representation Ceiling Confirmation (v0.12.0)
10 representation change experiments: fractional differencing, fine-grained quantiles (7/11 class),
extended context (32/48/64), cross-scale features, range-DWT, signed return regression,
auxiliary transition head.

- [x] Wave 1: fracdiff (FAIL), finegrain_7 (INTERESTING), finegrain_11 (INTERESTING), context_32 (FAIL), context_48 (FAIL), context_64 (OOM)
- [x] Wave 2: crossscale (FAIL), range_dwt (FAIL)
- [x] Wave 3: regression (INTERESTING), transition_head (INTERESTING)
- [x] Master termination: 0 passes across 3 waves. 64% ceiling confirmed across 66 total experiments.
- [x] Production system validated: A2i (large-move filter) on 2026 OOS — 66.7% accuracy, Sharpe +10.4

### Production Deployment
- [x] PnL simulation: 8-config comparison, A2i wins
- [x] 2026 OOS validation: MARGINAL (accuracy 66.7% < 70% threshold, but Sharpe +10.4 and expectancy +0.34% both pass)
- [x] Reversal filter invalidated OOS (78.9% → 63.8%, no generalization)
- [x] Production system documented in docs/PRODUCTION_SYSTEM.md
- [x] Forward test updated with A2i magnitude filter and dual-track metrics
```

### `CLAUDE.md`

Add under "Pipeline 3: Signal Generation & Backtesting":

```markdown
### Production Trading System (A2i)
```
WaveletGPT.predict_proba() → compute_signal_b() → magnitude_filter(top_tercile) → flat_sizing → trade
```
- Model: d1_augmented_v1 (trained 2021-2025, embed_dim=64, 3 layers)
- Trade filter: top tercile by predicted magnitude (Signal B = expected absolute return from softmax)
- 2026 OOS: 66.7% accuracy, +10.4 Sharpe, +0.339% per-trade expectancy, 1,644 trades in 8 weeks
- Reversal filter tested and REJECTED (78.9% on 2025 collapsed to 63.8% on 2026)
```

### `ARCHITECTURE.md`

Add a "Production System" section after the existing pipeline descriptions.

---

## Task 5: Write Phase 12 Results Document

### `PHASE12_RESULTS.md` (in project root, alongside PHASE3_RESULTS.md)

Comprehensive results document covering:

1. **Context**: 46 prior experiments proved BOLT ceiling at 63.1%, CE at 63.6%
2. **10 representation experiments** with full results table (from the Phase 12 results document provided above)
3. **PnL simulation results**: 8-config comparison showing A2i as winner
4. **2026 OOS validation**: A2i results, B2i comparison, reversal filter failure
5. **Cross-tab analysis**: The 2x2 (magnitude x transition) that revealed reversal detection structure on 2025 but didn't generalize
6. **Lessons learned**: Reversal structure overfit, large-move filter is robust, fine-grained and regression are library tools for transition-focused strategies
7. **Production system definition**: Complete spec
8. **What's next**: Forward testing accumulation, potential EnCQR overlay, new data sources as the only remaining research lever

Use the same format as `PHASE3_RESULTS.md` — tables, honest assessment sections, known limitations.

Data sources for this document:
- `~/.wavecast/audit/exp3/*_results.json` — all 10 representation experiment results
- `~/.wavecast/audit/production/pnl_simulation.json` — 8-config PnL comparison
- `~/.wavecast/audit/production/2026_validation.json` — A2i and B2i OOS results (also uploaded to this conversation)
- The Phase 12 results summary document provided in this conversation (document index 8)
- The cross-tab results provided in this conversation (document index 10)

---

## Task 6 (Optional): EnCQR Overlay Test

If time permits after the above, test the EnCQR selective overlay on top of A2i:

1. Train 5 CE models with seeds [42, 43, 44, 45, 46] on the full 2021-2025 dataset
2. For each prediction in the 2026 OOS data, compute all 5 models' predicted classes
3. Ensemble spread = max(predicted_class) - min(predicted_class)
4. A2i+EnCQR: only trade when A2i says trade AND spread <= 1
5. Report accuracy, Sharpe, trades on this combined filter vs A2i alone

The Round 2 EnCQR result was 67.9% at 44.6% coverage on 2025 data. Stacking it with A2i might further concentrate on high-quality trades. But given the reversal filter's failure to generalize, be skeptical — validate on 2026 before declaring it works.

Script: `scripts/production/test_encqr_overlay.py`

This is lower priority than Tasks 1-5. The core production system (A2i without EnCQR) is already validated and ready.

---

## Files Summary

| File | Type | Task | Purpose |
|------|------|------|---------|
| `docs/PRODUCTION_SYSTEM.md` | NEW | 1 | Production system reference document |
| `src/wavecast/forward/types.py` | MODIFY | 2 | Add softmax_probs, predicted_magnitude, a2i_trade fields |
| `src/wavecast/forward/runner.py` | MODIFY | 2 | Compute magnitude signal, flag A2i trades |
| `src/wavecast/forward/tracker.py` | MODIFY | 2 | Dual-track metrics (all vs A2i) |
| `src/wavecast/forward/report.py` | MODIFY | 2 | Side-by-side unfiltered vs A2i metrics |
| `scripts/forward_test_d1.py` | MODIFY | 2 | Pass magnitude threshold to runner |
| `scripts/run_forward_test.sh` | NEW | 3 | Cron-ready forward test runner |
| `ROADMAP.md` | MODIFY | 4 | Add Phase 12 + Production sections |
| `CLAUDE.md` | MODIFY | 4 | Add production system description |
| `ARCHITECTURE.md` | MODIFY | 4 | Add production system section |
| `PHASE12_RESULTS.md` | NEW | 5 | Comprehensive Phase 12 results |
| `scripts/production/test_encqr_overlay.py` | NEW | 6 | Optional EnCQR overlay test |

---

## What NOT To Do

- **No more model experiments.** The ceiling is proven across 66 experiments. The model is done.
- **No retraining.** The `d1_augmented_v1` model is the production model.
- **No new features.** All 10 feature additions degraded performance.
- **No reversal filtering.** It doesn't generalize.
- **Don't touch the evaluation framework** (`scripts/exp3/evaluate_representation.py`). It's battle-tested across 66 experiments. Archive it.
- **Don't change the 7bps cost assumption** without researching actual execution costs.

---

## Context Files for Reference

These files exist and contain the full research history:

| File | Contents |
|------|----------|
| `~/.wavecast/audit/exp3/*_results.json` | All 10 Phase 12 experiment results |
| `~/.wavecast/audit/production/pnl_simulation.json` | 8-config PnL simulation |
| `~/.wavecast/audit/production/2026_validation.json` | A2i and B2i OOS validation |
| `~/.wavecast/audit/feature_tests/*_results.json` | 46 prior experiment results |
| `.claude/handoff/2026-02-22-all-46-experiments-summary.md` | Round 1+2 experiment summary |
| `.claude/handoff/2026-02-17-ephemeral-branch-results.md` | Round 1 results |
| `.claude/handoff/2026-02-22-round2-experiment-results.md` | Round 2 results |
| `scripts/exp3/baseline_lock.json` | Frozen CE baseline reference |

---

## The Bottom Line

The research is done. 66 experiments, 3 rounds, 10 months of hourly data. The model works: 66.7% on large predicted moves, Sharpe 10.4 after costs, validated on unseen 2026 data. Ship the production system, set up forward testing, accumulate live performance data. The next genuine research lever is new data sources (cross-asset, multi-timeframe, non-price data), not more experiments on the same hourly close prices.
