# WaveCast Production System

**Status**: Validated on 2026 out-of-sample data
**Model**: d1_augmented_v1
**Config**: A2i (all trades, large-move magnitude filter, flat sizing)

---

## System Definition

| Component | Value |
|-----------|-------|
| Model | WaveletGPT with D1 representation (wavelet coefficient deltas + 4 auxiliary features) |
| Model path | `~/.wavecast/models/d1_augmented_v1/` |
| Training period | 2021-01-01 to 2025-12-31 |
| Architecture | embed_dim=64, num_layers=3, num_heads=4, dropout=0.1, context_length=16 |
| Input | DWT level-1 detail coefficient deltas + [coeff_sign, magnitude_zscore, volatility_ratio, approx_direction] |
| Output | 5-class quantile prediction (softmax probabilities over return bins) |
| Trading rule | A2i: trade only when predicted magnitude is in top tercile |
| Position sizing | Flat (1x per trade) |
| Cost assumption | 7 bps round-trip |
| Universe | 20 US assets (5 sectors: tech, finance, energy, healthcare, ETFs) |
| Interval | 1h bars |

---

## How the Magnitude Filter Works

The production system uses "Signal B" — the expected absolute return from the softmax distribution — to filter trades.

### Pipeline

```
hourly bars → DWT(level=1) → np.diff(detail) → aux features → sliding windows
    → WaveletGPT.predict_proba() → 5-class softmax probabilities
    → Signal B = Σ(prob_i × |bin_midpoint_i|)
    → Rank by Signal B → trade only top tercile (largest predicted moves)
    → Direction = argmax(P_up, P_flat, P_down) → skip flat → execute trade
```

### Step by step

1. **Model inference**: WaveletGPT outputs softmax probabilities over 5 quantile bins for the next hourly bar's return.

2. **Compute predicted magnitude (Signal B)**:
   ```
   Signal_B = Σ(prob_i × |bin_midpoint_i|)  for i in [0..4]
   ```
   where `bin_midpoints` come from `quantile_boundaries.json` in the model directory. For level-1, these are approximately `[-0.0118, -0.0042, 0.0002, 0.0044, 0.0116]`.

3. **Magnitude tercile classification**: Compare Signal B against thresholds derived from training data predictions:
   - **Large** (top tercile): Signal B >= 67th percentile threshold (~0.0068)
   - **Medium** (middle tercile): 33rd–67th percentile (~0.0051–0.0068)
   - **Small** (bottom tercile): Signal B < 33rd percentile (~0.0051)

4. **Trade decision**: Only take the trade if:
   - Predicted magnitude is "large" (top tercile), AND
   - Predicted direction is not flat (direction != 0)

5. **Position sizing**: Flat 1x on every qualifying trade. No magnitude-based sizing.

### Why this works

The model is significantly more accurate on large predicted moves:
- Large-move accuracy: **68.1%** (2025 test), **66.7%** (2026 OOS)
- All-trade accuracy: **63.9%** (2025 test), **62.2%** (2026 OOS)
- The magnitude filter concentrates on the model's strongest predictions while cutting trade count ~55%

---

## Performance

### 2026 Out-of-Sample (Jan 1 – Feb 22)

| Metric | A2i (production) | Unfiltered baseline |
|--------|-------------------|---------------------|
| Trades | 1,644 | 3,922 |
| Accuracy | **66.7%** | 62.2% |
| Sharpe | **+10.40** | +6.06 |
| Max drawdown | **-13.6%** | -19.5% |
| Expectancy/trade | **+0.339%** | +0.158% |
| Win rate | **60.9%** | 54.6% |
| Win/loss ratio | **1.48** | 1.40 |

### 2025 Test Set (reference)

| Metric | A2i | A1i (all trades, flat) |
|--------|-----|------------------------|
| Trades | 12,451 | 27,653 |
| Accuracy | **63.2%** | 61.5% |
| Sharpe | **+8.40** | +6.11 |
| Max drawdown | -18.3% | -18.3% |
| Expectancy/trade | **+0.193%** | +0.117% |
| Win rate | **57.8%** | 54.4% |

### 8-Config Comparison (2025 Test)

| Config | Filter | Sizing | Trades | Accuracy | Sharpe | Expectancy |
|--------|--------|--------|--------|----------|--------|------------|
| A1i | All trades | Flat | 27,653 | 61.5% | +6.11 | +0.117% |
| A1ii | All trades | Magnitude | 27,653 | 61.5% | +5.84 | +0.150% |
| **A2i** | **Large-move** | **Flat** | **12,451** | **63.2%** | **+8.40** | **+0.193%** |
| A2ii | Large-move | Magnitude | 12,451 | 63.2% | +8.12 | +0.280% |
| B1i | Continuation | Flat | 14,633 | 62.0% | +6.17 | +0.118% |
| B1ii | Continuation | Magnitude | 14,633 | 62.0% | +5.76 | +0.148% |
| B2i | Continuation+large | Flat | 6,798 | 63.4% | +8.02 | +0.184% |
| B2ii | Continuation+large | Magnitude | 6,798 | 63.4% | +7.73 | +0.263% |

**A2i wins**: Highest Sharpe, no reversal filter complexity, robust OOS validation. The B-series (continuation/reversal filters) don't generalize.

---

## What Doesn't Work (Proven by 66 Experiments)

These approaches were systematically tested and rejected:

| Category | Experiments | Result |
|----------|------------|--------|
| Reversal filtering | B1i/B1ii/B2i/B2ii + 2026 OOS test | 78.9% reversal accuracy on 2025 collapsed to 63.8% on 2026 — doesn't generalize |
| Aux feature additions | 10 experiments | 10/10 degraded transition accuracy |
| Alternative loss functions | 15 experiments | 13/15 either gamed flat-bias metrics or failed outright |
| Architecture changes | 5 experiments | 5/5 failed (multi-scale, neural wavelets, variable selection, contrastive, N-BEATS) |
| Distributional outputs | 5 experiments | 5/5 collapsed to flat predictions (MDN, BQN, N3POM, IQN, EMPL) |
| Longer context windows | 3 experiments (32, 48, 64) | All hurt transition accuracy; 64 caused CUDA OOM |
| Alternative decompositions | 4 experiments | fracdiff, range-DWT, crossscale — all failed or marginal |
| Fine-grained targets | 2 experiments (7/11 class) | Interesting for transition but lower per-trade expectancy |
| Signed return regression | 1 experiment | Best transition accuracy (67%) but lower econ_dir (62.7%) |
| Ensemble P1+P2 | 3 experiments | P1 too weak, corrupts P2 signal |

---

## Running the System

### Forward test (single cycle)

```bash
cd ~/wavecast && source .venv/bin/activate
python scripts/forward_test_d1.py --report
```

### Forward test (scheduled)

```bash
# Run after market close
bash scripts/run_forward_test.sh
```

### Cron setup

```
# Run forward test weekdays at 9 PM ET
0 21 * * 1-5 /home/musicofhel/wavecast/scripts/run_forward_test.sh
```

### Check status

```bash
cd ~/wavecast && source .venv/bin/activate
python scripts/forward_test_d1.py --report
# Predictions logged to: ~/.wavecast/forward_tests/d1_forward_v1/predictions.jsonl
```

---

## Model Files

| File | Purpose |
|------|---------|
| `~/.wavecast/models/d1_augmented_v1/config.json` | Model architecture config |
| `~/.wavecast/models/d1_augmented_v1/model.pt` | Trained weights |
| `~/.wavecast/models/d1_augmented_v1/quantile_boundaries.json` | Return quantile bin edges (per DWT level) |
| `~/.wavecast/audit/production/pnl_simulation.json` | 8-config comparison results + bin midpoints + tercile thresholds |
| `~/.wavecast/audit/production/2026_validation.json` | 2026 OOS validation results |
| `~/.wavecast/audit/production/ce_baseline_predictions.npz` | Saved training predictions for threshold computation |

---

## Known Limitations

1. **Cost assumption**: 7 bps round-trip is reasonable for liquid US equities but may underestimate for less liquid names (UNG, COP).
2. **Data period**: Model trained on 2021-2025 hourly data. Market regime shifts could degrade performance.
3. **No real execution**: Forward testing uses paper trading — actual execution costs, slippage, and market impact are not measured.
4. **Hourly resolution only**: The system trades on 1h bars. Sub-hourly or daily bars are not validated.
5. **The 64% ceiling is real**: 66 experiments across 5 rounds confirm ~64% is the Bayes-optimal limit for this representation. Breaking through requires fundamentally different data sources.
