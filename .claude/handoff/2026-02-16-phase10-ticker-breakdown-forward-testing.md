# Phase 10: Per-Ticker Breakdown + Forward Testing

**Date**: 2026-02-16
**Status**: PRD — not started
**Prerequisite**: Phase 9b complete (D1: Delta+Aux validated at 60% econ dir accuracy)

---

## Motivation

D1 achieved 60% economic directional accuracy and Sharpe+costs of +1.86 on aggregated 2025 test data across all 20 tickers. But aggregated metrics can hide concentration risk: if 4 tickers carry the Sharpe and 16 are noise, forward testing all 20 wastes compute and dilutes the signal.

The aggregated results JSON (`representation_comparison.json`) stores only top-level metrics — no per-ticker breakdown exists. We need to slice the evaluation before investing in forward testing infrastructure changes.

---

## Step 1: Per-Ticker/Sector Breakdown (~150 LOC new script)

### What

A script that re-runs the D1 pipeline (quick mode is sufficient — signal is in the representation, not model capacity) but tracks per-ticker and per-level predictions separately, then slices the evaluation.

### Script: `scripts/d1_ticker_breakdown.py`

**Approach**: Reuse `representation_comparison.py`'s D1 pipeline but intercept predictions at the ticker level. The key change: instead of building one giant X array across all tickers, tag each sample with its source ticker so we can evaluate per-group.

**Data flow**:
1. Load + split prices (same `load_and_split_prices()`)
2. Build D1 continuous dataset (same pipeline: DWT → detail_delta_coeffs → aux features → build_continuous_dataset)
3. Train one D1 model on all tickers combined (same as before — cross-ticker training)
4. Predict on test set
5. **New**: group predictions by ticker, by sector, by level, and evaluate each group independently

**Metadata to track per test sample**:
- `ticker: str` — source ticker
- `level: int` — DWT level (1, 2, or 5)
- `sector: str` — from universe (tech, finance, energy, healthcare, broad_etf, commodity_etf)

**Per-group metrics** (same battery as `evaluate_candidate()`):
- Economic directional accuracy (filtered |return| > 0.1%)
- Sharpe (raw + with 7bps costs)
- Sharpe percentile vs random (500 trials per ticker — higher than aggregate's 100 because per-ticker N is smaller and variance wider)
- Transition accuracy
- Sample count — **flag any result with N_test < 1,000 as UNRELIABLE** regardless of metrics (small N inflates percentile scores)
- Quantile accuracy

**Output tables**:

```
PER-TICKER BREAKDOWN (D1: Delta+Aux)
Ticker   Sector        N_test  Econ Dir  S+Cost  vs Rnd  Trans
AAPL     tech           3200    62.1%   +2.31   100.0%  47.2%
MSFT     tech           3150    58.4%   +1.45    98.0%  43.1%
...
UNG      commodity_etf  2900    64.3%   +3.12   100.0%  51.2%

PER-SECTOR BREAKDOWN
Sector          N_test  Econ Dir  S+Cost  vs Rnd  Trans
tech            15800    59.2%   +1.88   100.0%  44.1%
finance          9400    57.1%   +1.22    96.0%  42.3%
energy           9500    61.8%   +2.45   100.0%  48.7%
healthcare       9300    55.2%   +0.89    88.0%  40.1%
broad_etf        6200    58.9%   +1.76    99.0%  45.0%
commodity_etf   11600    63.1%   +2.91   100.0%  50.2%

PER-LEVEL BREAKDOWN
Level  N_test  Econ Dir  S+Cost  vs Rnd  Trans
1      28000    57.3%   +1.12    95.0%  42.1%
2      18000    60.2%   +2.05   100.0%  46.3%
5      12000    63.8%   +2.89   100.0%  49.1%
```

(Numbers above are illustrative — real results TBD.)

**Pass/Fail per ticker**:
- PASS: Econ Dir > 55% AND Sharpe+costs > 0.5 AND N_test > 500
- MARGINAL: Econ Dir 52-55%
- NOISE: Econ Dir 48-52%
- HARMFUL: Econ Dir < 48% — **check inverse signal before excluding**. If flipping predictions gives >55% accuracy, that's a systematic model bias on that ticker, not noise. Anti-predictive tickers can be traded with inverted signals.

**Output files**:
- `~/.wavecast/audit/d1_ticker_breakdown.json` — full per-ticker/sector/level metrics
- Console table (same format as representation_comparison.py)

### Implementation Notes

The continuous dataset's `ContinuousWindow` dataclass already has `ticker` and `level` fields — we just need to carry them through to evaluation. The existing `build_continuous_dataset()` returns a `ContinuousDataset` with `.windows` list, each window having `.ticker` and `.level`.

```python
# Pseudocode for per-ticker eval
ticker_groups: dict[str, list[int]] = {}  # ticker -> list of sample indices
for i, w in enumerate(test_ds.windows):
    ticker_groups.setdefault(w.ticker, []).append(i)

for ticker, indices in ticker_groups.items():
    idx = np.array(indices)
    evaluate_candidate(
        f"  {ticker}", pred_labels[idx], proba[idx],
        test_rets[idx], y_test[idx], test_valid[idx], ...
    )
```

### Decision Gate

After reviewing per-ticker results:
- **Proceed to forward testing** with tickers that PASS
- **Exclude** NOISE and HARMFUL tickers from forward test
- **Watch** MARGINAL tickers — include if sector shows aggregate edge
- If < 5 tickers pass, the signal may be too concentrated for a viable strategy (document this finding)
- If all sectors show edge, proceed with full universe

---

## Step 2: Forward Testing on D1 Pipeline (~200 LOC modified)

### What

Adapt the existing `ForwardTestRunner` to use the D1 representation (coefficient deltas + auxiliary features) instead of SAX tokens. Then run forward testing on 2026 data for the tickers identified in Step 1.

### Why This Is The Real Test

Everything on 2021-2024 train / 2025 test has the same limitation: D1 was *selected* because it performed well on 2025 data. Forward testing on 2026 data that played no role in selecting D1 is the only honest validation.

### Modified Files

#### 1. `scripts/train_d1_model.py` (NEW, ~120 LOC)

Train a production D1 model on all available data (2021-2025) for maximum signal. Save to `~/.wavecast/models/d1_augmented_v1/`.

```python
# Saves:
# - model checkpoint (WaveletGPT.save())
# - quantile_boundaries.json (per-level boundaries for return classification)
# - training_config.json (all hyperparams for reproducibility)
# - training_metrics.json (final loss, epoch, time)
```

**Training config**: Quick-mode architecture (embed=64, 3 layers, 20 epochs, dropout=0.1, lr=0.0005, patience=10). Phase 9 showed full-scale was marginally *worse* than quick (60.0% vs 60.5%, Sharpe 1.86 vs 2.08) — signal is in the representation, not model capacity. Using the smaller architecture avoids overfitting and gives faster retraining. Training on 2021-2025 with the bigger model simultaneously would introduce two changes (architecture + data), making any degradation unattributable.

**Training data**: All tickers, 2021-2025, all 3 detail levels. This is MORE data than Phase 9 used (which only had 2021-2024 for training). Using 2025 for training is fine here because forward testing will evaluate on 2026 only.

#### 2. `src/wavecast/forward/runner.py` (MODIFIED)

**Changes to `_build_pipeline_context()`**:

Current pipeline:
```
prices → DWT → detail_coeffs → SAX → extract_words → vocab.encode → build_sequence_dataset → X
```

D1 pipeline:
```
prices → DWT → detail_delta_coeffs → build_continuous_windows → compute_aux_features → X
```

Add a `_build_d1_pipeline_context()` method that:
1. Decomposes prices via DWT (same)
2. Computes `np.diff(detail_at_level(lvl))` for each level (NEW)
3. Computes auxiliary features via `compute_detail_auxiliary_features()` (NEW)
4. Builds sliding windows (using `build_continuous_dataset` or inline) (NEW)
5. Packs X array: `[ctx, aux_flat, level, asset_class]` (NEW)

**Auto-detection**: `_build_pipeline_context()` checks `model.input_mode`:
- `"tokenized"` → existing SAX pipeline (backward compatible)
- `"continuous"` → new D1 pipeline

This way a single `ForwardTestRunner` handles both old and new models.

#### 3. `src/wavecast/forward/config.py` (MODIFIED)

Add optional fields:
- `boundary_path: str = ""` — path to per-level quantile boundaries (needed for return interpretation). **CRITICAL**: Must point to the same model directory as `model_path` — boundaries shift when retraining on 2021-2025 vs 2021-2024. `_build_d1_pipeline_context()` must load boundaries from the model directory, not a hardcoded or stale path.
- `selected_tickers: list[str] | None = None` — if set, only forward test these tickers (from Step 1 results)
- `retrain_after_days: int | None = None` — placeholder for model staleness policy. D1 trained on 2021-2025 will drift as 2026 market structure evolves. Not solved now, but the config should have the hook.

#### 4. `scripts/forward_test_d1.py` (NEW, ~80 LOC)

Convenience script to run D1 forward test:
```bash
python scripts/forward_test_d1.py \
    --model-path ~/.wavecast/models/d1_augmented_v1 \
    --tickers AAPL,XOM,CVX,GLD,SLV,USO,UNG \
    --test-name d1_forward_v1
```

Fetches latest 2026 bars, runs single prediction cycle, logs to JSONL. Can be called repeatedly (cron or manual) to accumulate predictions over time.

### Forward Test Evaluation

After accumulating enough predictions (suggest minimum 100 per ticker):

```bash
wavecast forward report --test-name d1_forward_v1 --format text
```

**Key metric**: Economic directional accuracy on 2026 data.
- If > 55% on aggregate across selected tickers: **real edge confirmed**
- If 50-55%: edge may exist but is marginal, needs more data
- If < 50%: D1 was overfit to the 2025 test set, back to drawing board

---

## Implementation Order

1. **`scripts/d1_ticker_breakdown.py`** + run it (~30 min total: ~15 min code, ~15 min GPU)
2. **Review results** — decide which tickers to forward test
3. **`scripts/train_d1_model.py`** — train production model on 2021-2025 (~20 min GPU)
4. **`forward/runner.py`** modifications — D1 pipeline context builder (~30 min code)
5. **`forward/config.py`** — boundary_path + selected_tickers (~5 min)
6. **`scripts/forward_test_d1.py`** — convenience runner (~10 min)
7. **Run forward test** — first prediction cycle on latest 2026 data
8. **Tests**: ~10 new tests (d1 pipeline context builder, auto-detection, round-trip)

Total: ~2 hours including GPU time

---

## What We're NOT Doing (and why)

- **Family D multi-embedding (SAX + approx SAX + delta sign + Hurst + vol bucket)**: More research on the same 2025 test set that selected D1. Diminishing returns at best, p-hacking at worst.
- **Family E (LAMA leitmotifs)**: Same — backward optimization on already-seen data.
- **Transition accuracy improvement**: 44.9% vs 50% threshold. Important eventually, but forward testing tells us if the 60% directional accuracy is real first. If it's not real on 2026 data, improving transitions is moot.
- **Further architectural experiments**: The quick-mode vs full-scale comparison already showed signal is in the representation, not model capacity. More architecture tuning = more test-set overfitting risk.

---

## Files Summary

| File | Type | LOC | Purpose |
|------|------|-----|---------|
| `scripts/d1_ticker_breakdown.py` | NEW | ~150 | Per-ticker/sector/level D1 evaluation |
| `scripts/train_d1_model.py` | NEW | ~120 | Train production D1 model on 2021-2025 |
| `scripts/forward_test_d1.py` | NEW | ~80 | Convenience forward test runner |
| `src/wavecast/forward/runner.py` | MOD | ~80 new | D1 pipeline context + auto-detection |
| `src/wavecast/forward/config.py` | MOD | ~5 new | boundary_path, selected_tickers |
| Tests | NEW | ~60 | D1 forward pipeline tests |
| **Total** | | ~495 | |
