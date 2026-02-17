# Phase 11: 2026 OOS Validation + Level-1-Only Model + Feature Tests

**Date**: 2026-02-16
**Status**: COMPLETED

## Summary

Validated the frozen D1 production model on truly out-of-sample 2026 data (Jan 1 - Feb 14). Trained and compared a level-1-only variant. Built a reusable feature test harness and tested three feature sets (bar structure, range dynamics, wavelet jump filters). All three failed — the D1 representation (coefficient deltas + 4 aux features) is the signal ceiling.

## Results

### 2026 Out-of-Sample Validation (`scripts/validate_d1_2026.py`)

Frozen model from `~/.wavecast/models/d1_augmented_v1/` tested on unseen 2026 data:

| Metric | 2025 Reference | 2026 OOS | Delta |
|--------|---------------|----------|-------|
| Econ Dir Accuracy | 60.0% | 60.0% | 0.0pp |
| Sharpe (w/costs) | +1.86 | +2.182 | +0.32 |
| vs Random | 100% | 100% | - |

**Per-level (2026)**:
- Level 1: **63.8%** econ dir, +4.751 Sharpe
- Level 2: 51.4% (noise)
- Level 5: too few samples (76 total, 0 valid)

**Top tickers (2026)**: SPY 67.7%, JNJ 65.1%, GLD 64.4%, UNG 62.5%, GS 62.5%

**VERDICT: PASS (STRONG)** — zero degradation from 2025 to 2026

Results: `~/.wavecast/audit/d1_2026_validation.json`

### Level-1-Only Model (`scripts/train_d1_level1.py`, `scripts/validate_d1_level1_2026.py`)

Hypothesis: Since levels 2+5 are noise, training exclusively on level 1 should help.

| Metric | All-Levels (all test) | All-Levels (L1 filtered) | L1-Only Model |
|--------|----------------------|-------------------------|---------------|
| Econ Dir | 60.0% | 63.8% | 63.2% |
| Sharpe | +2.182 | +4.751 | +4.612 |
| n_valid | 6,367 | 4,357 | 4,357 |

**VERDICT: SAME** (-0.7pp vs all-levels filtered to L1). The all-levels model's level embedding already handles separation. The L1 model avoids noise-level predictions but doesn't improve L1 accuracy.

Model saved: `~/.wavecast/models/d1_level1_v1/`
Results: `~/.wavecast/audit/d1_level1_2026_validation.json`

### Feature Test Harness (`scripts/feature_tests/harness.py`)

Reusable A/B framework:
- Loads OHLCV from cache, splits train (<=2024-06-30) / test (>=2025-01-01)
- Trains baseline (4 aux features) vs challenger (4 + N new features) across 3 seeds
- Pass criteria: improve at least one metric without degrading any
- Thresholds: >+1pp econ dir, >+0.2 Sharpe, >+2pp transition (improve); <-1pp, <-0.3, <-1pp (degrade)

Feature function contract: `(ohlcv_df, detail_coeffs, approx_coeffs, level) -> NDArray(n_deltas, n_features)`

### Feature Test 1: Bar Structure (`scripts/feature_tests/test_bar_structure.py`)

3 features from OHLC: body_ratio, upper_wick, lower_wick

| Metric | Baseline | Challenger | Delta | Status |
|--------|----------|------------|-------|--------|
| Econ Dir | 63.9% | 64.2% | +0.3pp | Below threshold |
| Sharpe | +4.673 | +4.558 | -0.115 | No improvement |
| Transition | 55.2% | 51.1% | -4.1pp | **DEGRADED** |

**VERDICT: FAIL** — transition accuracy degraded

### Feature Test 2: Range Dynamics (`scripts/feature_tests/test_range_dynamics.py`)

2 features: range_expansion (short/long range ratio), effort_result (close move / bar range)

| Metric | Baseline | Challenger | Delta | Status |
|--------|----------|------------|-------|--------|
| Econ Dir | 63.9% | 62.2% | -1.7pp | **DEGRADED** |
| Sharpe | +4.673 | +3.785 | -0.888 | **DEGRADED** |
| Transition | 55.2% | 49.8% | -5.3pp | **DEGRADED** |

**VERDICT: FAIL** — all three metrics degraded

### Feature Test 3: Jump Filters (`scripts/feature_tests/test_jump_filters.py`)

2 features from Aubrun et al. 2024 (arXiv:2404.16467): causal ψMR (mean-reversion strength) and ψTR (trend consistency). 8-bar cosine-envelope kernels applied to log returns, backward-looking only (no future leakage).

| Metric | Baseline | Challenger | Delta | Status |
|--------|----------|------------|-------|--------|
| Econ Dir | 63.9% | 64.2% | +0.3pp | Below threshold |
| Sharpe | +4.673 | +4.660 | -0.013 | No improvement |
| Transition | 55.2% | 50.5% | -4.7pp | **DEGRADED** |

All 3 seeds had better econ dir (+0.2 to +0.4pp) but below the +1pp threshold. Seed 1 Sharpe (+0.248) was noise — seeds 2-3 pulled the mean back to flat. Transition degradation matches the pattern from tests 1 and 2.

**VERDICT: FAIL** — transition accuracy degraded

## Key Conclusions

1. **D1 representation is robust on OOS data** — zero degradation from 2025 to 2026
2. **Level 1 dominates** — 63.8% vs 51.4% for level 2. Level-1-only model adds no improvement.
3. **3/3 feature tests FAIL on transition accuracy** — bar structure (-4.1pp), range dynamics (-5.3pp), jump filters (-4.7pp). Adding features from the same price series consistently degrades the model's ability to predict reversals. The D1 representation (coefficient deltas + 4 aux) is the signal ceiling.
4. **Feature test harness is reusable** — can test any new feature set with `run_feature_test(name, fn, n_features)`
5. **60% econ dir / +2.18 Sharpe after costs is the system's edge.** Accept it and shift to operational work.

## New Files Created

| File | LOC | Purpose |
|------|-----|---------|
| `scripts/validate_d1_2026.py` | ~290 | 2026 OOS validation of frozen D1 model |
| `scripts/train_d1_level1.py` | ~160 | Train level-1-only D1 model |
| `scripts/validate_d1_level1_2026.py` | ~220 | Validate L1 model on 2026, compare with all-levels |
| `scripts/feature_tests/__init__.py` | 1 | Package init |
| `scripts/feature_tests/harness.py` | ~310 | Reusable feature test framework |
| `scripts/feature_tests/test_bar_structure.py` | ~68 | Bar structure feature test |
| `scripts/feature_tests/test_range_dynamics.py` | ~73 | Range dynamics feature test |
| `scripts/feature_tests/test_jump_filters.py` | ~105 | Wavelet jump filter feature test |

## What's Next

The D1 model is validated, stable, and at its signal ceiling. Remaining directions:
- **Operational**: Position sizing, risk management, scheduled execution, monitoring
- **Confidence calibration**: Meta-model that identifies which predictions to trust (doesn't require new features)
- **Cross-asset context**: Co-jump contagion from Aubrun et al. — architecture change, not feature addition
- **Forward testing**: Let `d1_forward_v1` accumulate predictions and resolve against actuals
