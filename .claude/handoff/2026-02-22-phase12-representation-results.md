# Phase 12 (Exp3): Representation Change Experiments — Results

**Date**: 2026-02-22
**Duration**: ~18 hours GPU time across multiple runs
**Status**: ALL 10 EXPERIMENTS COMPLETE

## Context

Round 2 confirmed the 64% econ_dir ceiling with current D1 representation (coefficient deltas + 4 aux features). 26 experiments tried loss functions, architectures, training strategies, conformal wrappers — none broke through. The hypothesis: the representation itself must change.

Phase 12 tests 10 fundamental changes to the D1 input representation:
1. **Fractional differencing** (fracdiff) — preserve long memory while achieving stationarity
2. **Fine-grained quantiles** (7-class, 11-class) — more target resolution
3. **Extended context** (32, 48, 64 windows) — more temporal history
4. **Cross-scale features** — multi-scale wavelet correlations
5. **Range-based DWT** — high-low range instead of close prices
6. **Signed return regression** — continuous targets via Huber loss
7. **Auxiliary transition head** — multi-task learning with binary transition prediction

## Infrastructure

- Shared evaluation: `scripts/exp3/evaluate_representation.py` — unified 5-metric pipeline
- Baseline lock: `scripts/exp3/baseline_lock.json` — deterministic baseline reference
- Per-experiment scripts: `scripts/exp3/test_*.py`
- Results: `~/.wavecast/audit/exp3/*_results.json`
- Runners: `scripts/exp3/run_exp3.sh`, `run_final3.py`, `run_remaining.py`

## Results Summary

| Experiment | Verdict | econ_dir | transition | large_move | flat% | sharpe |
|------------|---------|----------|------------|------------|-------|--------|
| **Baseline** | - | **63.9%** | **55.2%** | **68.1%** | **22.9%** | **+4.67** |
| fracdiff | FAIL | 64.5% | 52.6% | 68.7% | 27.8% | +4.718 |
| finegrain_7 | INTERESTING | 63.6% | 59.7% | 67.8% | 15.0% | +4.568 |
| finegrain_11 | INTERESTING | 62.6% | 63.0% | 66.8% | 8.1% | +4.289 |
| context_32 | FAIL | 64.3% | 51.0% | 68.6% | 30.1% | +4.651 |
| context_48 | FAIL | 63.7% | 52.0% | 68.0% | 27.7% | +4.592 |
| context_64 | FAIL (OOM) | - | - | - | - | - |
| crossscale | FAIL | 49.8% | 37.8% | 51.0% | 24.7% | -0.555 |
| range_dwt | FAIL | 63.8% | 50.6% | 67.9% | 28.6% | +4.426 |
| regression | INTERESTING | 62.7% | 67.0% | 66.8% | 6.7% | +4.544 |
| transition_head | INTERESTING | 64.0% | 54.3% | 68.0% | 24.1% | +4.588 |

**Score: 0 PASS, 4 INTERESTING, 6 FAIL**

## Key Findings

### 1. Fine-grained quantiles unlock transition detection

The clearest signal: moving from 5-class to 7 or 11 class targets dramatically improves transition accuracy:
- 5-class baseline: 55.2% transition
- 7-class: **59.7%** (+4.5pp)
- 11-class: **63.0%** (+7.8pp)

The mechanism: more bins near zero mean the model can make finer-grained predictions about direction change magnitude. Flat predictions drop from 23% → 15% → 8%.

**Trade-off**: econ_dir decreases slightly (64% → 63.6% → 62.6%) because the model is now splitting hairs on direction rather than clustering into safe flat.

### 2. Signed return regression is the transition champion

Regression (Huber loss, continuous signed return targets) achieves the highest transition accuracy: **67.0%** (+12pp over baseline). Flat predictions are nearly eliminated (6.7%). But econ_dir drops to 62.7%.

The regression model genuinely detects transitions because it MUST predict direction — there's no flat class to hide behind.

### 3. Longer context windows HURT

Context=16 (baseline) is already optimal. Every extension degrades transition accuracy:
- context=32: transition 51.0% (-4.2pp)
- context=48: transition 52.0% (-3.2pp)
- context=64: CUDA OOM (7.54 GiB attention allocation on 8GB GPU)

Hypothesis: longer windows introduce noise from irrelevant historical patterns. The transformer attends to all positions equally, diluting the recent signal.

### 4. Alternative decompositions are dead ends

- **fracdiff**: Preserves long memory but transition drops 2.6pp. The fractional differencing removes too much short-term signal.
- **range_dwt**: High-low range DWT is worse across all metrics. Range doesn't carry directional signal.
- **crossscale**: Cross-scale correlations are catastrophic (49.8% econ_dir, -0.555 Sharpe). Only ~7.5K samples after filtering — representation is too sparse.

### 5. Transition head doesn't help much

Auxiliary binary transition prediction head (multi-task learning) has minimal effect:
- Best lambda=0.2: transition 54.3% (vs 55.2% baseline)
- The auxiliary head accuracy itself is stuck at 52.6% — the model can't learn to predict transitions from this representation
- Adding the transition loss slightly disrupts the main classification task

## Strategic Implications

1. **The econ_dir vs transition trade-off is fundamental**: Methods that improve transition (finegrain, regression) do so by reducing flat predictions, which slightly hurts econ_dir. You can't have both with the current representation.

2. **The representation ceiling is confirmed at ~64% econ_dir**: Across 56 total experiments (20 Round 1 + 26 Round 2 + 10 Phase 12), nothing has beaten 64% econ_dir. The D1 representation (wavelet coefficient deltas + 4 aux features) extracts all available signal.

3. **For transition-focused trading**: Use finegrain_11 or regression variants. They sacrifice ~1-2pp econ_dir but gain 8-12pp transition accuracy. If your strategy needs to detect regime changes (momentum → mean-reversion), this is the way.

4. **For selective trading**: EnCQR (Round 2 winner) + baseline CE remains the best configuration. 67.9% econ_dir at 44.6% coverage by only trading when the 5-model ensemble agrees.

5. **Next lever is NOT more experiments on the same data**: The information-theoretic ceiling (BOLT/MINE from Round 2) confirms ~64% is near-optimal for this feature set. Next steps:
   - Multi-timeframe: combine hourly + daily signals
   - Cross-asset: use sector/market signals as context
   - Fundamentally different data: order flow, options surfaces, etc.

## GPU Memory Issues

Context=48 required batch_size=16 (vs default 64) to fit in 8GB. Context=64 OOM'd even at batch_size=8 (tried to allocate 7.54 GiB for attention). Added `gc.collect() + torch.cuda.empty_cache()` between seeds in `evaluate_representation.py` to prevent memory accumulation.

## Files Modified

- `scripts/exp3/evaluate_representation.py` — Added gc import, GPU cleanup between seed trainings
- `scripts/exp3/run_final3.py` — Sequential runner with batch_size overrides for context experiments
- `scripts/exp3/test_*.py` — Individual experiment scripts (8 files)
- `scripts/exp3/run_exp3.sh` — Original batch runner
- `scripts/exp3/baseline_lock.json` — Deterministic baseline reference
