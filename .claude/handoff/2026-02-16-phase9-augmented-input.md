# Phase 9: Augmented Input Representation — Session Handoff

**Date**: 2026-02-16
**Status**: D1 (Delta+Aux) validated at full scale — 60.0% economic directional accuracy

## What Was Done

### Phase 9a: Representation Comparison (8 candidates)
- Implemented `input_mode="continuous"` in WaveletGPT — bypasses SAX, uses `nn.Linear(1, embed_dim)` for raw float input
- Created `src/wavecast/data/continuous_dataset.py` — sliding window builder for raw coefficients
- Created `scripts/representation_comparison.py` — comparison harness for all candidates
- Quick-mode results (embed=64, 3 layers, 20 epochs):
  - **C3 (Raw Detail Deltas) PASSED** at 58.3% econ dir accuracy
  - Key insight: `np.diff(coefficients)` encodes rate-of-change (direction), raw coefficients encode level (magnitude)

### Phase 9b: Augmented Input (Option D, narrata-inspired)
- Added `n_aux_features` parameter to WaveletGPT — per-timestep auxiliary features via `nn.Linear(n_aux, embed_dim)`
- Created `src/wavecast/data/auxiliary_features.py` — 4 auxiliary feature channels:
  - Feature 0: `coeff_sign` — sign of original coefficient at position i+1
  - Feature 1: `abs_coeff_zscore` — z-scored |coefficient| magnitude
  - Feature 2: `volatility_ratio` — |delta| / EMA(|delta|), capped at 5.0
  - Feature 3: `approx_direction` — sign of approximation coefficient delta at mapped position
- X array format: `[ctx_0..ctx_{L-1}, aux_00..aux_{L-1}_{F-1}, level, asset_class]`

### D1 Results (full scale: embed=128, 6 layers, 80 epochs)

```
Economic directional accuracy:  60.0%    PASS (>55%)
Quantile accuracy:              31.5%
Sharpe (raw):                   +4.439
Sharpe (with 7bps costs):      +1.855   PASS (>0.5)
vs Random percentile:           100.0%   PASS (>95th)
Transition accuracy:            44.9%    close (threshold 50%)
Training time:                  1072.6s (~18 min)
```

Quick vs full produced nearly identical results — signal is in the data representation, not model capacity.

## Files Created/Modified

### New Files
- `src/wavecast/data/auxiliary_features.py` — `compute_detail_auxiliary_features()`, `N_AUX_FEATURES=4`
- `src/wavecast/data/continuous_dataset.py` — `ContinuousWindow`, `ContinuousDataset`, `build_continuous_dataset()`
- `scripts/representation_comparison.py` — full comparison harness with `--quick`, `--only=NAME` flags
- `tests/unit/test_auxiliary_features.py` — 10 tests
- `tests/unit/test_wavelet_gpt_augmented.py` — 5 tests

### Modified Files
- `src/wavecast/models/wavelet_gpt.py`:
  - `WaveletGPTNet`: added `n_aux_features` param, `aux_proj` layer, `aux_features` in `forward()`
  - `WaveletGPT`: added `n_aux_features` param, `_parse_x()` returns 4-tuple (ctx, lvl, ac, aux)
  - All predict/fit methods updated to pass aux through
  - Save/load persists `n_aux_features`
- `src/wavecast/models/batch_inference.py` — updated `_predict_batch()` for 4-tuple `_parse_x()`

### Results
- `~/.wavecast/audit/representation_comparison.json` — full comparison results

## Test Status
- **476 tests passing** (18 original GPT + 37 Phase 8 + 15 new = 476 total)
- **0 lint errors** on modified files (98 pre-existing warnings in other files)

## Comparison Table (Quick Mode, All 9 Candidates)

```
Candidate        Econ Dir  Sharpe+Cost  vs Rnd  Verdict
Baseline           50.5%      -0.100    38.0%
A: Approx SAX      53.1%      +1.151    84.0%  MARG
B1: Det Delta      51.9%      +0.003    49.0%
B2: Apx Delta      50.9%      -0.889    16.0%
C1: Raw Det        51.1%      -1.322     0.0%
C2: Raw Apx        52.7%      +1.362    93.0%  MARG
C3: Raw Delta      58.3%      +1.517   100.0%  PASS
C4: Raw A+D        50.7%      -1.855     0.0%
D1: Delta+Aux      60.5%      +2.081   100.0%  PASS ★
```

## Next Steps (from user's Phase 9 PRD)

1. **Family D multi-feature tokens**: SAX tokens + approx SAX + delta sign + Hurst regime + vol bucket as parallel embeddings (narrata-inspired, ~200 LOC new)
2. **Family E (LAMA leitmotifs)**: Cross-scale motif discovery using `leitmotif` package (GPL-3.0, external call only)
3. **Transition accuracy improvement**: D1 is at 44.9% vs 50% threshold — direction-change prediction is the weakest link
4. **Train production D1 model**: `scripts/train_return_model.py` adapted for D1 pipeline, save to `~/.wavecast/models/d1_augmented_v1/`
5. **Forward testing**: Integrate D1 into `ForwardTestRunner` (detect model's `input_mode` + `n_aux_features`)

## Key Architecture Decisions

- **Auxiliary features are ADDITIVE to embeddings**: `x = input_proj(values) + pos_embed + aux_proj(aux) + level_embed + asset_class_embed`. This preserves the existing embedding addition pattern.
- **Aux features computed from raw coefficients BEFORE normalization**: Sign, magnitude, and volatility ratio are properties of the original signal, not the z-normalized version.
- **Approx direction mapping uses linear ratio**: When approx and detail have different lengths, map by `index * (len_approx / len_detail)`. Works because both are derived from the same DWT decomposition.
- **Quick mode is sufficient for screening**: Full-scale model produced nearly identical results to quick mode, suggesting the edge is in the representation, not model capacity.
