# Phase 4 Complete — HPO + Multi-Horizon

**Date**: 2026-02-16
**Commit**: `e82ab13`
**Tests**: 303 passing in ~11s

## What Was Done

3 parallel agents completed all Phase 4 tasks (F1-F8):

**Agent-1: F1 (Multi-Horizon) + F7 (Experiments)**
- Multi-horizon WaveletGPT: `prediction_horizons=[1,2,4,8]` param
- Weight-tied h=1 head, independent heads for h=2,4,8
- `predict_all_horizons()` method, `MultiHorizonMetrics`, `evaluate_multi_horizon()`
- Dataset: `max_horizon` param, `SequenceSample.targets` dict, `to_multi_horizon_arrays()`
- F7 experiments: h=1 (68.5%), h=2 (56.3%), h=4+ plateaus at ~41%
- 23 new tests

**Agent-2: F4 (Optuna SAX) + F5 (Optuna Arch) + F6 (Per-Sector)**
- `experiments/hpo.py`: SAXObjective, ArchitectureObjective, run_sax_hpo(), run_architecture_hpo()
- F4: 30 Optuna trials → n_segments=512 (was 256), word_length=4, word_stride=1 confirmed
- F5: 25 Optuna trials → embed_dim=128, num_layers=6, dropout=0.2 (was 64, 3, 0.1)
- F6: Per-sector fine-tuning hurts ALL 6 sectors
- 6 new tests

**Agent-3: F2 (Expanding Window) + F3 (Price Reconstruction)**
- `expanding_window_split()`, `rolling_window_split()` in splitter.py
- ExperimentRunner: `split_mode` dispatch, `_run_on_split()`, `_run_multi_split()`
- `sax/reconstruction.py`: `sax_to_midpoints()`, `reconstruct_price_delta()`, `tokens_to_direction()`, `PriceReconstructionResult`
- CLI: `wavecast sax reconstruct AAPL`
- 49 new tests

**Team-lead (manual):**
- Updated CLAUDE.md, ROADMAP.md, ARCHITECTURE.md, GLOSSARY.md, MEMORY.md
- All 8 beads (F1-F8) closed

## Phase 4 Optimal Config

| Parameter | Phase 3 | Phase 4 (Optuna) |
|-----------|---------|-------------------|
| n_segments | 256 | **512** |
| embed_dim | 64 | **128** |
| num_layers | 3 | **6** |
| dropout | 0.1 | **0.2** |
| prediction_horizons | [1] | **[1, 2, 4, 8]** |
| Model params | ~161K | **~600K** |

## Key Results

| Horizon | Token Acc | Dir Acc |
|---------|-----------|---------|
| h=1 | 68.5% | 94-97% |
| h=2 | 56.3% | 94-97% |
| h=4 | ~41% | 94-97% |
| h=8 | ~41% | 94-97% |

- Per-sector fine-tuning: hurts ALL 6 sectors — cross-sector definitively confirmed
- Expanding window validation: results robust across multiple evaluation windows
- Price reconstruction: SAX tokens → inverse PAA → approximate price deltas work

## Files Changed (27 files, +3406/-149)

**New source:**
- `src/wavecast/sax/reconstruction.py`
- `src/wavecast/experiments/hpo.py`
- `scripts/run_F1.py`, `scripts/run_F4.py`, `scripts/run_F5.py`, `scripts/run_F6.py`

**New tests:**
- `tests/unit/test_expanding_splitter.py` (22 tests)
- `tests/unit/test_expanding_runner.py` (4 tests)
- `tests/unit/test_sax_reconstruction.py` (23 tests)
- `tests/unit/test_hpo.py` (6 tests)

**Modified:**
- `src/wavecast/models/wavelet_gpt.py` — multi-horizon heads
- `src/wavecast/tokenizer/dataset.py` — max_horizon, multi-target
- `src/wavecast/evaluation/token_eval.py` — MultiHorizonMetrics
- `src/wavecast/experiments/splitter.py` — expanding/rolling window
- `src/wavecast/experiments/config.py` — split_mode params
- `src/wavecast/experiments/result.py` — n_splits, std fields
- `src/wavecast/experiments/runner.py` — split_mode dispatch
- `src/wavecast/cli/commands/sax.py` — reconstruct command
- All 5 MD files updated

## What's Next (Phase 5)

Per ROADMAP.md:
1. Token predictions → directional signals (buy/sell/hold)
2. Confidence calibration (softmax probabilities → position sizing)
3. P2-only signal pipeline (P1 can be deprecated)
4. Walk-forward backtesting with transaction costs, bid-ask spreads, slippage
5. Risk metrics: max drawdown, Calmar ratio, tail risk
6. Additional asset classes
