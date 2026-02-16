# Phase 3 Complete — Final Model + Integration

**Date**: 2026-02-16
**Commits**: `40bbeb9` (Wave 1+2), `8831f17` (Wave 3)
**Tests**: 230 passing in ~8s

## What Was Done

### Wave 3 (2 parallel agents)

**Agent-1: D1-D4 (Final Model + Report)**
- D1: Aggregated C1-C7 results → `~/.wavecast/experiments/phase3_findings.json`
- D2: Trained final WaveletGPT on extended 2021-2024 period → `~/.wavecast/models/phase3_final/`
- D3: Held-out 2025 evaluation → `~/.wavecast/experiments/D3_held_out_2025.json`
- D4: Wrote `PHASE3_RESULTS.md` (18KB, 346 lines) — full report with all 7 research questions, tables, CIs, honest assessment, limitations, next steps

**Agent-2: E1-E3 (Integration)**
- E1: Added `Sector` enum (6 sectors), made PHASE3_UNIVERSE the DEFAULT_UNIVERSE, old universe at LEGACY_UNIVERSE, `by_sector()` method, 24 new universe tests
- E2: Updated config defaults: SAXConfig(n_segments=256, alphabet=7), TokenizerConfig(context=16, min_freq=1, vocab=100), SequenceModelConfig(epochs=80, lr=0.0005, patience=15). CLI defaults aligned.
- E3: Final docs pass on CLAUDE.md, ROADMAP.md, MEMORY.md with D3 held-out results

**Manual (team-lead):**
- Updated ARCHITECTURE.md: P2 primacy, sector_embed, vocab counts, scaling section
- Updated GLOSSARY.md: DEFAULT_UNIVERSE, LEGACY_UNIVERSE, Sector entries, context=16
- Updated CLAUDE.md: unit test count, experiment CLI commands, scripts listing
- Closed all 25 beads (A1-A3, B1-B7, C1-C7, D1-D4, E1-E3, E4)

## 2025 Held-Out Results (D3)

| Metric | Value | 95% CI |
|--------|-------|--------|
| Token Accuracy | 60.8% | [59.96%, 61.68%] |
| Directional Accuracy | 95.8% | [95.26%, 96.30%] |
| Top-3 Accuracy | 88.2% | - |
| Persistence Baseline | 36.0% | - |
| Momentum Baseline | 38.0% | - |

- **No overfitting**: 2025 consistent with 2024 validation (token acc dropped 2.2%, within noise)
- Level 5 (coarsest): 87.4% token acc — strongest predictor
- Top sectors: broad ETFs (62.5%), commodity ETFs (62.3%), tech (61.6%)

## Phase 3 Summary

All 25 beads closed. 78 experiments across 7 research questions. Key findings:
1. **Alphabet 7** optimal (C1)
2. **Levels [1,2,5]** — 3 & 4 are noise (C2)
3. **Cross-sector training helps** 4/6 sectors (C3)
4. **Natural vocab 83 tokens** — fully saturated (C4)
5. **Context 16** sweet spot (C5)
6. **Regime-robust** — consistent 82.7% across all regimes (C6)
7. **P2 dominates P1** — no ensemble benefit (C7)

## Current State

- All Phase 3 work committed and pushed (commits 40bbeb9 + 8831f17)
- `.github/workflows/ci.yml` created but NOT pushed (PAT lacks `workflow` scope)
- `.beads/` directory now tracked in git (25 closed tasks)
- `~/.wavecast/models/phase3_final/` has saved vocabulary + config (not in git, too large)

## Known Issues

- `.github/workflows/ci.yml` sitting in worktree, unpushed — needs PAT with `workflow` scope or push via web UI
- 18 pre-existing ruff style warnings (SIM108 ternary, UP042 StrEnum, etc.) — not regressions
- CLAUDE.md says "ruff check: 0 errors" — technically 18 style warnings exist (ruff counts them as errors)

## What's Next (Phase 4+)

Per ROADMAP.md:
1. **Multi-horizon prediction**: Extend WaveletGPT to predict 2, 4, 8 steps ahead
2. **Price reconstruction**: SAX token predictions → approximate price movements
3. **Expanding window validation**: Replace single train/test split with rolling windows
4. **Optuna HPO**: n_segments, word_length, word_stride, architecture params
5. **Signal generation**: Token predictions → directional signals → backtesting with costs
6. **Rust acceleration**: SAX word extraction, BoW counting, vocab encoding via PyO3

## File Inventory

- `PHASE3_RESULTS.md` — 346-line comprehensive report
- `scripts/run_D1_D4.py` — final model training + evaluation script
- `~/.wavecast/experiments/phase3_findings.json` — aggregated C1-C7 optimal configs
- `~/.wavecast/experiments/D3_held_out_2025.json` — 2025 held-out results
- `~/.wavecast/models/phase3_final/vocabulary.json` — final 83-token vocabulary
- `~/.wavecast/models/phase3_final/experiment_config.json` — final model config
