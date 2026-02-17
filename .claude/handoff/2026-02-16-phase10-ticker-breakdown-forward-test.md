# Phase 10: Per-Ticker Breakdown + Forward Testing — Session Handoff

**Date**: 2026-02-16
**Status**: Step 1 complete, Step 2 complete (first forward cycle run, awaiting resolution)

## What Was Done

### Step 1: Per-Ticker/Sector/Level Breakdown

Created `scripts/d1_ticker_breakdown.py` — trains D1 (quick mode), evaluates per-group with 500 random trials per ticker and inverse-signal check for HARMFUL tickers.

**Key finding: Signal is broad-based, not concentrated.**

All 20 tickers PASS (>55% econ dir accuracy + Sharpe+costs >0.5):

```
Top 5:  GLD (65.5%), QQQ (63.3%), SPY (62.6%), USO (61.6%), UNG (61.6%)
Bot 5:  COP (58.8%), GS (58.7%), UNH (58.5%), JPM (58.4%), AAPL (59.1%)
```

Per-sector (all PASS):
```
broad_etf:     63.0%  S+Cost +1.79
commodity_etf: 61.9%  S+Cost +3.29  (strongest Sharpe)
healthcare:    59.9%  S+Cost +1.87
tech:          59.9%  S+Cost +2.13
energy:        59.4%  S+Cost +1.56
finance:       58.8%  S+Cost +1.21
```

**Per-level (critical finding):**
```
Level 1 (2-bar):  64.3%  S+Cost +4.77  PASS  ← carries nearly all signal
Level 2 (4-bar):  50.6%  S+Cost -1.27  NOISE
Level 5 (32-bar): 51.9%  S+Cost +0.32  NOISE
```

Level 1 is 65% of test samples and has 64% directional accuracy. Levels 2+5 dilute the aggregate from 64% down to 60%.

Results: `~/.wavecast/audit/d1_ticker_breakdown.json`

### Step 2: Forward Testing Infrastructure

#### New Files
- `scripts/train_d1_model.py` — trains production D1 on 2021-2025 data (quick architecture: 64d, 3 layers, 20 epochs)
- `scripts/forward_test_d1.py` — convenience runner for forward test cycles (`--report` flag for per-ticker output)

#### Modified Files
- `src/wavecast/forward/runner.py`:
  - `_build_pipeline_context()` now auto-detects `model.input_mode`
  - `"continuous"` → `_build_d1_pipeline_context()` (DWT → delta coefficients → aux features → sliding windows)
  - `"tokenized"` → `_build_sax_pipeline_context()` (original SAX pipeline, unchanged)
  - `_build_aux_windows()` static method for auxiliary feature windowing
  - Vocab loading skipped for continuous models
- `src/wavecast/forward/config.py`: added `retrain_after_days: int | None = None` placeholder

#### Production Model
- Path: `~/.wavecast/models/d1_augmented_v1/`
- Artifacts: `model.pt`, `config.json`, `quantile_boundaries.json`, `training_config.json`, `training_metrics.json`
- Trained on 282,861 windows (20 tickers × 3 levels × 2021-2025)
- Loss: 1.4909, training time: 645s

#### First Forward Test Cycle
- 20 predictions logged to `~/.wavecast/forward_tests/d1_forward_v1/predictions.jsonl`
- Mix of dir=-1 (bearish) and dir=0 (flat) predictions — no resolutions yet (need next market close)
- Run again with: `python scripts/forward_test_d1.py --report`

### PRD Updates (from user feedback)
Applied 5 corrections to the PRD at `.claude/handoff/2026-02-16-phase10-ticker-breakdown-forward-testing.md`:
1. Quick-mode architecture for production model (not full-scale) — marginally better + less overfitting
2. 500 random trials per ticker (not 100) — smaller per-ticker N needs more trials
3. HARMFUL tickers: check inverse signal before excluding (anti-predictive = tradeable with flipped signal)
4. Boundary path must come from same model directory (boundaries shift with retraining data)
5. `retrain_after_days` placeholder for model staleness policy

## Test Status
- **476 tests passing**, 0 failures, 0 new lint errors
- No new test files this session (forward runner changes are backward-compatible, tested by existing `test_forward_runner.py`)

## Next Steps

1. **Accumulate forward predictions**: Run `python scripts/forward_test_d1.py --report` daily/hourly. Each run resolves pending predictions and logs new ones. Need ~100 resolved per ticker for reliable evaluation.
2. **Level-1-only variant**: The breakdown showed levels 2+5 are pure noise. Training D1 on level 1 only would eliminate 35% noise samples and likely improve aggregate accuracy from 60% toward 64%. Quick experiment (~15 min).
3. **Evaluate forward results**: After sufficient resolution, `wavecast forward report --test-name d1_forward_v1` shows directional accuracy on true out-of-sample 2026 data.
4. **Decision gate**: >55% on 2026 data = real edge. 50-55% = marginal. <50% = overfit to 2025 test set.
