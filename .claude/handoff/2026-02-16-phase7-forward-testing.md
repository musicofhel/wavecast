# Phase 7: Forward Testing Framework — Full Run Complete

**Date**: 2026-02-16
**Tests**: 424 total, 0 failures
**Ruff**: 0 errors on new code

## Commits
1. `1b595ab` — Phase 7 framework: forward testing module, CLI, 33 tests
2. `01c6779` — CLAUDE.md updated with Phase 7 docs
3. (pending) — Training script, model artifacts, buffer fixes, first live run

## What Was Built

Forward testing framework for paper trading — runs the full WaveCast signal pipeline on live/recent data, logs predictions, then validates against actual outcomes on the next run.

### New Module: `src/wavecast/forward/` (7 files, ~600 LOC)

| File | Purpose |
|------|---------|
| `types.py` | `ForwardPrediction` + `ForwardTestSummary` dataclasses with JSON serialization |
| `config.py` | `ForwardTestConfig` (Pydantic BaseSettings) — tickers, intervals, horizons, paths |
| `tracker.py` | `ForwardTestTracker` — JSONL-based prediction logging, resolution against actual prices, rolling metrics |
| `runner.py` | `ForwardTestRunner` — orchestrates fetch → resolve → DWT/SAX/tokenize → predict → log cycle |
| `report.py` | Text + JSON report generation (per-ticker, per-interval breakdowns) |
| `__init__.py` | Module exports |

### Training Script: `scripts/train_forward_model.py`

Trains WaveletGPT with Phase 3 optimal config on all 20 DEFAULT_UNIVERSE tickers (2021-2024 train, 2025 validation). Saves model + vocab to `~/.wavecast/models/forward_ready/`.

**Training results**: 62.4% token accuracy on held-out 2025 data (consistent with Phase 3 research). 43s on RTX 2060 SUPER.

### Model Artifacts: `~/.wavecast/models/forward_ready/`
- `model.pt` (632K) — WaveletGPT state dict
- `config.json` — model architecture config (vocab=102, ctx=16, embed=64, layers=3)
- `vocabulary.json` — SAXVocabulary (102 tokens incl. PAD+UNK)

### First Live Forward Test

```
wavecast forward run \
  --model-path ~/.wavecast/models/forward_ready \
  --vocab-path ~/.wavecast/models/forward_ready/vocabulary.json \
  --tickers AAPL,MSFT,GOOGL,SPY --interval 1h --test-name paper_v1
```

**First cycle predictions** (Feb 16):
- AAPL: DOWN (-1), 94.7% confidence
- MSFT: DOWN (-1), 93.6% confidence
- GOOGL: FLAT (0), 58.7% confidence
- SPY: DOWN (-1), 96.0% confidence

**Second cycle** resolved all 4 predictions (target timestamps had passed). Actual returns were 0 because targets fell on weekend/off-market hours — expected for a Sunday run. In production runs during trading hours, resolution will be meaningful.

### Bug Fixes During Live Run

1. **`fetch_latest_bars` buffer too small for intraday**: 1.5x multiplier gave ~12.5 days for 200 hourly bars, but stocks only trade ~7h/day × 5d/week. Fixed: 5x buffer for intraday intervals (1000 hours ≈ 42 calendar days → ~200+ hourly bars).

2. **DWT level 5 minimum data length**: db4 wavelet at level 5 requires ≥224 data points. 200 lookback bars was below minimum. Fixed: default lookback_bars increased from 200 to 300.

## CLI Commands

```bash
# Train model
python scripts/train_forward_model.py

# Run one cycle
wavecast forward run --model-path ~/.wavecast/models/forward_ready \
  --vocab-path ~/.wavecast/models/forward_ready/vocabulary.json \
  --tickers AAPL,MSFT,GOOGL,SPY --interval 1h --test-name paper_v1

# Check status
wavecast forward status --test-name paper_v1
wavecast forward report --test-name paper_v1 --format json
wavecast forward list
```

## Architecture

```
ForwardTestRunner.run_once()
  ├─ _load_model()     → WaveletGPT.load() (cached)
  ├─ _load_vocab()     → SAXVocabulary.load() (cached)
  └─ for ticker × interval:
       ├─ fetch_latest_bars(ticker, interval, n_bars=300)
       ├─ tracker.resolve_pending(ticker, interval, prices)
       ├─ _build_pipeline_context(prices)  → DWT → SAX → tokenize → X array
       ├─ model.predict_proba(X[-1:])      → softmax over vocabulary
       ├─ SignalGenerator.generate()        → direction, confidence
       └─ tracker.log_prediction()          → JSONL append
```

## Key Design Decisions

1. **Reuses exact pipeline from ExperimentRunner**: DWT → SAX → extract_words → encode with pre-loaded vocabulary → build_sequence_dataset → column_stack with level+asset_class IDs. No drift from training pipeline.

2. **No vocabulary rebuilding**: Forward test uses the vocabulary saved during training. New unseen words get UNK tokens, matching the train/test split protocol.

3. **JSONL storage**: Simple, appendable, human-readable. Full rewrite on resolution is fine — forward tests accumulate hundreds of predictions, not millions.

4. **5x buffer for intraday fetch**: Stocks trade ~7h/day, 5d/week, so hourly bars need much more calendar time than 1.5x buffer assumed.

5. **300 lookback bars default**: DWT level 5 with db4 needs ≥224 data points. 300 gives comfortable margin.

## Next Steps (Phase 8)

1. **Cron scheduling**: Set up cron/systemd to run `wavecast forward run` hourly during market hours
2. **Multi-timeframe**: Hourly predictions informing 5-minute context windows
3. **Incremental training**: Fine-tune on accumulated forward test data
4. **Dashboard**: CLI or web dashboard for live prediction monitoring
