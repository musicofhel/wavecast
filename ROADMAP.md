# WaveCast Roadmap

## Completed

### Phase 1: Core Library (v0.1.0)
- [x] DWT decomposition (PyWavelets, db4, 5 levels)
- [x] Shapelet discovery (W-TSS: z-score → binary → contiguous → IG scoring)
- [x] DTW matching (dtaidistance, constrained, ShapeDTW)
- [x] Fractal analysis (Hurst exponent, MFDFA, regime detection)
- [x] Feature pipeline (~40-50 features: wavelet + shapelet + fractal + market)
- [x] Models: WaveletLSTM, XGBoost, weighted ensemble
- [x] Evaluation: RMSE, MAE, directional accuracy, Sharpe, walk-forward backtest
- [x] CLI: data, discover, match, analyze, forecast, library, backtest
- [x] Massive.com data source with Parquet caching
- [x] 88 tests passing

### Phase 2: SAX + Token Prediction (v0.2.0)
- [x] PAA compression + inverse
- [x] SAX transform with breakpoints and MINDIST distance
- [x] Bag-of-Words extraction + TF-IDF corpus statistics
- [x] SAXVocabulary (PAD/UNK + corpus-built, save/load)
- [x] WaveletSAXTokenizer (multi-level DWT → token sequences)
- [x] Sequence dataset builder (sliding window context → target)
- [x] WaveletGPT causal transformer (161K params, weight-tied)
- [x] Token prediction evaluation (accuracy, top-3, directional, confusion)
- [x] Multi-asset universe (19 assets, 4 classes)
- [x] Library builder + token pipeline runner
- [x] CLI: sax transform, sax bow, tokenize vocab, tokenize run
- [x] GPU support verified (RTX 2060 SUPER, 34x speedup)
- [x] 167 tests passing (79 new)
- [x] Synthetic validation: 48.4% token acc, 87.7% directional acc

### Phase 3: Real Data Validation & Research Experiments (v0.3.0)
Connected to Massive.com API, ran 78 experiments across 7 research questions on 20 US assets with hourly bars. Final model trained on 2021-2024, evaluated on held-out 2025 data.

- [x] Fetch 5-year hourly+daily data for PHASE3_UNIVERSE (20 US assets, 5 sectors)
- [x] Experiment framework: ExperimentConfig, ExperimentResult, ExperimentRunner, walk-forward splitter
- [x] Walk-forward evaluation (train 2021-2023, test 2024) with bootstrap 95% CIs
- [x] Three baselines: most-frequent-token, persistence, momentum
- [x] C1: SAX alphabet sweep → optimal alphabet_size=7
- [x] C2: DWT level contribution → optimal levels=[1,2,5], levels 3&4 are noise
- [x] C3: Cross-sector transfer → multi-sector training helps 4/6 sectors
- [x] C4: Vocabulary saturation → natural vocab 83 tokens, fully saturated
- [x] C5: Context length sweep → context=16 optimal, 64 hurts
- [x] C6: Regime dependence → consistent 82.7% across regimes, +10.3% over persistence
- [x] C7: Ensemble comparison → P2 dominates P1, no ensemble benefit
- [x] D1: Aggregate results → optimal config per sector
- [x] D2: Train final WaveletGPT on 2021-2024 with optimal config
- [x] D3: Held-out evaluation on 2025 data → 60.8% token acc, 95.8% dir acc
- [x] D4: Generate PHASE3_RESULTS.md (full report)
- [x] E1: Update universe — PHASE3_UNIVERSE is now DEFAULT, Sector enum added
- [x] E2: Update default configs with optimal values
- [x] E3: Update docs with final findings
- [x] 230 tests passing (63 new experiment + universe tests)

### Phase 3 Results vs Success Criteria
- Token accuracy ~56-63% on real data (target was >30%) — **exceeded**
- Directional accuracy ~83-96% (target was >55%) — **exceeded**
- All 5 sectors show above-random prediction — **exceeded** (target was 2+ sectors)
- Multi-level [1,2,5] outperforms single-level — **confirmed** (levels 3&4 removal improved accuracy)
- 2025 held-out: 60.8% token acc, 95.8% dir acc — **no overfitting** (consistent with 2024)

### Phase 4: Hyperparameter Optimization & Multi-Horizon (v0.4.0)
Multi-horizon prediction, Optuna HPO, expanding window validation, price reconstruction.

- [x] F1: Multi-horizon WaveletGPT — predict 1, 2, 4, 8 steps ahead with separate heads
- [x] F2: Expanding/rolling window splitter — multiple train/test splits with aggregated metrics
- [x] F3: Price reconstruction — inverse SAX/PAA to approximate price deltas + directional signals
- [x] F4: Optuna SAX HPO (30 trials) → n_segments=512 (was 256), word_length=4, word_stride=1 confirmed
- [x] F5: Optuna architecture HPO (25 trials) → embed_dim=128, num_layers=6, dropout=0.2
- [x] F6: Per-sector fine-tuning → hurts ALL 6 sectors, cross-sector definitively confirmed
- [x] F7: Multi-horizon experiments → h=1 (68.5%), h=2 (56.3%), h=4+ plateaus at ~41%
- [x] 303 tests passing (73 new)

### Phase 4 Key Findings
- **n_segments=512** dominates 256 — higher SAX resolution captures finer patterns
- **Larger model preferred**: embed_dim=128, 6 layers, dropout=0.2 (was 64, 3 layers, 0.1)
- **Multi-horizon sweet spot is h=1 to h=2**: h=4+ token accuracy plateaus at ~41% (near persistence baseline), but directional accuracy stays 94-97% across all horizons
- **Per-sector fine-tuning hurts ALL sectors**: universal cross-sector model is definitively optimal (resolves Phase 3 C3 inconclusive result for commodity ETFs)
- **Expanding window validation**: results robust across multiple evaluation windows

### Phase 5: Signal Generation & Backtesting (v0.5.0)
Token predictions to trading signals with realistic transaction costs and risk metrics.

- [x] SignalGenerator: softmax probabilities → direction (+1/-1/0) + confidence via quartile aggregation
- [x] Temperature scaling calibration via NLL minimization on validation data
- [x] PositionSizer: fixed, linear, Kelly, fractional Kelly (0.5× Kelly with rolling lookback)
- [x] TransactionCostModel: commission + spread (bps) + slippage (bps); direction changes double costs
- [x] SignalBacktest: per-trade records with entry/exit timestamps, gross/net returns, cost breakdown
- [x] 16 risk metrics: Sharpe, Sortino, Calmar, max drawdown, profit factor, VaR, CVaR, win rate, avg win/loss ratio, expectancy, tail ratio, total/annualized return, volatility, num trades, avg trade return
- [x] CLI: `wavecast signal generate`, `wavecast signal backtest`, `wavecast signal calibrate`
- [x] Signal reporting: formatted text report with metrics + per-trade breakdown
- [x] 349 tests passing (46 new)

### Phase 6: Performance Optimization (v0.6.0)
Rust/PyO3 acceleration, AMP mixed-precision, memory-mapped datasets, batch inference.

- [x] Rust/PyO3 extension crate (`rust/`): `extract_words`, `build_bow`, `build_corpus_tfidf`, `encode_batch`, `build_sliding_windows`
- [x] Python fallback wrapper (`_rust.py`) — graceful degradation when Rust not compiled
- [x] Build system migration: hatchling → maturin (supports mixed Rust+Python packages)
- [x] `sax/bow.py` and `tokenizer/vocabulary.py` delegate to Rust when `HAS_RUST=True`
- [x] AMP mixed-precision: `torch.autocast` + `GradScaler` in WaveletGPT fit/predict — no-op on CPU
- [x] MMapSequenceDataset: zero-copy `.npy` loading via `np.load(mmap_mode='r')`
- [x] BatchPredictor: streaming batch inference with `torch.inference_mode()`, configurable batch_size
- [x] CLI: `--batch-size` and `--use-amp` flags on signal backtest command
- [x] 391 tests passing (42 new)

### Phase 7: Forward Testing & Model Audit (v0.7.0)
Forward testing framework for paper trading, plus comprehensive model audit revealing critical economic validity issues.

- [x] Forward testing module (`src/wavecast/forward/`, 7 files, ~600 LOC)
- [x] `ForwardTestRunner`: fetch → resolve → DWT/SAX/tokenize → predict → log cycle
- [x] `ForwardTestTracker`: JSONL prediction logging, resolution, rolling metrics
- [x] `ForwardTestConfig`: Pydantic BaseSettings, standalone (not in WaveCastConfig)
- [x] Text + JSON report generation
- [x] CLI: `wavecast forward run/status/report/list`
- [x] Training script: `scripts/train_forward_model.py` (20 tickers, 2021-2024)
- [x] Model artifacts saved: `~/.wavecast/models/forward_ready/` (model.pt, config.json, vocabulary.json)
- [x] First live forward test run (AAPL, MSFT, GOOGL, SPY on 1h)
- [x] Bug fixes: 5x intraday buffer, 300 lookback bars (DWT level 5 needs ≥224)
- [x] 33 forward testing tests (424 total)
- [x] **Model audit** (`scripts/model_audit.py`, ~550 LOC, 13 analyses)
- [x] Audit results: `~/.wavecast/audit/model_audit_results.json`
- [x] Handoff: `.claude/handoff/2026-02-16-model-audit-results.md`

### Phase 7 Model Audit — Critical Findings

**The model has NO tradeable edge.** The 95.8% directional accuracy is purely symbolic (compares SAX word ordinals, not price direction). Economic directional accuracy is ~46% — worse than a coin flip.

| Finding | Value |
|---------|-------|
| Token accuracy (confirmed) | 62.4% |
| Economic directional accuracy | ~46.5% (below 50% random) |
| Symbolic directional accuracy | 95.8% (meaningless — ordinal comparison) |
| Persistence predictions | 45.2% of all predictions are "same as last" |
| Persistence token accuracy | 79.2% (just reflects token repetition) |
| Change prediction accuracy | 48.5% (near random) |
| Fixed-sizing Sharpe | -0.5573 |
| Kelly Sharpe | +0.18 (artifact — near-zero positions) |
| Level 5 token persistence | 82.6% identical to previous |
| Confidence → economic direction | FLAT (~46%) across all deciles |
| Random baseline comparison | Model at 84th percentile (doesn't exceed P95) |

**Root cause**: The model learns SAX token statistics, not price dynamics. SAX tokens don't encode price direction — they encode wavelet coefficient levels. High token accuracy is real but economically meaningless.

### Phase 8: D1 Representation — Return Quantile Prediction (v0.8.0)
Pivoted from SAX token prediction to direct return quantile classification. The Phase 7 audit proved SAX tokens encode coefficient levels, not price direction. D1 representation predicts 5-class return quantiles from wavelet coefficient deltas + 4 auxiliary features.

- [x] D1 representation: DWT level-1 coefficient deltas + [coeff_sign, magnitude_zscore, volatility_ratio, approx_direction]
- [x] 5-class return quantile targets (from quantile_boundaries.json)
- [x] WaveletGPT with continuous input mode (no SAX tokenization)
- [x] Economic directional accuracy: **~64%** (vs 46% from SAX pipeline)
- [x] Transition accuracy: 55.2% (genuine, not flat-bias inflated)
- [x] Large-move accuracy: **68.1%** — the tradeable edge
- [x] Model trained on 2021-2025 hourly data, 20 US tickers

### Phase 12: Representation Ceiling Confirmation (v0.12.0)
10 representation change experiments tested whether the D1 ceiling could be broken. 66 total experiments across 5 rounds.

- [x] fracdiff (FAIL), finegrain_7 (INTERESTING), finegrain_11 (INTERESTING)
- [x] context_32 (FAIL), context_48 (FAIL), context_64 (OOM)
- [x] crossscale (FAIL), range_dwt (FAIL)
- [x] regression (INTERESTING — 67% transition, best transition detection)
- [x] transition_head (INTERESTING — minimal effect)
- [x] **Ceiling confirmed at ~64% econ_dir across 66 experiments**
- [x] BOLT information-theoretic analysis: CE is near-optimal for this representation
- [x] Full results: `PHASE12_RESULTS.md`

### Production Deployment
- [x] PnL simulation: 8-config comparison (A1i/A1ii/A2i/A2ii/B1i/B1ii/B2i/B2ii)
- [x] **A2i wins**: large-move magnitude filter, flat sizing, Sharpe +8.40 on 2025 test
- [x] 2026 OOS validation: 66.7% accuracy, Sharpe +10.40, expectancy +0.339% per trade
- [x] Reversal filter invalidated on 2026 OOS (78.9% → 63.8%)
- [x] Production system documented: `docs/PRODUCTION_SYSTEM.md`
- [x] Forward test updated with A2i magnitude filter and dual-track metrics
- [x] Forward test scheduler: `scripts/run_forward_test.sh`

## Current Focus

**Research phase complete.** 66 experiments across 5 rounds confirmed the D1 representation ceiling at ~64% economic directional accuracy. The A2i production system is validated on 2026 OOS data and running forward tests. No further model experiments are warranted on the current data.

## Future Phases

### Phase 9: Forward Test Accumulation
- [ ] Accumulate months of live A2i forward test data
- [ ] Track prediction accuracy, Sharpe, and drawdown over time
- [ ] Monitor for regime drift or performance degradation
- [ ] Test EnCQR overlay (5-model ensemble disagreement) on top of A2i

### Phase 10: New Data Sources (only remaining research lever)
- [ ] Multi-timeframe: combine hourly + daily signals
- [ ] Cross-asset: use sector/market signals as context
- [ ] Non-price data: order flow, options surfaces, sentiment
- [ ] Sub-hourly intervals (15m, 5m) — untested

## Research Questions — Answered (Phase 3)

1. **Optimal SAX granularity**: **alphabet=7 is optimal.** Best directional accuracy across all sectors. Higher alphabets (9, 11) suffer from high UNK rates.
2. **Level contribution**: **Levels [1,2,5] carry the signal.** Levels 3 and 4 are noise — removing them IMPROVES accuracy. Level 5 (coarsest) is the strongest single predictor (75.4%), level 1 (finest) second (59.6%).
3. **Cross-sector transfer**: **Multi-sector training helps 4/6 sectors.** Finance benefits most (+3.8%). Broad ETFs and energy slightly prefer isolation. Single-asset models overfit (~711 samples for 161K params).
4. **Vocabulary saturation**: **Natural vocab is only 83 tokens — fully saturated.** No pruning needed. min_freq=1 outperforms higher thresholds (rare words carry signal).
5. **Context length**: **16 is the sweet spot.** 32 is marginal, 64 hurts (20% fewer training samples outweighs longer context). 8 is too short.
6. **Regime dependence**: **Remarkably consistent across regimes.** 82.7% overall, mean-reverting slightly best (83.4%), trending uncertain (81.8% but wide CI due to only 55 samples). Beats persistence baseline in all regimes (+5-11%).
7. **Ensemble value**: **No benefit.** P2 (WaveletGPT on hourly) massively outperforms P1 (LSTM+XGB on daily) — 82-91% vs 51-57% directional accuracy. Combining them hurts. Error correlation near-zero but P1 signal too weak to contribute.

## Open Research Questions

1. ~~**n_segments sensitivity**~~: **Answered by F4**: n_segments=512 optimal (was 256). Higher resolution helps.
2. ~~**word_length sensitivity**~~: **Answered by F4**: word_length=4 confirmed optimal via Optuna.
3. ~~**Architecture search**~~: **Answered by F5**: embed_dim=128, 6 layers, dropout=0.2 (larger model preferred).
4. ~~**Per-sector fine-tuning**~~: **Answered by F6**: Hurts ALL 6 sectors. Cross-sector definitively confirmed.
5. ~~**Temporal stability**~~: **Answered by D3**: Yes — 60.8% token acc (vs 63.0% on 2024), 95.8% dir acc (vs 99.7%).
6. ~~**Multi-horizon utility**~~: **Answered by Phase 4**: h=1-2 useful for trading signals. h=4+ only useful for directional bias, not token-level prediction.
7. ~~**Signal-to-PnL gap**~~: **Answered by Phase 7 audit**: SignalGenerator + SignalBacktest with transaction costs, position sizing (Kelly), and 16 risk metrics. **Result: negative Sharpe (-0.56). No tradeable edge.** The gap is in the representation (SAX tokens don't encode price direction), not in the signal/backtest pipeline.
8. ~~**Forward test validation**~~: **Answered by Phase 7**: Forward testing framework built and working. First live run succeeded. But the model audit proved the underlying predictions have no economic edge (~46% directional accuracy).
9. ~~**Representation redesign**~~: **Answered by Phase 8 + Phase 12**: Yes — D1 return quantile prediction achieves ~64% economic directional accuracy (vs 46% from SAX). 66 experiments confirmed this as the ceiling for the current data.
10. **Sub-hourly resolution**: Does 15m/5m data improve h=1 signal quality, or does noise dominate? (Untested — potential Phase 10)
11. ~~**Regime-adaptive sizing**~~: **Answered by PnL simulation**: No — flat sizing (A2i) beats magnitude-based sizing (A2ii). Magnitude-based sizing increases max drawdown from -18.3% to -32.1% without improving Sharpe.

## Non-Goals (Explicit)

- **High-frequency trading**: WaveCast targets daily/hourly, not microsecond latency
- **Order execution**: WaveCast generates signals, not orders
- **Broker integration**: No live broker API — forward testing via paper trading only
- **Large language model scale**: WaveletGPT stays under 1M params — this is a specialized micro-model
- **General time series**: Optimized for financial data, not weather/medical/industrial
