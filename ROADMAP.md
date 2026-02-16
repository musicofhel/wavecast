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

## Current Focus

Phase 3 is complete. Next priorities are Phase 4 (hyperparameter optimization) and Phase 5 (signal generation).

## Future Phases

### Phase 4: Hyperparameter Optimization & Multi-Horizon
- [ ] Multi-horizon prediction: extend WaveletGPT to predict 2, 4, 8 steps ahead (not just next token)
- [ ] Price reconstruction: map SAX token predictions back to approximate price movements via inverse PAA/DWT
- [ ] Expanding window validation: replace single train/test split with rolling/expanding windows
- [ ] Optuna for remaining fixed params: n_segments, word_length, word_stride
- [ ] Optuna for WaveletGPT architecture (embed_dim, num_heads, num_layers)
- [ ] Per-sector fine-tuning (commodity ETFs showed marginal benefit from sector-only training)
- [ ] Regime-conditional models (low priority — Phase 3 C6 showed <2% variation across regimes)

### Phase 5: Signal Generation & Backtesting
- [ ] Token predictions → directional signals (buy/sell/hold)
- [ ] Confidence calibration (softmax probabilities → position sizing)
- [ ] P2-only signal pipeline (Phase 3 proved no ensemble benefit — P1 can be deprecated)
- [ ] Walk-forward backtesting with transaction costs, bid-ask spreads, slippage
- [ ] Risk metrics: max drawdown, Calmar ratio, tail risk
- [ ] Additional asset classes: international equities, fixed income, higher-frequency data

### Phase 6: Performance Optimization
- [ ] Rust acceleration via PyO3:
  - SAX word extraction (sliding window string ops)
  - Bag-of-Words counting (hash map accumulation)
  - Vocabulary encoding (batch string → int lookup)
  - Sequence dataset building (sliding window array construction)
- [ ] Mixed precision training (fp16) for larger models
- [ ] Data loading optimization (memory-mapped datasets)
- [ ] Batch inference for real-time signal generation

### Phase 7: Multi-Timeframe
- [ ] Sub-hourly intervals (15m, 5m) from Massive.com (hourly already validated in Phase 3)
- [ ] Timeframe-aware SAX (different params per interval)
- [ ] Hierarchical model: hourly predictions inform sub-hourly context
- [ ] Streaming mode: online vocabulary updates, incremental training

### Phase 8: Production
- [ ] FastAPI service for real-time predictions
- [ ] Streamlit dashboard for visualization
- [ ] Scheduled retraining pipeline
- [ ] Model versioning + A/B testing
- [ ] Alert system for regime changes
- [ ] Docker deployment

## Research Questions — Answered (Phase 3)

1. **Optimal SAX granularity**: **alphabet=7 is optimal.** Best directional accuracy across all sectors. Higher alphabets (9, 11) suffer from high UNK rates.
2. **Level contribution**: **Levels [1,2,5] carry the signal.** Levels 3 and 4 are noise — removing them IMPROVES accuracy. Level 5 (coarsest) is the strongest single predictor (75.4%), level 1 (finest) second (59.6%).
3. **Cross-sector transfer**: **Multi-sector training helps 4/6 sectors.** Finance benefits most (+3.8%). Broad ETFs and energy slightly prefer isolation. Single-asset models overfit (~711 samples for 161K params).
4. **Vocabulary saturation**: **Natural vocab is only 83 tokens — fully saturated.** No pruning needed. min_freq=1 outperforms higher thresholds (rare words carry signal).
5. **Context length**: **16 is the sweet spot.** 32 is marginal, 64 hurts (20% fewer training samples outweighs longer context). 8 is too short.
6. **Regime dependence**: **Remarkably consistent across regimes.** 82.7% overall, mean-reverting slightly best (83.4%), trending uncertain (81.8% but wide CI due to only 55 samples). Beats persistence baseline in all regimes (+5-11%).
7. **Ensemble value**: **No benefit.** P2 (WaveletGPT on hourly) massively outperforms P1 (LSTM+XGB on daily) — 82-91% vs 51-57% directional accuracy. Combining them hurts. Error correlation near-zero but P1 signal too weak to contribute.

## Open Research Questions (Phase 4+)

1. **n_segments sensitivity**: Fixed at 256 for hourly — is this optimal? Interacts with alphabet_size.
2. **word_length sensitivity**: Fixed at 4 — would 3 or 5 change the vocabulary characteristics?
3. **Architecture search**: embed_dim=64, 4 heads, 3 layers held constant — room for improvement?
4. **Per-sector fine-tuning**: Broad ETFs consistently outperform — would sector-specific heads help?
5. **Temporal stability**: ~~Does the 2024 test accuracy hold on 2025?~~ **Answered by D3**: Yes — 60.8% token acc (vs 63.0% on 2024), 95.8% dir acc (vs 99.7%). Modest degradation, no overfitting.

## Non-Goals (Explicit)

- **High-frequency trading**: WaveCast targets daily/hourly, not microsecond latency
- **Order execution**: WaveCast generates signals, not orders
- **Live trading integration**: No broker API integration planned
- **Large language model scale**: WaveletGPT stays under 1M params — this is a specialized micro-model
- **General time series**: Optimized for financial data, not weather/medical/industrial
