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

## Current Focus

### Phase 3: Real Data Validation
Connect to Massive.com API and validate on actual market data.

**Goal**: Prove the SAX → WaveletGPT pipeline works on real financial data with multi-level DWT decomposition across the full 19-asset universe.

- [ ] Fetch 5-year daily data for DEFAULT_UNIVERSE (19 assets)
- [ ] Multi-level SAX: run SAX on each DWT level (5 levels × 19 assets)
- [ ] Build real vocabulary from actual market data
- [ ] Train WaveletGPT on real multi-level, multi-asset token sequences
- [ ] Walk-forward evaluation (train on 2019-2023, test on 2024-2025)
- [ ] Compare: single-level vs multi-level SAX accuracy
- [ ] Compare: per-asset vs cross-asset training
- [ ] Cross-asset similarity analysis (TF-IDF cosine) on real data
- [ ] Report: which asset classes / wavelet levels are most predictable?

### Phase 3 Success Criteria
- Token accuracy > 30% on real data (vs <1% random baseline)
- Directional accuracy > 55% (above coin flip)
- At least 2 asset classes show consistent above-random prediction
- Multi-level SAX outperforms single-level

## Future Phases

### Phase 4: Hyperparameter Optimization
- [ ] Optuna integration for SAX params (n_segments, alphabet_size, word_length)
- [ ] Optuna for WaveletGPT architecture (embed_dim, num_heads, num_layers)
- [ ] Cross-validation strategy for financial time series (purged walk-forward)
- [ ] Per-asset-class optimal configurations
- [ ] Vocabulary size sensitivity analysis

### Phase 5: Signal Generation
- [ ] Token predictions → directional signals (buy/sell/hold)
- [ ] Confidence calibration (softmax probabilities → position sizing)
- [ ] Combine Pipeline 1 (ensemble) + Pipeline 2 (WaveletGPT) predictions
- [ ] Signal backtesting with transaction costs
- [ ] Risk metrics: max drawdown, Calmar ratio, tail risk

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
- [ ] Intraday intervals (1h, 15m, 5m) from Massive.com
- [ ] Timeframe-aware SAX (different params per interval)
- [ ] Hierarchical model: daily predictions inform intraday context
- [ ] Streaming mode: online vocabulary updates, incremental training

### Phase 8: Production
- [ ] FastAPI service for real-time predictions
- [ ] Streamlit dashboard for visualization
- [ ] Scheduled retraining pipeline
- [ ] Model versioning + A/B testing
- [ ] Alert system for regime changes
- [ ] Docker deployment

## Research Questions

These are open questions that the real-data validation (Phase 3) should help answer:

1. **Optimal SAX granularity**: Is alphabet_size=7 better than 5? Does it vary by asset class?
2. **Level contribution**: Which DWT levels carry the most predictable SAX patterns? (Hypothesis: levels 2-4 for daily data)
3. **Cross-asset transfer**: Does training on equities help predict crypto, or vice versa?
4. **Vocabulary saturation**: At what point do more words stop helping? Is 200 enough or do we need 500+?
5. **Context length**: 16 tokens worked for synthetic data. Does real data need 32 or 64?
6. **Regime dependence**: Does WaveletGPT accuracy change during trending vs mean-reverting periods?
7. **Ensemble value**: Does combining Pipeline 1 + Pipeline 2 beat either alone?

## Non-Goals (Explicit)

- **High-frequency trading**: WaveCast targets daily/hourly, not microsecond latency
- **Order execution**: WaveCast generates signals, not orders
- **Live trading integration**: No broker API integration planned
- **Large language model scale**: WaveletGPT stays under 1M params — this is a specialized micro-model
- **General time series**: Optimized for financial data, not weather/medical/industrial
