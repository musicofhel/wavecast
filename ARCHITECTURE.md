# WaveCast Architecture

## Design Philosophy

WaveCast treats financial time series as **multi-scale symbolic objects**. Instead of feeding raw prices into a black box, we:

1. **Decompose** price into wavelet frequency bands (DWT) — isolating trend, cycles, noise
2. **Symbolize** each band independently (SAX) — compressing continuous signals to discrete alphabets
3. **Discover patterns** in both domains — shapelets in wavelet space, word patterns in symbolic space
4. **Predict** using models matched to each representation — ensemble for features, transformer for tokens

Phase 3 experiments proved Pipeline 2 (SAX token prediction) dominates Pipeline 1 (feature-based ensemble) — 82-91% vs 51-57% directional accuracy. Pipeline 2 is the primary production path; Pipeline 1 remains for research and feature analysis.

## Data Flow

```
                        ┌─────────────────────────────────────────────┐
                        │              Raw OHLCV Data                  │
                        │         (Massive.com → Parquet cache)        │
                        └──────────────────┬──────────────────────────┘
                                           │
                                    ┌──────▼──────┐
                                    │     DWT      │
                                    │  (db4, L=5)  │
                                    └──────┬──────┘
                                           │
                          ┌────────────────┤────────────────┐
                          │                │                │
                   ┌──────▼──────┐  ┌──────▼──────┐  ┌─────▼──────┐
                   │  Shapelets  │  │  SAX + BoW  │  │  Fractal   │
                   │  (W-TSS)    │  │  (per level) │  │  (Hurst,   │
                   │             │  │             │  │   MFDFA)    │
                   └──────┬──────┘  └──────┬──────┘  └─────┬──────┘
                          │                │                │
              ┌───────────┤                │                │
              │           │         ┌──────▼──────┐         │
              │    ┌──────▼──────┐  │  Tokenize   │  ┌─────▼──────┐
              │    │  DTW Match  │  │  + Vocab     │  │  Feature   │
              │    └──────┬──────┘  └──────┬──────┘  │  Extraction│
              │           │                │         └─────┬──────┘
              │           │                │               │
              │    ┌──────▼────────────────▼───────────────▼──────┐
              │    │              Feature Vector (~50 dims)        │
              │    └──────────────────┬───────────────────────────┘
              │                       │
              │               ┌───────▼───────┐
              │               │   Ensemble    │  ← Pipeline 1
              │               │  (LSTM+XGB)   │
              │               └───────┬───────┘
              │                       │
              │                 RMSE, Sharpe,
              │                 Dir. Accuracy
              │
              │         ┌─────────────────────┐
              └────────►│    Sequence Model    │  ← Pipeline 2
                        │   (WaveletGPT)      │
                        └──────────┬──────────┘
                                   │
                            Token Accuracy,
                            Top-3, Directional
```

## Module Dependency Graph

```
core (types, config, exceptions, universe)
  ↑
data (sources, cache, preprocessing, storage)
  ↑
wavelets (dwt, cwt, reconstruction, features)
  ↑
├── shapelets (discovery, quality, library, clustering)
│     ↑
│     dtw (matching, similarity, subsequence, shape_dtw)
│
├── sax (paa, sax, bow)
│     ↑
│     tokenizer (vocabulary, tokenizer, dataset)
│
└── fractal (hurst, mfdfa, self_similarity, regime)

features (pipeline: wavelet + shapelet + fractal + market + sax)
  ↑
models (wavelet_lstm, wavelet_gpt, gradient_boost, ensemble, registry)
  ↑
evaluation (metrics, backtest, token_eval, reporting)
  ↑
├── pipeline (stages, runner, library_builder, token_pipeline)
│
└── experiments (config, result, splitter, runner, metrics, storage)
      ↑
cli (app, commands/* incl. experiment)
```

## Key Design Decisions

### Why DWT over raw prices?
DWT decomposes price into frequency bands with known periodicity. Level 1 ≈ 2-day cycles, level 5 ≈ 32-day trends. Shapelets found in level 3 (weekly) are more meaningful than in raw noise. SAX on DWT coefficients captures regime changes at each scale independently.

### Why SAX over learned embeddings?
SAX provides **interpretable** symbols. When the model predicts `bbbb→cccc`, you can read that as "continued low → transition to mid" at that wavelet scale. Breakpoints from the normal distribution give theoretically grounded quantization. The vocabulary is naturally small (~83 tokens with alphabet=7, word_length=4) — fully saturated and ideal for a lightweight transformer.

### Why weight tying in WaveletGPT?
With vocab_size ~83 (real data) to ~154 (synthetic), the embedding and output matrices are the largest parameter blocks. Tying them (a) regularizes the model, (b) keeps params under 200K — important for limited training data per asset, (c) forces the embedding space to be predictive, not just descriptive.

### Why multi-asset training?
Financial assets share structural patterns across sectors. A trending equity and a trending commodity produce similar SAX words at comparable wavelet scales. Cross-asset training gives the model more examples of each regime — Phase 3 confirmed this helps 4/6 sectors, with finance gaining +3.8%. The `sector_embed` (formerly `asset_class_embed`) lets it learn sector-specific adjustments.

### Why Bag-of-Words + TF-IDF?
BoW/TF-IDF over SAX words gives a fixed-size feature vector per asset, enabling:
- Cross-asset similarity (cosine distance between TF-IDF vectors)
- Feature engineering (SAX features concatenated into FeatureVector for the ensemble)
- Interpretable asset clustering (which assets have similar symbolic structure?)

### Why levels [1,2,5] only? (Phase 3 finding)
Level 5 (coarsest, ~32-day trends) is the strongest single predictor at 75.4% token accuracy. Level 1 (finest, ~2-day cycles) adds complementary high-frequency signal at 59.6%. Level 2 contributes marginally. Levels 3 and 4 are pure noise — removing them from the model actually improves accuracy by 1-2%. This suggests the mid-frequency bands carry overlapping information that confuses the transformer.

### Why P2 over P1? (Phase 3 finding)
Pipeline 1 (LSTM+XGBoost on daily features) achieves 51-57% directional accuracy on real data — barely above coin flip. Pipeline 2 (WaveletGPT on hourly SAX tokens) achieves 82-91%. The key advantage: hourly resolution gives P2 ~6.5x more training data, and the symbolic representation captures regime transitions that raw features miss. Combining P1+P2 in an ensemble actually hurts — P1's noise corrupts P2's strong signal.

### Why split before transform?
Walk-forward splitting must happen at the raw price level, before DWT decomposition or SAX transformation. If you decompose first and then split, the z-normalization statistics leak future information into the training set. Each split gets independently decomposed, normalized, and tokenized. The vocabulary is built from training data only — test sequences get UNK tokens for unseen words.

## Model Architecture: WaveletGPT

```
Input: [token_ids (B, L), level_ids (B,), sector_ids (B,)]
  │
  ├─ token_embed(token_ids)       → (B, L, D)
  ├─ pos_embed(0..L-1)            → (B, L, D)    sum
  ├─ level_embed(level_ids)       → (B, 1, D) broadcast ──→ (B, L, D)
  └─ sector_embed(sector_ids)     → (B, 1, D) broadcast ──→ (B, L, D)
                                                            │
                                          ┌─────────────────▼─────────────────┐
                                          │  TransformerEncoder (3 layers)     │
                                          │  (causal mask, norm_first=True)    │
                                          └─────────────────┬─────────────────┘
                                                            │
                                                     LayerNorm(D)
                                                            │
                                                  x[:, -1, :] → (B, D)
                                                            │
                                                   Linear(D, V) ← weight tied with token_embed
                                                            │
                                                     logits (B, V)
```

- D=64, H=4 heads, 3 layers, dropout=0.1
- ~161K parameters (vocab_size=154 synthetic, ~83 real data)
- 7 sector embeddings (tech, finance, energy, healthcare, broad ETF, commodity ETF + legacy)
- Causal masking ensures autoregressive prediction
- Only last position used for next-token prediction

## Scaling Considerations

### Phase 2 (synthetic validation)
- 7 assets, 2000 points each, single wavelet level
- 763 samples, 154-token vocabulary
- Training: 3.9s on RTX 2060 SUPER

### Phase 3 (real data — complete)
- 20 US assets (DEFAULT_UNIVERSE), 5-year hourly data (~4875 bars/asset), 6 sectors
- 3 wavelet levels (1, 2, 5) per asset — levels 3&4 excluded as noise
- 23,700 training samples, 83-token vocabulary (naturally saturated)
- Training (full 20-asset run): ~3-4 min on RTX 2060 SUPER
- Full experiment suite (78 runs): ~2.5 hours
- **2025 held-out**: 60.8% token accuracy, 95.8% directional accuracy (no overfitting)

### Future (Phase 4+ — larger universe + sub-hourly)
- 50+ assets, sub-hourly intervals (15m, 5m)
- Multi-horizon prediction (2, 4, 8 steps ahead)
- May need: gradient checkpointing, mixed precision (fp16), larger embed_dim
- RTX 2060 SUPER 8GB should handle up to ~1M params comfortably

## File Organization Principles

1. **One concept per file** — `paa.py`, `sax.py`, `bow.py` not `sax_utils.py`
2. **Types in core, logic in modules** — dataclasses in `types.py`, algorithms in domain modules
3. **Config is declarative** — Pydantic BaseSettings, not scattered constants
4. **Exceptions are specific** — `SAXError`, `TokenizerError`, not bare `ValueError`
5. **Tests mirror source** — `test_paa.py` tests `paa.py`, fixtures in `conftest.py`
6. **CLI is thin** — commands call pipeline functions, no business logic in CLI layer
