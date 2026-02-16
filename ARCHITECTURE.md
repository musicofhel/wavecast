# WaveCast Architecture

## Design Philosophy

WaveCast treats financial time series as **multi-scale symbolic objects**. Instead of feeding raw prices into a black box, we:

1. **Decompose** price into wavelet frequency bands (DWT) — isolating trend, cycles, noise
2. **Symbolize** each band independently (SAX) — compressing continuous signals to discrete alphabets
3. **Discover patterns** in both domains — shapelets in wavelet space, word patterns in symbolic space
4. **Predict** using models matched to each representation — ensemble for features, transformer for tokens

This dual-pipeline architecture lets us combine statistical feature engineering (Phase 1) with learned sequence prediction (Phase 2) — each captures structure the other misses.

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
pipeline (stages, runner, library_builder, token_pipeline)
  ↑
cli (app, commands/*)
```

## Key Design Decisions

### Why DWT over raw prices?
DWT decomposes price into frequency bands with known periodicity. Level 1 ≈ 2-day cycles, level 5 ≈ 32-day trends. Shapelets found in level 3 (weekly) are more meaningful than in raw noise. SAX on DWT coefficients captures regime changes at each scale independently.

### Why SAX over learned embeddings?
SAX provides **interpretable** symbols. When the model predicts `bbbb→cccc`, you can read that as "continued low → transition to mid" at that wavelet scale. Breakpoints from the normal distribution give theoretically grounded quantization. The vocabulary is small enough (~100-300 words) for a lightweight transformer.

### Why weight tying in WaveletGPT?
With vocab_size ~150, the embedding and output matrices are the largest parameter blocks. Tying them (a) regularizes the model, (b) keeps params under 200K — important for limited training data per asset, (c) forces the embedding space to be predictive, not just descriptive.

### Why multi-asset training?
Financial assets share structural patterns across classes. A trending equity and a trending commodity produce similar SAX words at comparable wavelet scales. Cross-asset training gives the model more examples of each regime. The `asset_class_embed` lets it learn class-specific adjustments.

### Why Bag-of-Words + TF-IDF?
BoW/TF-IDF over SAX words gives a fixed-size feature vector per asset, enabling:
- Cross-asset similarity (cosine distance between TF-IDF vectors)
- Feature engineering (SAX features concatenated into FeatureVector for the ensemble)
- Interpretable asset clustering (which assets have similar symbolic structure?)

## Model Architecture: WaveletGPT

```
Input: [token_ids (B, L), level_ids (B,), asset_class_ids (B,)]
  │
  ├─ token_embed(token_ids)       → (B, L, D)
  ├─ pos_embed(0..L-1)            → (B, L, D)    sum
  ├─ level_embed(level_ids)       → (B, 1, D) broadcast ──→ (B, L, D)
  └─ asset_class_embed(ac_ids)    → (B, 1, D) broadcast ──→ (B, L, D)
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
- ~161K parameters at vocab_size=154
- Causal masking ensures autoregressive prediction
- Only last position used for next-token prediction

## Scaling Considerations

### Current (Phase 2 — synthetic validation)
- 7 assets, 2000 points each, single wavelet level
- 763 samples, 154-token vocabulary
- Training: 3.9s on RTX 2060 SUPER

### Target (Phase 3 — real data)
- 19 assets (DEFAULT_UNIVERSE), 5+ years daily data (~1300 pts each)
- 5 wavelet levels per asset = 5x token sequences per asset
- Estimated: ~5000-15000 samples, ~300-500 token vocabulary
- Training: ~30-60s on RTX 2060 SUPER (extrapolating from benchmarks)

### Future (Phase 4+ — intraday + larger universe)
- 50+ assets, intraday intervals (1h, 15m)
- Longer sequences → larger context windows
- May need: gradient checkpointing, mixed precision (fp16), larger embed_dim
- RTX 2060 SUPER 8GB should handle up to ~1M params comfortably

## File Organization Principles

1. **One concept per file** — `paa.py`, `sax.py`, `bow.py` not `sax_utils.py`
2. **Types in core, logic in modules** — dataclasses in `types.py`, algorithms in domain modules
3. **Config is declarative** — Pydantic BaseSettings, not scattered constants
4. **Exceptions are specific** — `SAXError`, `TokenizerError`, not bare `ValueError`
5. **Tests mirror source** — `test_paa.py` tests `paa.py`, fixtures in `conftest.py`
6. **CLI is thin** — commands call pipeline functions, no business logic in CLI layer
