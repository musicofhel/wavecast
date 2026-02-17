# WaveCast Architecture

## Design Philosophy

WaveCast treats financial time series as **multi-scale symbolic objects**. Instead of feeding raw prices into a black box, we:

1. **Decompose** price into wavelet frequency bands (DWT) — isolating trend, cycles, noise
2. **Symbolize** each band independently (SAX) — compressing continuous signals to discrete alphabets
3. **Discover patterns** in both domains — shapelets in wavelet space, word patterns in symbolic space
4. **Predict** using models matched to each representation — ensemble for features, transformer for tokens

Phase 3 experiments proved Pipeline 2 (SAX token prediction) dominates Pipeline 1 (feature-based ensemble) — 82-91% vs 51-57% symbolic directional accuracy. Pipeline 2 is the primary production path; Pipeline 1 remains for research and feature analysis. **NOTE (Phase 7 audit)**: The "directional accuracy" metric compares SAX word ordinals, not actual price direction. Economic directional accuracy is ~46% — below random. See CLAUDE.md "Model Audit Results" and `.claude/handoff/2026-02-16-model-audit-results.md`.

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
data (sources, cache, preprocessing, storage, mmap_dataset)
  ↑
wavelets (dwt, cwt, reconstruction, features)
  ↑
├── shapelets (discovery, quality, library, clustering)
│     ↑
│     dtw (matching, similarity, subsequence, shape_dtw)
│
├── sax (paa, sax, bow ←[Rust], reconstruction)
│     ↑
│     tokenizer (vocabulary ←[Rust], tokenizer, dataset)
│
└── fractal (hurst, mfdfa, self_similarity, regime)

features (pipeline: wavelet + shapelet + fractal + market + sax)
  ↑
models (wavelet_lstm, wavelet_gpt [AMP], gradient_boost, ensemble, registry, batch_inference)
  ↑
evaluation (metrics, backtest, token_eval, reporting)
  ↑
├── signals (generator, backtest, position, costs, types, config)
│
├── pipeline (stages, runner, library_builder, token_pipeline)
│
└── experiments (config, result, splitter, runner, metrics, storage, hpo)
      ↑
_rust.py (Rust/PyO3 fallback wrapper)
      ↑
cli (app, commands/* incl. experiment, signal)
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
Pipeline 1 (LSTM+XGBoost on daily features) achieves 51-57% symbolic directional accuracy on real data — barely above coin flip. Pipeline 2 (WaveletGPT on hourly SAX tokens) achieves 82-91% symbolic directional accuracy. The key advantage: hourly resolution gives P2 ~6.5x more training data, and the symbolic representation captures regime transitions that raw features miss. Combining P1+P2 in an ensemble actually hurts — P1's noise corrupts P2's strong signal. **Phase 7 audit caveat**: P2's higher symbolic accuracy does NOT translate to economic value — both pipelines have negative Sharpe ratios. The representation (SAX on coefficient levels) is the bottleneck.

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

- D=128, H=4 heads, 6 layers, dropout=0.2 (Phase 4 Optuna-optimized; was D=64, 3 layers, 0.1)
- ~600K parameters (vocab_size=154 synthetic, ~83 real data)
- 7 sector embeddings (tech, finance, energy, healthcare, broad ETF, commodity ETF + legacy)
- Causal masking ensures autoregressive prediction
- Only last position used for next-token prediction

### Multi-Horizon Heads (Phase 4)

```
                           x[:, -1, :] → (B, D)
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
             Linear(D, V)  Linear(D, V)  Linear(D, V)
             h=1 (tied)    h=2 (indep)   h=4 (indep)  ...
                    │             │             │
              logits_h1     logits_h2     logits_h4
```

- `prediction_horizons=[1,2,4,8]` — configurable list
- h=1 head: weight-tied with token embedding (regularized)
- h=2,4,8 heads: independent Linear(D, V) (different distributions per horizon)
- Multi-horizon training: dataset provides `targets={1: t1, 2: t2, ...}`, loss averaged across horizons
- Phase 4 finding: h=1 (68.5% acc), h=2 (56.3%), h=4+ plateaus at ~41%

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
- **2025 held-out**: 60.8% token accuracy, 95.8% symbolic directional accuracy (no overfitting). **Phase 7 audit: economic directional accuracy is ~46% — no tradeable edge.**

### Phase 4 (HPO + multi-horizon — complete)
- Optuna-optimized: n_segments=512, embed_dim=128, 6 layers, dropout=0.2 (~600K params)
- Multi-horizon: h=1,2,4,8 with separate classification heads
- Expanding/rolling window validation for robustness
- Price reconstruction: SAX tokens → approximate price deltas via inverse PAA
- Per-sector fine-tuning: hurts ALL 6 sectors — cross-sector definitively confirmed
- RTX 2060 SUPER handles 600K params comfortably (10s training for 80 epochs)

### Phase 5 (Signal Generation — complete)
- SignalGenerator converts softmax probabilities → trading signals (direction + confidence)
- SignalBacktest with TransactionCostModel and PositionSizer (Kelly, fractional Kelly)
- 16 risk metrics: Sharpe, Sortino, Calmar, VaR, CVaR, win rate, expectancy, etc.
- Per-trade records with cost breakdown (commission, spread, slippage)

### Phase 6 (Performance — complete)
- Rust/PyO3 extension: 5 hot-path functions accelerated (extract_words, build_bow, encode_batch, etc.)
- AMP mixed-precision: torch.autocast + GradScaler in WaveletGPT — no-op on CPU
- MMapSequenceDataset: zero-copy .npy loading for large datasets
- BatchPredictor: streaming batch inference with torch.inference_mode()
- Build: maturin (Rust 1.93), Python fallback when Rust unavailable

### Future (Phase 7+ — multi-timeframe + forward testing)
- Paper trading forward test on live hourly data
- Sub-hourly intervals (15m, 5m) from Massive.com
- May need: gradient checkpointing for larger universe
- RTX 2060 SUPER 8GB handles 600K params comfortably; can scale to ~1M

## File Organization Principles

1. **One concept per file** — `paa.py`, `sax.py`, `bow.py` not `sax_utils.py`
2. **Types in core, logic in modules** — dataclasses in `types.py`, algorithms in domain modules
3. **Config is declarative** — Pydantic BaseSettings, not scattered constants
4. **Exceptions are specific** — `SAXError`, `TokenizerError`, not bare `ValueError`
5. **Tests mirror source** — `test_paa.py` tests `paa.py`, fixtures in `conftest.py`
6. **CLI is thin** — commands call pipeline functions, no business logic in CLI layer
