# WaveCast

Wavelet-shapelet financial forecasting library with SAX tokenization and transformer-based sequence prediction.

## Quick Start

```bash
cd ~/wavecast
source .venv/bin/activate
pytest tests/ -q          # 167 tests, ~8s on GPU
ruff check src/ tests/    # 0 errors
```

## Environment

- Python 3.12.12, venv at `.venv/`
- PyTorch 2.10.0+cu128, CUDA 13.1
- GPU: NVIDIA RTX 2060 SUPER (8GB VRAM) — auto-detected via `torch.cuda.is_available()`
- Data API: Massive.com — `MASSIVE_API_KEY` in `.env` (gitignored)
- `.env.example` has the template

## Project Layout

```
src/wavecast/
  core/         — types, config, exceptions, universe
  data/         — Massive.com fetcher, cache, preprocessing, storage
  wavelets/     — DWT decompose, CWT scalogram, reconstruction
  shapelets/    — W-TSS discovery, quality metrics, library, clustering
  dtw/          — DTW matching, ShapeDTW, similarity, subsequence search
  fractal/      — Hurst exponent, MFDFA, regime detection, self-similarity
  features/     — pipeline combining wavelet+shapelet+fractal+market+SAX features
  sax/          — PAA, SAX transform, Bag-of-Words + TF-IDF
  tokenizer/    — SAXVocabulary, WaveletSAXTokenizer, sequence dataset builder
  models/       — WaveletLSTM, WaveletGPT, XGBoost, ensemble, registry
  evaluation/   — metrics, walk-forward backtest, token prediction eval
  pipeline/     — stage orchestration, runner, library builder, token pipeline
  cli/          — Typer CLI: data, discover, match, analyze, forecast, library, backtest, sax, tokenize
  viz/          — scalogram, shapelet gallery, DTW alignment, forecast, fractal plots
tests/
  unit/         — 35 test files, 161 unit tests
  integration/  — 1 pipeline integration test
  fixtures/     — deterministic generators (seed=42)
```

81 source files, 42 test files, 167 tests passing.

## Architecture: Two Pipelines

### Pipeline 1: Wavelet-Shapelet Forecasting (Phase 1)
```
data → decompose → discover → match → fractal → features → forecast → evaluate
```
- Fetch OHLCV → DWT (db4, 5 levels) → shapelet discovery (W-TSS) → DTW matching
- Fractal analysis (Hurst, MFDFA) → ~40-50 feature vector
- Models: WaveletLSTM + XGBoost + weighted ensemble
- Evaluate: RMSE, MAE, directional accuracy, Sharpe, walk-forward backtest

### Pipeline 2: SAX Token Prediction (Phase 2)
```
data → decompose → SAX → tokenize → train WaveletGPT → evaluate
```
- DWT coefficients → PAA → SAX symbols → sliding window words → vocabulary
- Bag-of-Words + TF-IDF for cross-asset similarity
- WaveletGPT: causal transformer (token+position+level+asset_class embeddings)
- Weight tying between token embedding and output head
- Evaluate: token accuracy, top-3 accuracy, directional accuracy

## Key APIs

```python
# Phase 1
from wavecast.data.sources import fetch_massive
from wavecast.wavelets.dwt import decompose
from wavecast.shapelets.discovery import discover_shapelets
from wavecast.dtw.matching import match_against_library
from wavecast.fractal.hurst import wavelet_hurst
from wavecast.features.pipeline import FeaturePipeline
from wavecast.models.ensemble import EnsembleModel
from wavecast.pipeline.runner import PipelineRunner

# Phase 2
from wavecast.sax.sax import sax_transform, sax_distance
from wavecast.sax.bow import extract_words, build_bow, build_corpus_tfidf
from wavecast.tokenizer.vocabulary import SAXVocabulary
from wavecast.tokenizer.tokenizer import WaveletSAXTokenizer
from wavecast.tokenizer.dataset import build_sequence_dataset
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.evaluation.token_eval import evaluate_token_predictions
from wavecast.core.universe import get_universe, DEFAULT_UNIVERSE
from wavecast.pipeline.token_pipeline import TokenPipelineRunner
```

## CLI

```bash
# Data
wavecast data fetch AAPL --start 2020-01-01 --interval 1d

# Phase 1
wavecast discover run AAPL
wavecast match find AAPL --against SPY
wavecast analyze hurst AAPL
wavecast forecast run AAPL --model ensemble
wavecast library build --universe default
wavecast backtest run AAPL

# Phase 2
wavecast sax transform AAPL --segments 64 --alphabet 7
wavecast sax bow AAPL --word-length 4
wavecast tokenize vocab --universe default
wavecast tokenize run --universe default --epochs 50
```

## Config System

Pydantic `BaseSettings` hierarchy in `core/config.py`:
- `WaveletConfig` — wavelet family, decomposition level, mode
- `ShapeletConfig` — z-threshold, min_length, top_k, IG minimum
- `DTWConfig` — window, pruning, normalization
- `FractalConfig` — Hurst method/window, MFDFA q-range, regime thresholds
- `ModelConfig` — LSTM/XGBoost hyperparams, ensemble weights
- `BacktestConfig` — capital, position size, commission, walk-forward splits
- `SAXConfig` — n_segments, alphabet_size, word_length, word_stride
- `TokenizerConfig` — context_length, min_word_freq, max_vocab_size
- `SequenceModelConfig` — embed_dim, num_heads, num_layers, dropout, epochs, lr, patience
- `WaveCastConfig` — top-level aggregator with data/library/model/cache dirs

## Type System

Core dataclasses in `core/types.py`:
- `TimeSeries`, `WaveletDecomposition`, `LevelStats`
- `Shapelet`, `ShapeletMatch`, `MatchResult`
- `HurstResult`, `MFDFAResult`, `SelfSimilarityResult`, `RegimeDetection`
- `FeatureVector` (wavelet + shapelet + fractal + market + sax features)
- `SAXRepresentation`, `SAXWord`, `TokenSequence`, `MultiLevelTokenSequence`
- Enums: `MarketLabel`, `RegimeType`, `AssetClass`

## Exception Hierarchy

```
WaveCastError
├── DataError → DataNotFoundError
├── DecompositionError
├── ShapeletError → ShapeletLibraryError
├── DTWError
├── FractalError
├── ModelError → ModelNotTrainedError, SequenceModelError
├── PipelineError
├── SAXError
├── TokenizerError
└── ConfigError
```

## Known Quirks

- `label_returns()` takes price series, computes returns internally — don't pass pre-computed returns
- `ShapeletLibrary.get()` raises on missing ID (doesn't return None)
- `ParquetCache.clear()` clears ALL cached data (no per-ticker clear)
- `FeaturePipeline()` takes no constructor args
- Hurst exponent can exceed 1.0 on integrated processes — mathematically correct, not a bug
- `shape_descriptor()` returns len-1 array (np.diff, no padding)
- `subsequence_search()` raises DTWError when query > series length
- `AssetClass.value` returns strings ('equity', etc.) not ints — use a mapping dict for numeric IDs
- `SAXVocabulary` supports `len()` and `.size` property — both return total including PAD+UNK
- `build_sequence_dataset()` expects `MultiLevelTokenSequence` objects, not raw dicts
- WaveletGPT X format: `[context_token_0, ..., context_token_{L-1}, level_id, asset_class_id]`
- Weight tying means WaveletGPTNet vocab_size affects both embedding and output head simultaneously

## Testing

```bash
pytest tests/ -q                                          # full suite (~8s GPU)
pytest tests/unit/ -q                                     # unit only (~4s)
pytest tests/unit/test_wavelet_gpt.py -v                  # GPU model tests
pytest tests/ --cov=wavecast --cov-report=term-missing    # coverage
ruff check src/ tests/                                    # lint
```

## Benchmarks (RTX 2060 SUPER)

- Full test suite: 167 tests in 8.4s (was 2m20s CPU-only)
- WaveletGPT 6 tests: 4s (was 2m16s CPU-only)
- WaveletGPT training (161K params, 610 samples, 80 epochs): 3.9s
- SAX+BoW+TF-IDF on 7 assets x 2000 points: <1s

## Data Source

- **Massive.com** (formerly Polygon.io) — `MASSIVE_API_KEY` env var
- Package: `massive>=2.0` on PyPI
- Auto-pagination via `client.list_aggs()`, timestamps in Unix ms
- Results cached to `~/.wavecast/cache/` as Parquet
- Universe: 19 default assets across equity/crypto/forex/commodity
