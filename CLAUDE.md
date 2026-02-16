# WaveCast

Wavelet-shapelet financial forecasting library with SAX tokenization and transformer-based sequence prediction.

## Quick Start

```bash
cd ~/wavecast
source .venv/bin/activate
pytest tests/ -q          # 219 tests, ~7s on GPU
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
  core/         — types, config, exceptions, universe (incl. PHASE3_UNIVERSE)
  data/         — Massive.com fetcher, cache, preprocessing, storage, decomposition cache
  wavelets/     — DWT decompose, CWT scalogram, reconstruction
  shapelets/    — W-TSS discovery, quality metrics, library, clustering
  dtw/          — DTW matching, ShapeDTW, similarity, subsequence search
  fractal/      — Hurst exponent, MFDFA, regime detection, self-similarity, HurstCache
  features/     — pipeline combining wavelet+shapelet+fractal+market+SAX features
  sax/          — PAA, SAX transform, Bag-of-Words + TF-IDF
  tokenizer/    — SAXVocabulary, WaveletSAXTokenizer, sequence dataset builder
  models/       — WaveletLSTM, WaveletGPT, XGBoost, ensemble, registry
  evaluation/   — metrics, walk-forward backtest, token prediction eval
  experiments/  — ExperimentConfig, ExperimentResult, Runner, Splitter, Metrics, Storage
  pipeline/     — stage orchestration, runner, library builder, token pipeline
  cli/          — Typer CLI: data, discover, match, analyze, forecast, library, backtest, sax, tokenize, experiment
  viz/          — scalogram, shapelet gallery, DTW alignment, forecast, fractal plots
tests/
  unit/         — 41 test files, 213 unit tests
  integration/  — 1 pipeline integration test
  fixtures/     — deterministic generators (seed=42)
scripts/        — experiment runners (run_C1.py through run_C7.py), fetch_phase3_universe.py
```

90 source files, 48 test files, 219 tests passing.

## Architecture: Two Pipelines

### Pipeline 1: Wavelet-Shapelet Forecasting (Phase 1)
```
data → decompose → discover → match → fractal → features → forecast → evaluate
```
- Fetch OHLCV → DWT (db4, 5 levels) → shapelet discovery (W-TSS) → DTW matching
- Fractal analysis (Hurst, MFDFA) → ~40-50 feature vector
- Models: WaveletLSTM + XGBoost + weighted ensemble
- Evaluate: RMSE, MAE, directional accuracy, Sharpe, walk-forward backtest

### Pipeline 2: SAX Token Prediction (Phase 2) — **Primary pipeline**
```
data → decompose → SAX → tokenize → train WaveletGPT → evaluate
```
- DWT coefficients → PAA → SAX symbols → sliding window words → vocabulary
- Bag-of-Words + TF-IDF for cross-asset similarity
- WaveletGPT: causal transformer (token+position+level+asset_class embeddings)
- Weight tying between token embedding and output head
- Evaluate: token accuracy, top-3 accuracy, directional accuracy
- Phase 3 proved P2 dominates P1 — no ensemble benefit

### Experiment Framework (Phase 3)
```
ExperimentConfig → ExperimentRunner → walk-forward split → train → evaluate → ExperimentResult
```
- Walk-forward splitter: splits at RAW PRICE level before any DWT/SAX transformation
- ExperimentRunner: full pipeline per-split with bootstrap 95% CIs
- Three baselines: most-frequent-token, persistence, momentum
- Level-0 directional accuracy for multi-level SAX (detail levels use token accuracy only)
- Results stored as JSON in `~/.wavecast/experiments/`

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
from wavecast.core.universe import get_universe, DEFAULT_UNIVERSE, PHASE3_UNIVERSE
from wavecast.pipeline.token_pipeline import TokenPipelineRunner

# Phase 3 Experiments
from wavecast.experiments.config import ExperimentConfig
from wavecast.experiments.result import ExperimentResult
from wavecast.experiments.runner import ExperimentRunner
from wavecast.experiments.splitter import walk_forward_split
from wavecast.experiments.metrics import level0_directional_accuracy, compute_bootstrap_ci, compute_baselines
from wavecast.experiments.storage import save_results, load_results, compare_results
from wavecast.data.decomposition_cache import DecompositionCache
from wavecast.fractal.hurst import rolling_hurst_with_regimes, HurstCache
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
- `ExperimentConfig` — full experiment specification (tickers, interval, SAX/model params, split dates)
- `ExperimentResult` — metrics + CIs + baselines + per-asset/sector/level breakdowns

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
- Cached OHLCV parquets use `{ticker}_1h_ohlcv.parquet` format (6 cols) — ExperimentRunner handles loading
- Rolling Hurst on raw prices always returns H > 1.0 (integrated processes) — use `use_returns=True` for regime detection on log returns
- `PHASE3_UNIVERSE` has 20 US assets (5 sectors); `DEFAULT_UNIVERSE` still has the original 19 (incl. crypto/forex)
- Directional accuracy is level-0 only for multi-level SAX — detail levels represent oscillation magnitude, not price direction

## Testing

```bash
pytest tests/ -q                                          # full suite (~7s GPU)
pytest tests/unit/ -q                                     # unit only (~5s)
pytest tests/unit/test_wavelet_gpt.py -v                  # GPU model tests
pytest tests/unit/test_experiment_runner.py -v             # experiment framework tests
pytest tests/ --cov=wavecast --cov-report=term-missing    # coverage
ruff check src/ tests/                                    # lint
```

## Benchmarks (RTX 2060 SUPER)

- Full test suite: 219 tests in 7s (was 2m20s CPU-only before GPU)
- WaveletGPT 6 tests: 4s (was 2m16s CPU-only)
- WaveletGPT training (161K params, 610 samples, 80 epochs): 3.9s
- SAX+BoW+TF-IDF on 7 assets x 2000 points: <1s
- Full 20-ticker experiment run (hourly, all levels): ~3-4 min
- Phase 3 full experiment suite (78 runs): ~2.5 hours

## Phase 3 Research Results (real data, 20 assets, hourly bars)

**Optimal config**: alphabet=7, levels=[1,2,5], context=16, vocab=100, min_freq=1, cross-sector training

| Question | Answer |
|----------|--------|
| Q1: Alphabet size | **7** — best directional accuracy, good granularity/learnability balance |
| Q2: DWT levels | **[1,2,5]** — levels 3&4 are noise (removing them improves accuracy) |
| Q3: Cross-sector | **Multi-sector helps** 4/6 sectors, biggest gain for finance (+3.8%) |
| Q4: Vocabulary | **100 max, min_freq=1** — natural vocab is only 83 tokens, fully saturated |
| Q5: Context length | **16** — sweet spot; 32 marginal gain, 64 hurts (fewer samples) |
| Q6: Regime | **Consistent** — 82.7% overall, +10.3% over persistence, mean-reverting slightly best |
| Q7: Ensemble | **No benefit** — P2 dominates P1, combining hurts performance |

## Data Source

- **Massive.com** (formerly Polygon.io) — `MASSIVE_API_KEY` env var
- **Plan**: US Stocks, Unlimited API Calls, 5yr history, Minute Aggregates
- Package: `massive>=2.0` on PyPI
- Auto-pagination via `client.list_aggs()`, timestamps in Unix ms
- Results cached to `~/.wavecast/cache/` as Parquet (40 files, 11.9 MB for Phase 3)
- **PHASE3_UNIVERSE**: 20 US assets across 5 sectors (tech, finance, energy, healthcare, broad ETFs, commodity ETFs)
- **DEFAULT_UNIVERSE**: 19 assets across equity/crypto/forex/commodity (Phase 1-2 legacy)
- Hourly bars: ~4,875 per asset for training period (2021-2023)
- All commodity ETFs (GLD, SLV, USO, UNG) verified working on US Stocks plan
