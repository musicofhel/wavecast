# WaveCast Initial Build — 2026-02-15

## What Was Built
Full `wavecast` Python library at `~/wavecast/` — wavelet-shapelet financial forecasting.

## Structure (65 source files, 22 test files, 88 tests passing)
- **core/**: 16 dataclasses (TimeSeries, WaveletDecomposition, Shapelet, HurstResult, etc.), Pydantic config, exceptions
- **data/**: Massive.com fetcher (`fetch_massive`), CSV/Parquet loaders, preprocessing (z-score, log returns, NaN, labeling), Parquet cache, HDF5 shapelet storage
- **wavelets/**: DWT decompose (pywt), CWT scalogram, per-level stats, reconstruction, denoising
- **shapelets/**: W-TSS discovery (z-score→binary→contiguous regions→IG scoring), quality metrics (entropy, IG, F-stat), DTW-based dedup, ShapeletLibrary (in-memory + HDF5 persist)
- **dtw/**: Constrained DTW via dtaidistance, pairwise similarity matrices, subsequence search, ShapeDTW
- **fractal/**: Wavelet-based Hurst exponent, MFDFA (singularity spectrum), cross-scale self-similarity, regime detection
- **features/**: Pipeline combining wavelet (7 per level) + shapelet (DTW distances) + fractal (Hurst, spectrum width) + market (vol, momentum) → ~40-50 features
- **models/**: WaveletLSTM (multi-branch PyTorch, one LSTM per DWT level), XGBoost, weighted ensemble with stacking, model registry
- **evaluation/**: Metrics (RMSE, MAE, dir accuracy, Sharpe, max drawdown), walk-forward backtest, matplotlib reports
- **pipeline/**: Stage orchestration with timing, full runner (data→forecast)
- **cli/**: Typer CLI — `wavecast {data,discover,match,analyze,forecast,library,backtest}`
- **viz/**: Scalogram, shapelet gallery, DTW alignment, forecast plots, fractal plots

## Data Source
- **Massive.com API** (replaced yfinance). Reads `MASSIVE_API_KEY` from env var.
- `fetch_massive(ticker, start, end, interval)` → uses `client.list_aggs()` with auto-pagination
- Intervals: 1m, 5m, 15m, 30m, 1h, 1d, 1w, 1mo
- Results cached to `~/.wavecast/cache/` as Parquet

## Key APIs
```python
from wavecast.data.sources import fetch_massive
from wavecast.wavelets.dwt import decompose, compute_level_stats
from wavecast.shapelets.discovery import discover_shapelets
from wavecast.dtw.matching import match_against_library
from wavecast.fractal.hurst import wavelet_hurst
from wavecast.features.pipeline import FeaturePipeline
from wavecast.models.gradient_boost import GradientBoostModel
from wavecast.models.wavelet_lstm import WaveletLSTM
from wavecast.models.ensemble import EnsembleModel
from wavecast.evaluation.backtest import WalkForwardBacktest
from wavecast.pipeline.runner import PipelineRunner
```

## Setup
```bash
cd ~/wavecast && source .venv/bin/activate
export MASSIVE_API_KEY="..."
pytest tests/ -q  # 88 passed
wavecast --help
```

## Known Gaps / Next Steps
1. **No git commit yet** — files staged, not committed
2. **label_returns()** computes simple returns from prices internally — callers should pass price series, not returns
3. **FeaturePipeline.build_feature_matrix()** doesn't pass pre-computed Hurst/MFDFA/match results to rolling windows (uses decomp-level features only per window)
4. **Hurst exponent** can exceed 1.0 on some data (integrated processes) — not clamped, which is mathematically correct
5. **Coverage at 53%** — CLI, viz, pipeline runner, data sources uncovered (need network/display)
6. **ParquetCache.clear()** takes no args (clears all), no per-ticker clear
7. **ShapeletLibrary.get()** raises on missing ID instead of returning None
8. Phase 2 (FastAPI + Streamlit dashboard) not started
