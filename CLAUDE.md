# WaveCast

Wavelet-shapelet financial forecasting library.

## Setup

```bash
cd ~/wavecast
source .venv/bin/activate
# API key auto-loaded from .env via python-dotenv
pytest tests/ -q
```

## Project Layout

- `src/wavecast/` — library source (65 files)
- `tests/` — unit + integration tests (88 passing)
- `.env` — contains `MASSIVE_API_KEY` (gitignored)
- `.env.example` — template for required env vars

## Pipeline Stages

```
data → decompose → discover → match → fractal → features → forecast → evaluate
```

1. **data**: Fetch OHLCV via Massive.com API (`fetch_massive`), cache as Parquet
2. **decompose**: DWT via PyWavelets (db4, 5 levels)
3. **discover**: W-TSS shapelet discovery in wavelet domain (z-score → binary → contiguous → IG scoring)
4. **match**: DTW matching against shapelet library via dtaidistance
5. **fractal**: Hurst exponent (wavelet method), MFDFA, cross-scale self-similarity
6. **features**: ~40-50 features from wavelet + shapelet + fractal + market
7. **forecast**: WaveletLSTM (multi-branch PyTorch) + XGBoost + weighted ensemble
8. **evaluate**: RMSE, MAE, directional accuracy, Sharpe, walk-forward backtest

## Key APIs

```python
from wavecast.data.sources import fetch_massive
from wavecast.wavelets.dwt import decompose
from wavecast.shapelets.discovery import discover_shapelets
from wavecast.dtw.matching import match_against_library
from wavecast.fractal.hurst import wavelet_hurst
from wavecast.features.pipeline import FeaturePipeline
from wavecast.models.ensemble import EnsembleModel
from wavecast.pipeline.runner import PipelineRunner
```

## CLI

```bash
wavecast data fetch AAPL --start 2020-01-01 --interval 1d
wavecast discover run AAPL
wavecast match find AAPL --against SPY
wavecast analyze hurst AAPL
wavecast forecast run AAPL --model ensemble
wavecast backtest run AAPL
```

## Data Source

- **Massive.com** (formerly Polygon.io) — `MASSIVE_API_KEY` env var required
- Package: `massive>=2.0` on PyPI
- Auto-pagination via `client.list_aggs()`, timestamps in Unix ms
- Results cached to `~/.wavecast/cache/` as Parquet

## Known Quirks

- `label_returns()` takes price series, computes returns internally — don't pass pre-computed returns
- `ShapeletLibrary.get()` raises on missing ID (doesn't return None)
- `ParquetCache.clear()` clears ALL cached data (no per-ticker clear)
- `FeaturePipeline()` takes no constructor args
- Hurst exponent can exceed 1.0 on integrated processes — mathematically correct, not a bug
- `shape_descriptor()` returns len-1 array (np.diff, no padding)
- `subsequence_search()` raises DTWError when query > series length

## Testing

```bash
pytest tests/ -q                          # all tests
pytest tests/unit/ -q                     # unit only
pytest tests/integration/ -q              # pipeline integration
pytest tests/ --cov=wavecast --cov-report=term-missing  # coverage (53%)
```

## Linting

```bash
ruff check src/ tests/
mypy src/wavecast/
```
