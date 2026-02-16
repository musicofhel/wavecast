# WaveCast Glossary

Quick reference for the domain-specific terms used throughout the codebase.

## Wavelet Domain

| Term | Definition | Where in code |
|------|-----------|---------------|
| **DWT** | Discrete Wavelet Transform. Decomposes a signal into frequency bands (approximation + detail coefficients at each level). | `wavelets/dwt.py` |
| **CWT** | Continuous Wavelet Transform. Produces a 2D scalogram (scale × time). Used for visualization, not features. | `wavelets/cwt.py` |
| **Level** | Decomposition depth. Level 1 = highest frequency (~2-day cycles for daily data), Level 5 = lowest (~32-day trends). | `WaveletConfig.level` |
| **db4** | Daubechies-4 wavelet. Default mother wavelet — good balance of smoothness and compact support. | `WaveletConfig.wavelet` |
| **Approximation coefficients** | Low-frequency component at a given level. The "trend" at that scale. | `WaveletDecomposition.coefficients[0]` |
| **Detail coefficients** | High-frequency component at a given level. The "oscillations" at that scale. | `WaveletDecomposition.coefficients[1:]` |
| **Reconstruction** | Inverse DWT — rebuilds signal from coefficients. Can selectively zero out levels for denoising. | `wavelets/reconstruction.py` |

## Shapelet Domain

| Term | Definition | Where in code |
|------|-----------|---------------|
| **Shapelet** | A short subsequence that is discriminative for a time series class. Found in wavelet coefficient space. | `core/types.py:Shapelet` |
| **W-TSS** | Wavelet-domain Time Series Shapelet discovery. Applies z-score → binarize → extract contiguous regions → score by information gain. | `shapelets/discovery.py` |
| **Information Gain (IG)** | How well a shapelet separates classes. Higher = more discriminative. | `shapelets/quality.py` |
| **ShapeletLibrary** | Collection of discovered shapelets with HDF5 persistence. Supports queries by level, ticker, IG threshold. | `shapelets/library.py` |

## DTW Domain

| Term | Definition | Where in code |
|------|-----------|---------------|
| **DTW** | Dynamic Time Warping. Elastic distance measure that aligns two time series, tolerating speed differences. | `dtw/matching.py` |
| **ShapeDTW** | Shape-aware DTW variant that matches on shape descriptors (slopes, derivatives) instead of raw values. | `dtw/shape_dtw.py` |
| **Warping path** | The alignment between two series found by DTW. Tells you which points in series A correspond to points in series B. | `ShapeletMatch.warping_path` |
| **MINDIST** | Lower-bound distance for SAX strings. Uses breakpoint lookup table — faster than full DTW. | `sax/sax.py:sax_distance` |

## Fractal Domain

| Term | Definition | Where in code |
|------|-----------|---------------|
| **Hurst exponent (H)** | Measures long-range dependence. H>0.5 = trending, H<0.5 = mean-reverting, H≈0.5 = random walk. | `fractal/hurst.py` |
| **MFDFA** | Multifractal Detrended Fluctuation Analysis. Measures how scaling behavior varies across fluctuation sizes. | `fractal/mfdfa.py` |
| **Spectrum width** | Width of the MFDFA singularity spectrum. Wider = more multifractal = more complex dynamics. | `MFDFAResult.spectrum_width` |
| **Regime** | Market state classification: TRENDING, MEAN_REVERTING, or RANDOM_WALK. Based on Hurst + MFDFA. | `fractal/regime.py` |

## SAX Domain

| Term | Definition | Where in code |
|------|-----------|---------------|
| **SAX** | Symbolic Aggregate approXimation. Converts continuous time series to a string of discrete symbols (e.g., "aabbccdd"). | `sax/sax.py` |
| **PAA** | Piecewise Aggregate Approximation. Mean-pools a series into fixed-width segments. Pre-processing step for SAX. | `sax/paa.py` |
| **Breakpoints** | Quantile boundaries from the standard normal distribution that map PAA values to SAX symbols. | `sax/sax.py:breakpoints()` |
| **Alphabet size** | Number of distinct symbols (e.g., 5 → {a,b,c,d,e}, 7 → {a,b,c,d,e,f,g}). Larger = finer granularity. | `SAXConfig.alphabet_size` |
| **SAX word** | Sliding window substring of SAX symbols (e.g., "abbc"). Length set by `word_length`. | `sax/bow.py:extract_words` |
| **Bag-of-Words (BoW)** | Frequency histogram of SAX words in a sequence. Analogous to text BoW. | `sax/bow.py:build_bow` |
| **TF-IDF** | Term Frequency × Inverse Document Frequency. Weights SAX words by how distinctive they are across assets. | `sax/bow.py:build_corpus_tfidf` |

## Tokenizer Domain

| Term | Definition | Where in code |
|------|-----------|---------------|
| **SAXVocabulary** | Maps SAX words ↔ integer token IDs. Includes PAD (0) and UNK (1) special tokens. | `tokenizer/vocabulary.py` |
| **PAD** | Padding token (ID=0). Used to left-pad short sequences to `context_length`. | `tokenizer/vocabulary.py:PAD_ID` |
| **UNK** | Unknown token (ID=1). Used for SAX words not in the vocabulary. | `tokenizer/vocabulary.py:UNK_ID` |
| **Context length** | Number of preceding tokens used to predict the next one. Default: 16 (Phase 3 optimal). | `SequenceModelConfig.context_length` |
| **MultiLevelTokenSequence** | Token sequences for all DWT levels of one asset. Keyed by level number. | `core/types.py` |

## Model Domain

| Term | Definition | Where in code |
|------|-----------|---------------|
| **WaveletGPT** | Causal transformer for next-SAX-word prediction. ~600K params (Phase 4 Optuna: D=128, 6 layers). Supports multi-horizon prediction. | `models/wavelet_gpt.py` |
| **Weight tying** | Sharing weight matrix between token embedding and output classification head. Reduces params and regularizes. | `WaveletGPTNet.__init__` |
| **Causal mask** | Upper-triangular boolean mask that prevents attention from seeing future tokens. Makes the model autoregressive. | `WaveletGPTNet.forward` |
| **WaveletLSTM** | Multi-branch LSTM with one branch per DWT level. Each branch processes that level's coefficients independently. | `models/wavelet_lstm.py` |
| **Ensemble** | Weighted combination of WaveletLSTM + XGBoost predictions. Optional stacking via meta-learner. | `models/ensemble.py` |
| **BaseModel** | Abstract base class: `fit()`, `predict()`, `save()`, `load()`, `name`. All models implement this. | `models/base.py` |

## Evaluation Domain

| Term | Definition | Where in code |
|------|-----------|---------------|
| **Token accuracy** | Exact match rate: predicted token == actual next token. | `evaluation/token_eval.py` |
| **Top-3 accuracy** | Rate at which the correct token is in the top 3 predictions by probability. | `evaluation/token_eval.py` |
| **Directional accuracy** | Whether the predicted token implies the same price direction as the actual. Level-0 only for multi-level SAX. | `experiments/metrics.py` |
| **Walk-forward** | Train on window [0, T], test on [T, T+k], slide forward. Prevents look-ahead bias in financial backtests. | `evaluation/backtest.py` |
| **Sharpe ratio** | Risk-adjusted return: mean(returns) / std(returns) × √252. Higher = better risk/reward. | `evaluation/metrics.py` |

## Experiment Domain

| Term | Definition | Where in code |
|------|-----------|---------------|
| **ExperimentConfig** | Complete specification of an experiment: tickers, interval, SAX/model params, split dates. | `experiments/config.py` |
| **ExperimentResult** | Full output of an experiment: metrics with CIs, baselines, per-asset/sector/level breakdowns. | `experiments/result.py` |
| **ExperimentRunner** | Orchestrates the full pipeline: load data → split → decompose → SAX → tokenize → train → evaluate. | `experiments/runner.py` |
| **Walk-forward splitter** | Splits raw prices by date BEFORE any DWT/SAX transformation. Prevents normalization leakage. | `experiments/splitter.py` |
| **Bootstrap CI** | 95% confidence interval via 1000 bootstrap resamples of test predictions. Quantifies uncertainty. | `experiments/metrics.py` |
| **Persistence baseline** | Predict next token = last token in context window. Strong baseline for trending regimes. | `experiments/metrics.py` |
| **Most-frequent baseline** | Always predict the mode of training tokens. Lowest-effort baseline. | `experiments/metrics.py` |
| **Momentum baseline** | Predict same direction as majority direction in context window. Trading-oriented baseline. | `experiments/metrics.py` |
| **Level-0 directional accuracy** | Directional accuracy computed on approximation level only. Detail levels represent oscillation, not direction. | `experiments/metrics.py` |
| **UNK rate** | Fraction of test tokens that map to UNK (unseen in training vocabulary). High UNK = vocab too restrictive. | `experiments/result.py` |
| **Ablation** | Removing one component (e.g., a DWT level) and measuring accuracy change. Used in C2 to identify noise levels. | `scripts/run_C2.py` |
| **DEFAULT_UNIVERSE** | 20 US assets across 6 sectors. Replaces the original 19-asset universe (now `LEGACY_UNIVERSE`). `PHASE3_UNIVERSE` is an alias. | `core/universe.py` |
| **LEGACY_UNIVERSE** | Original 19-asset universe from Phase 1-2 (equities, crypto, forex, commodity ETFs). Accessible via `get_universe("legacy")`. | `core/universe.py` |
| **Sector** | Sector-based classification: TECH, FINANCE, ENERGY, HEALTHCARE, BROAD_ETF, COMMODITY_ETF. Used for sector embeddings in WaveletGPT. | `core/types.py:Sector` |
| **PHASE3_UNIVERSE** | Alias for DEFAULT_UNIVERSE. 20 US assets across 6 sectors: tech (5), finance (3), energy (3), healthcare (3), broad ETFs (2), commodity ETFs (4). | `core/universe.py` |

## Phase 4 Domain

| Term | Definition | Where in code |
|------|-----------|---------------|
| **Multi-horizon prediction** | Predicting multiple steps ahead (h=1,2,4,8) simultaneously. Each horizon gets a separate classification head. h=1-2 useful, h=4+ plateaus. | `models/wavelet_gpt.py` |
| **Prediction horizon (h)** | Number of steps ahead to predict. h=1 = next token, h=2 = two tokens ahead, etc. | `SequenceModelConfig.prediction_horizons` |
| **Expanding window** | Validation strategy where training set grows: [0,T1], [0,T2], [0,T3]... while test windows slide forward. More data per split than rolling. | `experiments/splitter.py:expanding_window_split` |
| **Rolling window** | Validation strategy with fixed-size training window that slides: [T0,T1], [T1,T2]... Tests non-stationarity. | `experiments/splitter.py:rolling_window_split` |
| **Optuna HPO** | Hyperparameter optimization via Optuna's TPE sampler. Used for SAX params (F4) and model architecture (F5). | `experiments/hpo.py` |
| **Price reconstruction** | Inverse SAX → PAA → approximate price deltas. Maps predicted tokens back to approximate price movements. | `sax/reconstruction.py` |
| **PriceReconstructionResult** | Dataclass containing reconstructed deltas, directions (+1/-1/0), confidence, and breakpoint info. | `sax/reconstruction.py` |
| **Per-sector fine-tuning** | Training separate models per sector. Phase 4 proved this hurts ALL 6 sectors — cross-sector is definitively optimal. | `scripts/run_F6.py` |

## Phase 5 Domain — Signals & Backtesting

| Term | Definition | Where in code |
|------|-----------|---------------|
| **TradingSignal** | Single signal: direction (+1 buy, -1 sell, 0 hold), confidence (0-1), raw probability, token ID, horizon, ticker. | `signals/types.py` |
| **SignalSeries** | Ordered list of TradingSignals for one ticker/horizon. Properties: directions, confidences, timestamps arrays. | `signals/types.py` |
| **SignalGenerator** | Converts WaveletGPT softmax probabilities to trading signals via quartile aggregation. Configurable confidence threshold and temperature calibration. | `signals/generator.py` |
| **Temperature scaling** | Post-hoc calibration: divide logits by T before softmax. T found by minimizing NLL on validation data. T<1 sharpens, T>1 softens distributions. | `signals/generator.py:calibrate` |
| **PositionSizer** | Maps confidence to position size. Methods: fixed (constant), linear (confidence × max), Kelly (edge-based), fractional Kelly (0.5× Kelly). | `signals/position.py` |
| **Kelly criterion** | Optimal bet size: f* = (p×b - q) / b, where p=win_rate, b=avg_win/avg_loss, q=1-p. Maximizes log-growth. | `signals/position.py:_kelly_size` |
| **TransactionCostModel** | Models three cost types: commission (flat rate), spread (bid-ask in bps), slippage (market impact in bps). Direction changes double costs. | `signals/costs.py` |
| **SignalBacktest** | Iterates signals with actual returns, applies position sizing and costs, computes equity curve and 16 risk metrics. | `signals/backtest.py` |
| **TradeRecord** | Per-trade record: entry/exit timestamps, direction, position size, gross/net return, cost breakdown (commission, spread, slippage), confidence. | `signals/types.py` |
| **Sortino ratio** | Like Sharpe but only penalizes downside deviation. Better for asymmetric return distributions. | `evaluation/metrics.py` |
| **Calmar ratio** | Annualized return / max drawdown. Measures return per unit of worst-case loss. | `evaluation/metrics.py` |
| **VaR / CVaR** | Value at Risk (5th percentile loss) / Conditional VaR (mean of losses beyond VaR). Tail risk measures. | `evaluation/metrics.py` |
| **Expectancy** | Expected return per trade: win_rate × avg_win - loss_rate × avg_loss. Positive = profitable system. | `evaluation/metrics.py` |

## Phase 6 Domain — Performance

| Term | Definition | Where in code |
|------|-----------|---------------|
| **HAS_RUST** | Boolean flag indicating whether the Rust/PyO3 extension is compiled and available. Python fallbacks used when False. | `_rust.py` |
| **maturin** | Build system for mixed Rust+Python packages. Replaced hatchling in Phase 6. `maturin develop --release` builds and installs. | `pyproject.toml` |
| **AMP** | Automatic Mixed Precision. Uses float16 for forward pass, float32 for gradient accumulation. Speeds up training on CUDA GPUs. | `models/wavelet_gpt.py` |
| **GradScaler** | Scales gradients to prevent underflow in float16. Paired with torch.autocast. No-op when AMP disabled. | `models/wavelet_gpt.py` |
| **MMapSequenceDataset** | Memory-mapped dataset backed by .npy files. Zero-copy loading via numpy mmap_mode='r'. | `data/mmap_dataset.py` |
| **BatchPredictor** | Streaming batch inference wrapper for WaveletGPT. Splits large arrays into chunks, uses torch.inference_mode(). | `models/batch_inference.py` |
| **torch.inference_mode** | Stricter version of torch.no_grad(). Disables autograd tracking AND version counting for slightly better perf. | `models/batch_inference.py` |
