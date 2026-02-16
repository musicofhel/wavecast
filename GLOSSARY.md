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
| **Context length** | Number of preceding tokens used to predict the next one. Default: 16 or 32. | `SequenceModelConfig.context_length` |
| **MultiLevelTokenSequence** | Token sequences for all DWT levels of one asset. Keyed by level number. | `core/types.py` |

## Model Domain

| Term | Definition | Where in code |
|------|-----------|---------------|
| **WaveletGPT** | Causal transformer for next-SAX-word prediction. ~161K params. | `models/wavelet_gpt.py` |
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
| **Directional accuracy** | Whether the predicted token implies the same price direction as the actual. The trading-relevant metric. | `evaluation/token_eval.py` |
| **Walk-forward** | Train on window [0, T], test on [T, T+k], slide forward. Prevents look-ahead bias in financial backtests. | `evaluation/backtest.py` |
| **Sharpe ratio** | Risk-adjusted return: mean(returns) / std(returns) × √252. Higher = better risk/reward. | `evaluation/metrics.py` |
