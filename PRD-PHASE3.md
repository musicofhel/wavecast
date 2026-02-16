# Phase 3 PRD: Real-Data Validation & Research Experiments

## Objective

Answer the 7 research questions using real market data, determine optimal WaveletGPT configuration per sector, and produce a walk-forward evaluation proving (or disproving) that SAX token prediction works on actual financial time series.

## Success Criteria

- Token accuracy significantly above naive baselines (most-frequent-token AND persistence)
- Directional accuracy > 55% on at least 2 sectors (above both coin flip AND momentum baseline)
- Multi-level SAX outperforms single-level by measurable margin with overlapping confidence intervals excluded
- Each research question answered with data, confidence intervals, and baseline comparisons
- Final model config chosen and justified by experiment results

---

## Data Plan: Massive.com API

**Plan features**: All US Stock Tickers, Unlimited API Calls, 5 Years Historical, 100% Market Coverage, 15-min Delayed, Minute Aggregates, Technical Indicators, Corporate Actions, WebSockets, Snapshots.

**Implications**:
- **No rate limit risk** — unlimited calls, fetch everything upfront
- **US Stocks + US-listed ETFs only** — crypto (`X:` prefix) and forex (`C:` prefix) tickers likely unavailable. Commodity ETFs (GLD, SLV, USO, UNG) ARE US-listed and should work.
- **5-year lookback** — from Feb 2026, earliest data is ~Feb 2021 (NOT Jan 2020). Training period shrinks to ~3 years.
- **Minute aggregates available** — can use hourly bars to multiply sample count 6.5× vs daily
- **Technical indicators built-in** — potential Pipeline 1 feature supplement

### Revised Universe: US Equities + Commodity ETFs (20 assets, 5 sectors)

```python
# Tech (5)
AAPL, MSFT, GOOGL, AMZN, NVDA

# Finance (3)
JPM, GS, BAC

# Energy (3)
XOM, CVX, COP

# Healthcare (3)
JNJ, UNH, PFE

# Broad ETFs (2)
SPY, QQQ

# Commodity ETFs (4) — US-listed, should work on US Stocks plan
GLD, SLV, USO, UNG
```

Cross-asset experiment (C3) becomes cross-SECTOR instead of cross-asset-class. `AssetClass` enum gets replaced/extended with `Sector` or we repurpose the existing enum values.

### Interval Strategy: Daily vs Hourly

**Daily bars** (primary):
- ~750 trading days for training (Feb 2021 – Dec 2023)
- With n_segments=64: only 45 SAX words per level → too few samples
- With n_segments=128: ~109 words → ~93 samples per level per asset

**Hourly bars** (for sample-hungry experiments):
- ~4,875 hourly bars for training (750 days × 6.5 hrs)
- With n_segments=256: ~237 samples per level per asset × 5 levels × 20 assets = **23,700 samples**
- Hourly is the sweet spot: enough samples without drowning in 1-min noise

**Recommendation**: Use hourly bars as the default interval. Daily as a secondary comparison. Minute data reserved for Phase 7 (intraday).

---

## Stream A: Data Foundation

### A1 — Fetch Universe Data
Fetch hourly + daily data for all 20 assets via Massive.com API.
- Date range: 2021-02-01 to 2025-12-31 (~5 years, matching plan limit)
- Intervals: `1h` (primary), `1d` (secondary)
- Output: Parquet cache at `~/.wavecast/cache/`
- Handle: missing hours (forward-fill within trading hours only), stock splits (adjusted close via Corporate Actions), half-days (keep as-is)
- Validation: assert each asset has >= 3000 hourly bars, print summary table
- Verify: test `GLD`, `SLV`, `USO`, `UNG` tickers actually return data on this plan. If any fail, replace with alternatives (e.g., `IAU` for gold, `PDBC` for broad commodity).
- **File**: `src/wavecast/data/sources.py` (existing `fetch_massive`)

### A2 — Precompute & Cache (Exploration Only)
DWT decompose all 20 assets × full period for exploration and visualization.
- **NOT used by experiments** — experiments decompose per-split internally (see B3)
- Cache to `~/.wavecast/cache/decompositions/` for ad-hoc analysis
- **New file**: `src/wavecast/data/decomposition_cache.py`

### A3 — Compute Rolling Hurst + Regime Labels
For each asset, compute rolling Hurst exponent (window=252 trading days ≈ 1638 hourly bars) and classify into regimes.
- Output: per-asset array of `RegimeType` labels aligned to timestamps
- Cache alongside raw data
- Needed for: experiment C6 (regime dependence)
- **Note**: Hurst is computed on raw prices (not DWT coefficients), so no leakage concern — train/test regime labels can be computed on the full series.
- **File**: `src/wavecast/fractal/hurst.py` (existing `rolling_hurst`), new caching wrapper

---

## Stream B: Experiment Framework

### B1 — ExperimentConfig
```python
@dataclass
class ExperimentConfig:
    name: str
    # Data
    tickers: list[str]
    interval: str = "1h"          # "1h" or "1d"
    train_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    # SAX
    alphabet_size: int = 7
    n_segments: int = 256          # CHANGED from 64 — need more for sample count
    word_length: int = 4
    word_stride: int = 1
    # Tokenizer
    min_word_freq: int = 2
    max_vocab_size: int = 300
    context_length: int = 16
    # Model
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 0.0005
    patience: int = 15
    # Experiment control
    dwt_levels: list[int] | None = None
    sectors: list[str] | None = None   # renamed from asset_classes
    cross_sector_training: bool = True
```
- `n_segments` default bumped to 256 (was 64). At hourly resolution with ~4875 training bars, 256 segments gives ~237 samples per level per asset.
- **New file**: `src/wavecast/experiments/config.py`

### B2 — ExperimentResult
```python
@dataclass
class ExperimentResult:
    config: ExperimentConfig
    # Aggregate metrics (with confidence intervals)
    token_accuracy: float
    token_accuracy_ci: tuple[float, float]    # bootstrap 95% CI
    top3_accuracy: float
    directional_accuracy: float
    directional_accuracy_ci: tuple[float, float]
    # Baselines
    baseline_most_frequent: float              # always predict most common token
    baseline_persistence: float                # predict next = current
    baseline_momentum: float                   # predict same direction as last N
    # Breakdowns
    per_asset_accuracy: dict[str, float]
    per_sector_accuracy: dict[str, float]      # renamed from per_class
    per_level_accuracy: dict[int, float]
    # Meta
    vocab_size: int
    unk_rate: float
    n_train_samples: int
    n_test_samples: int
    training_time_seconds: float
    timestamp: str
```
- Added `_ci` fields for confidence intervals (bootstrap, 1000 resamples)
- Added three baseline metrics computed on the same test set
- **New file**: `src/wavecast/experiments/result.py`

### B3 — Walk-Forward Splitter
**CRITICAL**: Split at RAW PRICE level BEFORE any transformation.

```python
def walk_forward_split(
    price_series: dict[str, TimeSeries],
    train_end: str,
    test_start: str,
) -> tuple[dict[str, TimeSeries], dict[str, TimeSeries]]:
```

The ExperimentRunner (B4) applies the full pipeline to each split independently:
1. Split raw prices by date → train_prices, test_prices
2. DWT decompose each split independently
3. Z-normalize each split's coefficients independently (train stats for train, test stats for test)
4. SAX transform each split separately
5. Build vocabulary from TRAIN SAX words only
6. Encode both splits using train-built vocabulary (test gets UNK for unseen words)
7. Build sequence datasets from each
8. Train on train, evaluate on test

**PAA resolution mismatch**: With n_segments=256, training series (~4875 hourly bars) has ~19 bars per segment. Test series (~1625 hourly bars for 2024) has ~6.3 bars per segment. Different temporal resolution, but SAX symbols are z-normalized so statistical meaning is preserved. The model learns symbolic transition probabilities, not temporal dynamics. This is acceptable but documented as a known limitation.

- **New file**: `src/wavecast/experiments/splitter.py`

### B4 — ExperimentRunner
```python
class ExperimentRunner:
    def __init__(self, cache_dir: Path): ...
    def run(self, config: ExperimentConfig) -> ExperimentResult: ...
    def run_sweep(self, configs: list[ExperimentConfig]) -> list[ExperimentResult]: ...
```
- Loads cached RAW PRICE DATA (from A1), NOT decompositions
- Splits → decomposes → SAX transforms → tokenizes → trains → evaluates per-split
- Computes baselines on the same test set:
  - Most-frequent-token: predict the mode of training tokens for every test sample
  - Persistence: predict next token = last token in context window
  - Momentum: predict same direction as the majority direction in the context window
- Computes bootstrap 95% CI on accuracy metrics (1000 resamples)
- **New file**: `src/wavecast/experiments/runner.py`

### B5 — Directional Accuracy Definition
**This needs explicit definition for multi-level SAX.**

For the **approximation level** (level 0): SAX symbols directly represent trend. Higher symbol = higher value. Direction is: predicted symbol > current symbol → UP, < → DOWN, = → HOLD.

For **detail levels** (1-5): symbols represent oscillation magnitude, NOT price direction. Two options:
- **Option A**: Only compute directional accuracy on level 0. Use token accuracy for detail levels.
- **Option B**: Train a small meta-classifier that takes multi-level token predictions and outputs a directional signal. This is essentially what C7 (ensemble) does.

**Decision**: Use Option A for experiments C1-C6. Directional accuracy is level-0 only. C7 tests the full multi-level → direction pipeline.

- Implement `level0_directional_accuracy()` as a dedicated metric
- Document in ExperimentResult which levels contribute to which metrics
- **New file**: `src/wavecast/experiments/metrics.py`

### B6 — Results Storage & Comparison
```python
def save_results(results: list[ExperimentResult], path: Path) -> None: ...
def load_results(path: Path) -> list[ExperimentResult]: ...
def compare_results(results: list[ExperimentResult], metric: str = "directional_accuracy") -> DataFrame: ...
```
- JSON serialization handles None, int dict keys, tuple CIs
- Output dir: `~/.wavecast/experiments/`
- **New file**: `src/wavecast/experiments/storage.py`

### B7 — Experiment CLI
```bash
wavecast experiment run CONFIG_FILE.yaml
wavecast experiment sweep SWEEP_FILE.yaml
wavecast experiment compare DIR
wavecast experiment show RESULT_FILE.json
```
- **New file**: `src/wavecast/cli/commands/experiment.py`

---

## Stream C: Research Experiments

Each experiment answers one research question. All depend on Streams A + B.

All experiments report: accuracy ± 95% CI, plus all three baselines. Results are only meaningful if they beat ALL baselines.

### C1 — SAX Granularity Sweep (Q1: alphabet_size)
**Hypothesis**: Alphabet 7 is near-optimal for equity. Commodity ETFs may need different granularity.

**Method**: Hold everything else constant (n_segments=256, word_length=4, context=16). Sweep `alphabet_size ∈ {3, 5, 7, 9, 11}` across all 20 assets, hourly bars.

**Note**: alphabet=3 gives only 3^4=81 possible words (tiny vocab). alphabet=11 gives 11^4=14,641 possible words (most never appear). Both extremes are informative.

**Configs**: 5 experiments.

**Output**: Table of (alphabet_size × sector → directional_accuracy ± CI). Compare against persistence baseline at each setting.

**Decision**: Pick alphabet_size with highest lift over persistence baseline. If it varies by sector by > 5% absolute, record per-sector optimal.

### C2 — Level Contribution Analysis (Q2: which DWT levels matter)
**Hypothesis**: Levels 2-4 carry the most predictable SAX patterns. Level 1 is noise, level 5 is too smooth for hourly data.

**Method**: Use optimal alphabet from C1. If C1 shows different optima per sector (>5% difference), run C2 with the globally best AND the most divergent sector-optimal alphabet (max 2 alphabet settings).

- Phase 2a: 5 single-level models (levels 1-5)
- Phase 2b: 1 all-levels-combined model (baseline)
- Phase 2c: 5 ablation models (drop one level each)

**Configs**: 11 experiments (or 22 if two alphabet settings needed).

**Output**: Per-level token accuracy. Ablation impact (accuracy drop when level removed). Level 0 directional accuracy vs detail-level token accuracy.

**Decision**: Exclude levels that contribute < 1% accuracy improvement in ablation. These are noise.

### C3 — Cross-Sector Transfer (Q3: does multi-sector training help)
**Hypothesis**: Cross-sector training helps within similar sectors (tech↔broad ETFs) but may hurt across dissimilar sectors (energy→healthcare).

**Method**: Use optimal alphabet + levels from C1-C2.
- Baseline: train on all 20 assets, test on all
- Per-sector isolation: train only on tech, test on tech. Repeat for each sector.
- Leave-one-sector-out: train on 4 sectors, test on held-out sector
- Single-asset: train per-ticker, test per-ticker

**Configs**: 1 (all) + 5 (per-sector) + 5 (leave-one-out) + 20 (single-asset) = 31 experiments.

**Warning on single-asset experiments**: With hourly data and n_segments=256, each asset yields ~237 samples per level × optimal levels ≈ ~700-1200 training samples. The 161K-param model will overfit. These results should be treated as directional indicators with high variance, not precise measurements. Consider reporting single-asset results as a group distribution (median ± IQR across 20 assets) rather than per-ticker.

**Output**: Transfer matrix (train_sector × test_sector → accuracy). All-asset vs per-sector vs single-asset comparison.

**Decision**: Universal model vs per-sector models. If per-sector beats universal by > 3%, use per-sector.

### C4 — Vocabulary Saturation (Q4: how many words do we need)
**Hypothesis**: Diminishing returns after ~200 words. Rare words are noise.

**Method**: Fix optimal settings from C1-C3. Sweep:
- `max_vocab_size ∈ {50, 100, 200, 300, 500}`
- `min_word_freq ∈ {1, 2, 3, 5}`

**Configs**: 5 × 4 = 20 experiments.

**Output**: Accuracy vs vocab_size curve. UNK rate vs min_freq. Find the knee: where does adding more words stop helping? Where does min_freq start killing test coverage (UNK > 10%)?

**Decision**: Pick (vocab_size, min_freq) at the knee.

### C5 — Context Length Sweep (Q5: how much history matters)
**Hypothesis**: 32 is better than 16 for hourly data. 64 may help or may overfit.

**Method**: Fix optimal settings from C1-C4. Sweep `context_length ∈ {8, 16, 32, 64}`.

**Note**: Changing context_length changes the positional embedding size AND reduces the number of training samples (longer context = fewer sliding windows). With n_segments=256 and context=64: 256-4+1-64 = 189 samples per level (vs 237 at context=16). Report both accuracy and sample efficiency.

**Configs**: 4 experiments.

**Output**: Accuracy vs context_length. Training time vs context_length. Accuracy per training sample vs context_length.

**Decision**: Pick context_length maximizing accuracy. If 32 and 64 are within CI of each other, prefer 32 (more samples, faster training).

### C6 — Regime Dependence (Q6: trending vs mean-reverting)
**Hypothesis**: WaveletGPT is better at predicting during trending regimes.

**Method**: Use rolling Hurst labels from A3. Train one model with full optimal config. Evaluate separately on:
- Trending periods (H > 0.6)
- Mean-reverting periods (H < 0.4)
- Random walk periods (0.4 ≤ H ≤ 0.6)

**Configs**: 1 experiment, 3 evaluation slices.

**Warning**: If regimes are unevenly distributed (e.g., 80% random walk, 10% trending, 10% mean-reverting), the minority regime slices will have wide CIs. Report sample counts per regime.

**Output**: Per-regime accuracy ± CI with sample counts. Per-regime comparison against persistence baseline (persistence should be strong in trending, weak in mean-reverting).

**Decision**: If accuracy varies by > 10% between regimes, flag regime-conditional models as a Phase 4 priority.

### C7 — Ensemble Value (Q7: Pipeline 1 + Pipeline 2)
**Hypothesis**: Combining wavelet-shapelet features with SAX token predictions outperforms either alone.

**Prerequisite**: Pipeline 1 has never run on real Massive.com data. Sub-task: validate P1 end-to-end on at least 3 assets before attempting the full comparison. If P1 fails, skip ensemble and report P2-only results.

**Method**:
- P1 alone: WaveletLSTM + XGBoost → directional accuracy (same walk-forward split)
- P2 alone: WaveletGPT (optimal config, level 0) → directional accuracy
- P2 multi-level: all optimal levels → token probabilities → meta-classifier → directional accuracy
- Combined: P1 features + P2 token probabilities → XGBoost meta-learner → directional accuracy
- Error correlation: Pearson between P1 and P2 error vectors

**Configs**: 4 experiments (or 3 if P1 fails).

**Output**: Side-by-side accuracy. Error correlation. If corr < 0.3, ensemble has theoretical justification.

**Decision**: If combined > max(P1, P2) by > 2% absolute and outside CI overlap, build permanent ensemble combiner.

---

## Stream D: Optimal Config & Final Model

### D1 — Aggregate Results
- Load all C-stream results
- Build decision table: optimal (alphabet_size, levels, vocab_size, context_length, training_strategy) per sector
- Determine: universal model vs per-sector models
- Write to `~/.wavecast/experiments/phase3_findings.json`

### D2 — Train Final Model
- Using optimal config, train WaveletGPT on 2021-2024 data (includes former test period — standard practice for final model)
- Save model weights + vocabulary + config to `~/.wavecast/models/phase3_final/`

### D3 — Held-Out Evaluation
- Evaluate on 2025 data (NEVER seen during ANY experiment or config selection)
- This is the true out-of-sample test
- Report: all metrics + baselines + CIs, per-asset, per-sector, per-regime
- If 2025 accuracy is substantially worse than 2024 accuracy (> 5% drop), flag potential overfitting to validation period

### D4 — Generate Report
- `PHASE3_RESULTS.md` in repo root
- For each research question: hypothesis → experiment → result → conclusion
- Tables with CIs and baseline comparisons
- Honest assessment: what works, what doesn't, what's inconclusive
- Recommended next steps for Phase 4

---

## Stream E: Integration

### E1 — Update Universe
- Replace DEFAULT_UNIVERSE with the 20-asset US equities + commodity ETFs universe
- Replace/extend `AssetClass` enum with sector-based classification
- Update all references

### E2 — Update Defaults
- Set optimal SAX/tokenizer/model defaults in `core/config.py`
- Update CLI default arguments
- Set default interval to `1h` if hourly proves superior

### E3 — Update Docs
- CLAUDE.md: real-data benchmarks, hourly interval notes
- ROADMAP.md: Phase 3 findings, updated Phase 4 plan
- MEMORY.md: key findings for future sessions

### E4 — Tests
- Add experiment framework unit tests (config, result, splitter, runner)
- Add integration test: fetch 1 asset → split → DWT → SAX → train → evaluate (small epochs, real data)
- Add baseline computation tests

---

## Dependency Graph

```
A1 (fetch data)
 ├→ A2 (exploration decompositions — NOT experiment dependency)
 ├→ A3 (Hurst + regime labels)
 └→ B1-B7 (experiment framework)
      └→ C1 (alphabet sweep)
          └→ C2 (level contribution)
              └→ C3 (cross-sector transfer)
                  └→ C4 (vocab saturation)
                      └→ C5 (context length)
                          ├→ C6 (regime dependence) [also needs A3]
                          └→ C7 (ensemble)
                              └→ D1-D4 (final model + report)
                                  └→ E1-E4 (integration)
```

A2 is decoupled from the experiment chain. C6 and C7 run in parallel after C5.

## Beads Task Breakdown

```
Epic: Phase 3 — Real-Data Validation & Research Experiments

# Stream A: Data Foundation
A1  Fetch 20-asset universe (hourly + daily) from Massive.com         [blocks: everything]
A2  Precompute DWT decompositions for exploration (optional)
A3  Compute rolling Hurst + regime labels for all assets              [blocks: C6]

# Stream B: Experiment Framework
B1  Create ExperimentConfig dataclass
B2  Create ExperimentResult dataclass (with CIs + baselines)
B3  Build walk-forward splitter (raw price level, pre-transform)
B4  Build ExperimentRunner (split → decompose → SAX → train → eval)  [blocks: C1-C7]
B5  Define directional accuracy for multi-level SAX (level-0 only)
B6  Build results storage + comparison utilities
B7  Add experiment CLI commands
E4  Add experiment framework unit tests                               [dep: B4]

# Stream C: Research Experiments
C1  SAX granularity sweep (alphabet 3/5/7/9/11)                      [dep: A1, B4]
C2  DWT level contribution (single-level + ablation, 11-22 configs)  [dep: C1]
C3  Cross-sector transfer matrix (31 configs)                        [dep: C2]
C4  Vocabulary saturation sweep (vocab × min_freq, 20 configs)       [dep: C3]
C5  Context length sweep (8/16/32/64)                                [dep: C4]
C6  Regime dependence analysis (trending/reverting/random)           [dep: C5, A3]
C7  Ensemble comparison (P1/P2/multi-level/combined)                 [dep: C5]

# Stream D: Final Model
D1  Aggregate results → optimal config per sector                    [dep: C6, C7]
D2  Train final WaveletGPT on 2021-2024 with optimal config         [dep: D1]
D3  Held-out evaluation on 2025 data                                 [dep: D2]
D4  Generate PHASE3_RESULTS.md                                       [dep: D3]

# Stream E: Integration
E1  Update universe to US equities + commodity ETFs                  [dep: D1]
E2  Update default configs with optimal values                       [dep: D1]
E3  Update CLAUDE.md, ROADMAP.md, MEMORY.md                         [dep: D4]
```

## Parallelization Strategy (3 agents)

### Wave 1: Foundation (all 3 parallel)
- **data-agent**: A1 → A3
- **framework-agent**: B1 → B2 → B3 → B4 → B5
- **infra-agent**: B6 → B7 → E4

### Wave 2: Experiments (all 3 parallel)
- **agent-1**: C1 → C2 → C3 (heaviest: 31 configs in C3)
- **agent-2**: C4 (once C3 done) → C5 → C6
- **agent-3**: C7 (once C5 done, includes P1 validation sub-task)

### Wave 3: Finalize (2 agents)
- **agent-1**: D1 → D2 → D3 → D4
- **agent-2**: E1 → E2 → E3

## Sample Count Estimates (corrected)

With hourly bars, n_segments=256, word_length=4, stride=1:

| Scenario | Training bars | Words/level | Samples/level (ctx=16) | Levels | Assets | Total |
|---|---|---|---|---|---|---|
| All assets, all levels | ~4,875/asset | 253 | 237 | 5 | 20 | **23,700** |
| All assets, 3 levels | ~4,875/asset | 253 | 237 | 3 | 20 | **14,220** |
| Per-sector (5 assets) | ~4,875/asset | 253 | 237 | 3 | 5 | **3,555** |
| Single-asset | ~4,875 | 253 | 237 | 3 | 1 | **711** |
| Daily, all assets | ~750/asset | 253* | 237* | 5 | 20 | **23,700** |

*Daily with n_segments=256 on 750 bars: each segment is ~2.9 bars. Very few raw data points per segment — may be too aggressive. Daily experiments should use n_segments=64-128.

## Temporal Split Strategy

- **Training**: 2021-02-01 to 2023-12-31 (~3 years)
- **Validation/Test**: 2024-01-01 to 2024-12-31 (all C-stream experiments evaluate here)
- **Held-out**: 2025-01-01 to 2025-12-31 (touched ONCE in D3)
- **Final model training** (D2): 2021-02-01 to 2024-12-31 (absorbs former test period)

## Fixed Parameters (deferred to Phase 4)

- `n_segments = 256` — higher than Phase 2's 64 due to hourly resolution. Could be swept but interacts with alphabet_size.
- `word_length = 4` — could be 3-6. Each value changes vocab composition entirely.
- `word_stride = 1` — stride > 1 reduces samples, counterproductive with limited data.
- `wavelet = db4` — alternative wavelets are a separate research axis.
- `embed_dim = 64, num_heads = 4, num_layers = 3` — architecture held constant. Optuna sweep is Phase 4.
- `interval = 1h` — daily as secondary comparison only.

## Known Limitations

1. **PAA resolution mismatch**: Training SAX (4875 bars / 256 segments ≈ 19 bars/segment) vs test SAX (1625 bars / 256 segments ≈ 6.3 bars/segment). Symbols are z-normalized so statistical meaning is preserved, but temporal granularity differs 3×. The model learns symbolic patterns, not temporal patterns — acceptable.
2. **Single-asset experiments are underpowered**: ~711 training samples for 161K params. Results are directional indicators, not precise measurements.
3. **Experiment selection bias**: We choose optimal configs based on 2024 performance across 72+ experiments. The 2025 holdout is the check against this bias.
4. **Directional accuracy is level-0 only**: Detail levels don't map to price direction. The multi-level → direction pipeline is tested only in C7.
5. **Pipeline 1 untested on real data**: C7 ensemble depends on P1 working. May degrade to P2-only evaluation.

## Risks

1. **Real data is noisier**: Synthetic hit 48% token accuracy, 87.7% directional. Real data will be lower. Success bar is beating baselines, not matching synthetic.
2. **Insufficient hourly data quality**: Hourly bars may have gaps (overnight, holidays, early closes). Forward-fill within trading hours only — don't fill overnight gaps.
3. **Commodity ETFs may not work**: GLD/SLV/USO/UNG are US-listed but verify in A1. Have fallback tickers ready.
4. **Experiment cascade bottleneck**: C1→C5 is sequential. If any experiment is ambiguous (no clear winner), use reasonable default and note the ambiguity.
5. **2025 holdout is only 1 year**: Limited statistical power for final evaluation. Consider multi-year rolling holdout in Phase 4.
