# WaveCast Agent Roles

Agent definitions for parallel development using Claude Code teams + beads task tracking.

## Core Agents

### data-agent
**Domain**: Data ingestion, caching, preprocessing
**Owns**: `src/wavecast/data/`, `src/wavecast/core/universe.py`
**Skills**: Massive.com API, Parquet caching, time series preprocessing, multi-asset universe management
**When to use**: Fetching new assets, adding data sources, cache optimization, preprocessing pipeline changes

### wavelet-agent
**Domain**: Wavelet decomposition, reconstruction, CWT
**Owns**: `src/wavecast/wavelets/`, `src/wavecast/fractal/`
**Skills**: DWT/CWT via PyWavelets, Hurst exponent, MFDFA, regime detection, self-similarity
**When to use**: Wavelet parameter tuning, new decomposition methods, fractal feature engineering

### sax-agent
**Domain**: SAX transformation, tokenization, vocabulary
**Owns**: `src/wavecast/sax/`, `src/wavecast/tokenizer/`, `src/wavecast/features/sax_features.py`
**Skills**: PAA, SAX breakpoints, Bag-of-Words, TF-IDF, vocabulary management, sequence dataset building
**When to use**: SAX parameter optimization, new word extraction strategies, vocabulary experiments

### model-agent
**Domain**: ML models, training, evaluation
**Owns**: `src/wavecast/models/`, `src/wavecast/evaluation/`
**Skills**: PyTorch (WaveletGPT, WaveletLSTM), XGBoost, ensemble methods, walk-forward backtesting, token eval
**When to use**: Model architecture changes, hyperparameter tuning, new evaluation metrics, GPU optimization

### experiment-agent
**Domain**: Experiment framework, research experiments
**Owns**: `src/wavecast/experiments/`, `scripts/run_C*.py`
**Skills**: ExperimentConfig/Result, ExperimentRunner, walk-forward splitting, bootstrap CIs, baseline computation, results storage
**When to use**: Running new experiments, adding metrics, modifying the experiment pipeline

### pipeline-agent
**Domain**: End-to-end orchestration, CLI, library building
**Owns**: `src/wavecast/pipeline/`, `src/wavecast/cli/`
**Skills**: Stage orchestration, token pipeline runner, library builder, Typer CLI commands
**When to use**: New CLI commands, pipeline integration, multi-asset batch processing

### test-agent
**Domain**: Testing, coverage, fixtures
**Owns**: `tests/`
**Skills**: pytest, deterministic fixtures (seed=42), integration testing, coverage analysis
**When to use**: Writing tests for new features, improving coverage, regression testing

## Team Configurations

### 3-Agent Sprint (Default)
For feature implementation with clear stream separation:
```
sax-agent    → SAX/tokenizer work
model-agent  → model/evaluation work
pipeline-agent → integration/CLI work
```

### 3-Agent Experiment Wave (Phase 3 pattern)
For running research experiments in parallel:
```
Wave 1 (foundation):  data-agent | framework-agent | infra-agent
Wave 2 (experiments): agent-1 (C1→C2→C3) | agent-2 (C4→C5→C6) | agent-3 (C7)
Wave 3 (finalize):    agent-1 (D1→D4) | agent-2 (E1→E3)
```
Dependency chains managed via beads (`br dep add`). Agents poll for predecessor completion.

### 2-Agent Research
For exploration and prototyping:
```
data-agent   → data fetching/analysis
model-agent  → model experiments
```

### Full Team (6 agents)
For major phase implementations:
```
data-agent → wavelet-agent → sax-agent → model-agent → experiment-agent → pipeline-agent
```
With dependency chains managed via beads (`br dep add`).

## Task Tracking

Use beads (`br`) for agent task coordination:
```bash
br init                          # create workspace (once)
br create "Task title"           # create task
br dep add CHILD PARENT          # set dependency
br update ID -s in_progress      # claim task
br close ID                      # mark done
br list                          # list all
br ready                         # list unblocked tasks
br blocked                       # list blocked tasks
```

## Conventions

- Each agent should own non-overlapping file sets to avoid merge conflicts
- Tests for a module are written by the agent that owns the module
- Pipeline-agent handles integration between modules
- All agents must run `ruff check` on their files before marking tasks done
- Commit messages follow: `Add/Fix/Update <what> — <brief why>`
