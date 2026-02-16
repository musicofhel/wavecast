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

### 2-Agent Research
For exploration and prototyping:
```
data-agent   → data fetching/analysis
model-agent  → model experiments
```

### Full Team (5 agents)
For major phase implementations:
```
data-agent → wavelet-agent → sax-agent → model-agent → pipeline-agent
```
With dependency chains managed via beads (`br dep add`).

## Task Tracking

Use beads (`br`) for agent task coordination:
```bash
br init                          # create workspace (once)
br add -t "Task title"           # add task
br dep add CHILD PARENT          # set dependency
br start ID                      # claim task
br close ID                      # mark done
br ls                            # list all
br ls --open                     # list open tasks
```

## Conventions

- Each agent should own non-overlapping file sets to avoid merge conflicts
- Tests for a module are written by the agent that owns the module
- Pipeline-agent handles integration between modules
- All agents must run `ruff check` on their files before marking tasks done
- Commit messages follow: `Add/Fix/Update <what> — <brief why>`
