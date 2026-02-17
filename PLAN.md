# WaveCast Phase 6: Performance Optimization

## Overview

Four workstreams: (A) Rust/PyO3 acceleration of SAX/BoW/tokenizer hot paths, (B) PyTorch AMP mixed-precision training, (C) memory-mapped dataset loading, (D) batched inference for signal generation. All changes are additive — existing Python implementations become fallbacks when Rust extension isn't available.

## Existing Stack

- **Build**: hatchling via `pyproject.toml`
- **Rust**: 1.93 installed, maturin NOT installed (needs `pip install maturin`)
- **GPU**: PyTorch 2.10 + CUDA 13.1, RTX 2060 SUPER — no AMP currently
- **Data**: NumPy arrays via `SequenceDataset.to_arrays()`, no memory mapping
- **Inference**: Single-batch `predict()`/`predict_proba()` on full array, no streaming

## Agent Parallelization (3 agents)

### Agent A: Rust/PyO3 Acceleration (the big one)

**Creates:**
- `rust/Cargo.toml` — crate `wavecast_rs`, cdylib, deps: pyo3 0.23, numpy (rust-numpy)
- `rust/src/lib.rs` — PyO3 module init
- `rust/src/sax_words.rs` — `extract_words_rs(symbols: &str, word_length: usize, stride: usize) -> Vec<String>` sliding window
- `rust/src/bow.rs` — `build_bow_rs(words: Vec<String>) -> HashMap<String, usize>` + `build_corpus_tfidf_rs(bow_list) -> (ndarray, Vec<String>)` using rust-numpy zero-copy return
- `rust/src/vocab.rs` — `encode_batch_rs(words: Vec<String>, word_to_id: HashMap<String, i64>, unk_id: i64) -> Vec<i64>` batch lookup
- `rust/src/dataset.rs` — `build_sliding_windows_rs(tokens: Vec<i64>, context_length: usize, max_horizon: usize) -> (contexts: ndarray, targets: ndarray)` returns 2D numpy arrays directly
- `src/wavecast/_rust.py` — thin Python wrapper with try/except fallback:
  ```python
  try:
      from wavecast_rs import extract_words_rs, build_bow_rs, ...
      HAS_RUST = True
  except ImportError:
      HAS_RUST = False
  ```

**Modifies:**
- `pyproject.toml` — switch build-backend to maturin, add `[tool.maturin]` config with `python-source = "src"`, keep all existing Python deps
- `src/wavecast/sax/bow.py` — `extract_words()` and `build_bow()` delegate to Rust when `HAS_RUST`
- `src/wavecast/tokenizer/vocabulary.py` — `encode_sequence()` delegates to Rust batch encoder
- `src/wavecast/tokenizer/dataset.py` — `build_sequence_dataset()` inner sliding window loop delegates to Rust
- `tests/unit/test_rust_acceleration.py` — test Rust vs Python produce identical results for all 4 functions, plus benchmarks

**Key design decisions:**
- maturin replaces hatchling as build backend (maturin supports pure Python + Rust mixed packages natively)
- `[tool.maturin] python-source = "src"` keeps existing src-layout
- `module-name = "wavecast._wavecast_rs"` puts the compiled .so inside `wavecast/` package
- All Rust functions return the SAME types as Python originals (lists, dicts, numpy arrays)
- GIL released during computation via `py.allow_threads(|| ...)`

### Agent B: Mixed Precision Training + Memory-Mapped Dataset

**Creates:**
- `src/wavecast/data/mmap_dataset.py` — `MMapSequenceDataset(torch.utils.data.Dataset)`:
  - `save(path, contexts, targets, levels, asset_classes)` — writes 4 `.npy` files with `np.save`
  - `__init__(path)` — opens with `np.load(mmap_mode='r')`
  - `__getitem__` — returns tensors from memory-mapped arrays (zero-copy)
  - `__len__` — returns array length
- `tests/unit/test_mmap_dataset.py` — save/load round-trip, __getitem__ correctness, multi-horizon support

**Modifies:**
- `src/wavecast/models/wavelet_gpt.py`:
  - `fit()` gains `use_amp: bool = False` parameter
  - Wraps forward pass in `torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)`
  - Adds `torch.amp.GradScaler("cuda", enabled=use_amp)` for gradient scaling
  - `predict()`/`predict_proba()` also wrap in `torch.autocast` when `use_amp=True`
  - `fit()` gains `dataset_path: Path | None = None` to load MMapSequenceDataset instead of in-memory TensorDataset
- `src/wavecast/core/config.py` — add `use_amp: bool = False` and `mmap_dataset_dir: str | None = None` to `SequenceModelConfig`
- `tests/unit/test_wavelet_gpt.py` — add test for AMP training (skip if no CUDA), test for mmap dataset loading

### Agent C: Batch Inference + CLI + Integration Tests

**Creates:**
- `src/wavecast/models/batch_inference.py` — `BatchPredictor` class:
  - `__init__(model: WaveletGPT, batch_size: int = 256)`
  - `predict_stream(X: NDArray, horizon: int) -> Iterator[NDArray]` — yields batch-sized prediction chunks
  - `predict_proba_stream(X: NDArray, horizon: int) -> Iterator[NDArray]` — yields batch-sized probability chunks
  - `predict_all(X: NDArray, horizon: int) -> NDArray` — convenience that concatenates stream
  - Uses `torch.inference_mode()` instead of `torch.no_grad()` for slightly better perf
  - Optional `torch.autocast` support for fp16 inference
- `tests/unit/test_batch_inference.py` — batch vs full-array equivalence, streaming correctness, edge cases (last batch < batch_size)
- `tests/integration/test_performance.py` — integration test that benchmarks Rust vs Python paths, AMP vs fp32, batch vs single inference. Uses `pytest-benchmark` style timing or simple `time.perf_counter` assertions.

**Modifies:**
- `src/wavecast/signals/generator.py` — `generate()` uses `BatchPredictor` internally when available
- `src/wavecast/cli/commands/signal.py` — add `--batch-size` and `--use-amp` flags
- `src/wavecast/cli/commands/backtest.py` or experiment commands — add `--use-amp` flag
- `ROADMAP.md` — mark Phase 6 items as complete, update Phase 5 to complete

## Build System Migration (hatchling → maturin)

This is the trickiest part. Maturin can build mixed Rust+Python packages:

```toml
[build-system]
requires = ["maturin>=1.8,<2.0"]
build-backend = "maturin"

[tool.maturin]
python-source = "src"
module-name = "wavecast._wavecast_rs"
features = ["pyo3/extension-module"]
```

The Cargo.toml goes in `rust/` (or project root — we'll use `rust/` to keep separation):

```toml
# pyproject.toml addition:
[tool.maturin]
manifest-path = "rust/Cargo.toml"
```

**Fallback**: If Rust compilation fails (e.g., no Rust toolchain), the package still installs as pure Python. The `_rust.py` wrapper catches `ImportError` gracefully.

## Verification Criteria

1. `pytest tests/ -q` — all existing 349 tests still pass
2. New tests: ~20-25 tests across 4 new test files
3. `ruff check src/ tests/` — 0 errors
4. Rust extension builds: `maturin develop --release` succeeds
5. Benchmark: Rust `extract_words` + `build_bow` ≥ 5x faster than Python on 10K-char SAX string
6. AMP training: no accuracy regression (within 1%) vs fp32 on synthetic data
7. Batch inference: identical outputs to single-pass inference
8. Memory-mapped dataset: correct round-trip save/load

## Estimated New Files: 10-12
## Estimated Modified Files: 8-10
## Estimated New Tests: 20-25
