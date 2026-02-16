"""Performance integration tests for Phase 6 optimizations."""
from __future__ import annotations

import time

import numpy as np

from wavecast.models.batch_inference import BatchPredictor
from wavecast.models.wavelet_gpt import WaveletGPT


def _make_large_data(n=1000, ctx_len=16, vocab=100, seed=42):
    rng = np.random.default_rng(seed)
    contexts = rng.integers(0, vocab, size=(n, ctx_len))
    levels = rng.integers(0, 5, size=(n, 1))
    acs = rng.integers(0, 4, size=(n, 1))
    X = np.column_stack([contexts, levels, acs]).astype(np.float64)
    y = rng.integers(0, vocab, size=n).astype(np.float64)
    return X, y


class TestBatchInferencePerformance:
    def test_batch_produces_identical_results(self):
        """Batch inference matches single-pass on larger dataset."""
        X, y = _make_large_data(n=500)
        model = WaveletGPT(
            vocab_size=100,
            context_length=16,
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            epochs=2,
            batch_size=64,
            n_levels=5,
            n_asset_classes=4,
        )
        model.fit(X[:400], y[:400])

        full_preds = model.predict(X[400:])
        bp = BatchPredictor(model, batch_size=32)
        batch_preds = bp.predict_all(X[400:])
        np.testing.assert_array_equal(full_preds, batch_preds)

    def test_batch_proba_matches(self):
        """Batch probability inference matches single-pass."""
        X, y = _make_large_data(n=500)
        model = WaveletGPT(
            vocab_size=100,
            context_length=16,
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            epochs=2,
            batch_size=64,
            n_levels=5,
            n_asset_classes=4,
        )
        model.fit(X[:400], y[:400])

        full_proba = model.predict_proba(X[400:])
        bp = BatchPredictor(model, batch_size=32)
        batch_proba = bp.predict_proba_all(X[400:])
        np.testing.assert_allclose(full_proba, batch_proba, atol=1e-5)


class TestRustAcceleration:
    def test_rust_available(self):
        """Check if Rust extension is available (informational)."""
        from wavecast._rust import HAS_RUST

        # This test passes regardless -- just logs availability
        if HAS_RUST:
            print("Rust acceleration: AVAILABLE")
        else:
            print("Rust acceleration: NOT AVAILABLE (using Python fallback)")

    def test_extract_words_benchmark(self):
        """Benchmark extract_words (Rust vs Python if available)."""
        from wavecast.sax.bow import extract_words

        # Generate a large SAX string
        rng = np.random.default_rng(42)
        symbols = "".join(
            chr(ord("a") + i) for i in rng.integers(0, 7, size=10000)
        )

        start = time.perf_counter()
        for _ in range(100):
            extract_words(symbols, 4, 1)
        elapsed = time.perf_counter() - start

        print(f"extract_words: 100 iterations on 10K chars = {elapsed:.3f}s")
        assert elapsed < 30.0  # generous timeout
