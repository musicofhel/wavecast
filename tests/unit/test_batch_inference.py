"""Tests for BatchPredictor."""
from __future__ import annotations

import numpy as np
import pytest

from wavecast.models.batch_inference import BatchPredictor
from wavecast.models.wavelet_gpt import WaveletGPT


def _make_trained_model(n_samples=100, ctx_len=8, vocab=10, seed=42):
    """Create and train a small WaveletGPT for testing."""
    rng = np.random.default_rng(seed)
    contexts = rng.integers(0, vocab, size=(n_samples, ctx_len))
    levels = rng.integers(0, 5, size=(n_samples, 1))
    acs = rng.integers(0, 4, size=(n_samples, 1))
    X = np.column_stack([contexts, levels, acs]).astype(np.float64)
    y = rng.integers(0, vocab, size=n_samples).astype(np.float64)

    model = WaveletGPT(
        vocab_size=vocab,
        context_length=ctx_len,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        epochs=3,
        batch_size=32,
        n_levels=5,
        n_asset_classes=4,
    )
    model.fit(X[:80], y[:80])
    return model, X, y


class TestBatchVsFullEquivalence:
    def test_predict_matches(self):
        """Batch predict gives same results as full predict."""
        model, X, _ = _make_trained_model()
        bp = BatchPredictor(model, batch_size=16)

        full = model.predict(X[80:])
        batched = bp.predict_all(X[80:])
        np.testing.assert_array_equal(full, batched)

    def test_predict_proba_matches(self):
        """Batch predict_proba gives same results as full predict_proba."""
        model, X, _ = _make_trained_model()
        bp = BatchPredictor(model, batch_size=16)

        full = model.predict_proba(X[80:])
        batched = bp.predict_proba_all(X[80:])
        np.testing.assert_allclose(full, batched, atol=1e-5)


class TestStreaming:
    def test_stream_yields_chunks(self):
        """Stream yields multiple chunks for large input."""
        model, X, _ = _make_trained_model()
        bp = BatchPredictor(model, batch_size=5)

        chunks = list(bp.predict_stream(X[80:]))  # 20 samples / 5 = 4 chunks
        assert len(chunks) == 4
        for chunk in chunks:
            assert len(chunk) == 5

    def test_stream_last_batch_remainder(self):
        """Last batch handles remainder (< batch_size)."""
        model, X, _ = _make_trained_model()
        bp = BatchPredictor(model, batch_size=7)

        chunks = list(bp.predict_stream(X[80:]))  # 20 / 7 = 2 full + 1 remainder
        assert len(chunks) == 3
        assert len(chunks[0]) == 7
        assert len(chunks[1]) == 7
        assert len(chunks[2]) == 6  # remainder


class TestEdgeCases:
    def test_single_sample(self):
        """Works with a single sample."""
        model, X, _ = _make_trained_model()
        bp = BatchPredictor(model, batch_size=256)

        result = bp.predict_all(X[80:81])
        assert len(result) == 1

    def test_batch_size_larger_than_input(self):
        """Works when batch_size > n_samples."""
        model, X, _ = _make_trained_model()
        bp = BatchPredictor(model, batch_size=1000)

        result = bp.predict_all(X[80:])
        assert len(result) == 20

    def test_invalid_model_type_raises(self):
        """Non-WaveletGPT model raises TypeError."""
        with pytest.raises(TypeError):
            BatchPredictor("not a model")
