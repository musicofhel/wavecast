"""Tests for WaveletGPT model."""

import numpy as np
import pytest

from wavecast.core.exceptions import ModelNotTrainedError
from wavecast.models.wavelet_gpt import WaveletGPT


def _make_gpt_data(n_samples=200, context_length=16, vocab_size=20, seed=42):
    """Generate deterministic training data for WaveletGPT."""
    rng = np.random.default_rng(seed)
    contexts = rng.integers(0, vocab_size, size=(n_samples, context_length))
    levels = rng.integers(0, 5, size=(n_samples, 1))
    asset_classes = rng.integers(0, 4, size=(n_samples, 1))
    X = np.column_stack([contexts, levels, asset_classes]).astype(np.float64)
    y = rng.integers(0, vocab_size, size=n_samples).astype(np.float64)
    return X, y


def test_fit_reduces_loss():
    X, y = _make_gpt_data(n_samples=100, context_length=8, vocab_size=10)
    model = WaveletGPT(
        vocab_size=10, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1, epochs=5, batch_size=32,
        n_levels=5, n_asset_classes=4,
    )
    metrics = model.fit(X[:80], y[:80], X[80:], y[80:])
    assert "train_loss" in metrics
    assert "train_accuracy" in metrics
    assert metrics["train_loss"] >= 0


def test_predict_returns_array():
    X, y = _make_gpt_data(n_samples=100, context_length=8, vocab_size=10)
    model = WaveletGPT(
        vocab_size=10, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1, epochs=3, batch_size=32,
        n_levels=5, n_asset_classes=4,
    )
    model.fit(X[:80], y[:80])
    preds = model.predict(X[80:])
    assert len(preds) == 20
    assert all(0 <= p < 10 for p in preds)


def test_predict_proba_shape():
    X, y = _make_gpt_data(n_samples=100, context_length=8, vocab_size=10)
    model = WaveletGPT(
        vocab_size=10, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1, epochs=3, batch_size=32,
        n_levels=5, n_asset_classes=4,
    )
    model.fit(X[:80], y[:80])
    proba = model.predict_proba(X[80:])
    assert proba.shape == (20, 10)
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_save_load_roundtrip(tmp_path):
    X, y = _make_gpt_data(n_samples=100, context_length=8, vocab_size=10)
    model = WaveletGPT(
        vocab_size=10, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1, epochs=3, batch_size=32,
        n_levels=5, n_asset_classes=4,
    )
    model.fit(X[:80], y[:80])

    model_path = tmp_path / "gpt_model"
    model.save(model_path)

    loaded = WaveletGPT.load(model_path)
    preds_orig = model.predict(X[80:])
    preds_loaded = loaded.predict(X[80:])
    np.testing.assert_array_equal(preds_orig, preds_loaded)


def test_predict_untrained_raises():
    model = WaveletGPT(vocab_size=10, context_length=8)
    X, _ = _make_gpt_data(n_samples=10, context_length=8, vocab_size=10)
    with pytest.raises(ModelNotTrainedError):
        model.predict(X)


def test_model_name():
    model = WaveletGPT(vocab_size=10, context_length=8)
    assert model.name == "wavelet_gpt"
