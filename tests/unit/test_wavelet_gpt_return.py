"""Tests for WaveletGPT return-target task support."""

import numpy as np
import pytest

from wavecast.models.wavelet_gpt import (
    TASK_RETURN_QUANTILE,
    TASK_RETURN_REGRESSION,
    TASK_TOKEN,
    WaveletGPT,
    WaveletGPTNet,
)


def _make_quantile_data(n_samples=200, context_length=8, vocab_size=20, n_classes=5, seed=42):
    """Generate data for return-quantile task."""
    rng = np.random.default_rng(seed)
    contexts = rng.integers(0, vocab_size, size=(n_samples, context_length))
    levels = rng.integers(0, 5, size=(n_samples, 1))
    asset_classes = rng.integers(0, 4, size=(n_samples, 1))
    X = np.column_stack([contexts, levels, asset_classes]).astype(np.float64)
    y = rng.integers(0, n_classes, size=n_samples).astype(np.float64)
    return X, y


def _make_regression_data(n_samples=200, context_length=8, vocab_size=20, seed=42):
    """Generate data for return-regression task."""
    rng = np.random.default_rng(seed)
    contexts = rng.integers(0, vocab_size, size=(n_samples, context_length))
    levels = rng.integers(0, 5, size=(n_samples, 1))
    asset_classes = rng.integers(0, 4, size=(n_samples, 1))
    X = np.column_stack([contexts, levels, asset_classes]).astype(np.float64)
    y = rng.normal(0, 0.01, size=n_samples).astype(np.float64)
    return X, y


# --- Quantile task tests ---


def test_quantile_fit_predict():
    """Quantile model trains and produces valid class predictions."""
    X, y = _make_quantile_data(n_samples=100, n_classes=5)
    model = WaveletGPT(
        vocab_size=20, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1, epochs=3, batch_size=32,
        n_levels=5, n_asset_classes=4,
        task=TASK_RETURN_QUANTILE, n_output_classes=5,
    )
    metrics = model.fit(X[:80], y[:80], X[80:], y[80:])
    assert "train_loss" in metrics
    assert metrics["train_loss"] >= 0

    preds = model.predict(X[80:])
    assert len(preds) == 20
    assert all(0 <= p < 5 for p in preds)


def test_quantile_predict_proba():
    """Quantile model produces valid probability distributions."""
    X, y = _make_quantile_data(n_samples=100, n_classes=5)
    model = WaveletGPT(
        vocab_size=20, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1, epochs=3, batch_size=32,
        n_levels=5, n_asset_classes=4,
        task=TASK_RETURN_QUANTILE, n_output_classes=5,
    )
    model.fit(X[:80], y[:80])
    proba = model.predict_proba(X[80:])
    assert proba.shape == (20, 5)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_quantile_save_load(tmp_path):
    """Save/load preserves quantile task config and predictions."""
    X, y = _make_quantile_data(n_samples=100, n_classes=5)
    model = WaveletGPT(
        vocab_size=20, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1, epochs=3, batch_size=32,
        n_levels=5, n_asset_classes=4,
        task=TASK_RETURN_QUANTILE, n_output_classes=5,
    )
    model.fit(X[:80], y[:80])

    model_path = tmp_path / "quantile_model"
    model.save(model_path)

    loaded = WaveletGPT.load(model_path)
    assert loaded.task == TASK_RETURN_QUANTILE
    preds_orig = model.predict(X[80:])
    preds_loaded = loaded.predict(X[80:])
    np.testing.assert_array_equal(preds_orig, preds_loaded)


def test_quantile_no_weight_tying():
    """Quantile task does NOT weight-tie h=1 head."""
    net = WaveletGPTNet(
        vocab_size=20, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1,
        task=TASK_RETURN_QUANTILE, n_output_classes=5,
    )
    # Output head should have 5 outputs, not 20 (vocab_size)
    assert net.heads["1"].weight.shape[0] == 5
    # Should NOT be tied to token embedding
    assert net.heads["1"].weight is not net.token_embed.weight


def test_quantile_class_weights():
    """Class weights are applied without errors."""
    X, y = _make_quantile_data(n_samples=100, n_classes=5)
    model = WaveletGPT(
        vocab_size=20, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1, epochs=3, batch_size=32,
        n_levels=5, n_asset_classes=4,
        task=TASK_RETURN_QUANTILE, n_output_classes=5,
        class_weights=[2.0, 1.0, 0.5, 1.0, 2.0],
    )
    metrics = model.fit(X[:80], y[:80])
    assert metrics["train_loss"] >= 0


# --- Regression task tests ---


def test_regression_fit_predict():
    """Regression model trains and produces float predictions."""
    X, y = _make_regression_data(n_samples=100)
    model = WaveletGPT(
        vocab_size=20, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1, epochs=3, batch_size=32,
        n_levels=5, n_asset_classes=4,
        task=TASK_RETURN_REGRESSION,
    )
    metrics = model.fit(X[:80], y[:80], X[80:], y[80:])
    assert "train_loss" in metrics

    preds = model.predict(X[80:])
    assert len(preds) == 20
    assert preds.dtype == np.float32 or preds.dtype == np.float64


def test_regression_save_load(tmp_path):
    """Save/load preserves regression task config."""
    X, y = _make_regression_data(n_samples=100)
    model = WaveletGPT(
        vocab_size=20, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1, epochs=3, batch_size=32,
        n_levels=5, n_asset_classes=4,
        task=TASK_RETURN_REGRESSION,
    )
    model.fit(X[:80], y[:80])

    model_path = tmp_path / "regression_model"
    model.save(model_path)

    loaded = WaveletGPT.load(model_path)
    assert loaded.task == TASK_RETURN_REGRESSION
    preds_orig = model.predict(X[80:])
    preds_loaded = loaded.predict(X[80:])
    np.testing.assert_allclose(preds_orig, preds_loaded, atol=1e-5)


def test_regression_output_dim():
    """Regression network has output_dim=1."""
    net = WaveletGPTNet(
        vocab_size=20, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1,
        task=TASK_RETURN_REGRESSION,
    )
    assert net.output_dim == 1
    assert net.heads["1"].weight.shape[0] == 1


# --- Backward compatibility tests ---


def test_default_task_is_token():
    """Default task is 'token' for backward compatibility."""
    model = WaveletGPT(vocab_size=20, context_length=8)
    assert model.task == TASK_TOKEN


def test_token_task_weight_tying():
    """Token task still weight-ties h=1 head."""
    net = WaveletGPTNet(
        vocab_size=20, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1,
        task=TASK_TOKEN,
    )
    assert net.heads["1"].weight is net.token_embed.weight


def test_invalid_task_raises():
    """Invalid task name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown task"):
        WaveletGPT(vocab_size=20, context_length=8, task="invalid")
