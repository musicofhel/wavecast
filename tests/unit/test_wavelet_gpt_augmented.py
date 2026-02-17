"""Tests for WaveletGPT with auxiliary features (n_aux_features > 0)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from wavecast.models.wavelet_gpt import WaveletGPT


def _make_augmented_data(
    n_samples: int = 100,
    context_length: int = 8,
    n_aux: int = 4,
    n_classes: int = 5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic X with auxiliary features and y labels.

    X layout: [ctx_0..ctx_{L-1}, aux_00..aux_{L-1}_{F-1}, level, asset_class]
    """
    if rng is None:
        rng = np.random.default_rng(42)

    ctx = rng.standard_normal((n_samples, context_length))
    aux = rng.standard_normal((n_samples, context_length * n_aux))
    levels = rng.integers(0, 3, size=(n_samples, 1))
    asset_classes = rng.integers(0, 5, size=(n_samples, 1))
    X = np.column_stack([ctx, aux, levels, asset_classes]).astype(np.float64)
    y = rng.integers(0, n_classes, size=n_samples).astype(np.float64)
    return X, y


def test_augmented_fit_predict():
    """Model with aux features trains and predicts without errors."""
    ctx_len = 8
    n_aux = 4
    X, y = _make_augmented_data(n_samples=80, context_length=ctx_len, n_aux=n_aux)
    X_val, y_val = _make_augmented_data(
        n_samples=20, context_length=ctx_len, n_aux=n_aux,
        rng=np.random.default_rng(99),
    )

    model = WaveletGPT(
        vocab_size=1,
        context_length=ctx_len,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        epochs=3,
        batch_size=16,
        task="return_quantile",
        n_output_classes=5,
        input_mode="continuous",
        n_aux_features=n_aux,
    )
    metrics = model.fit(X, y, X_val=X_val, y_val=y_val)
    assert "train_loss" in metrics
    assert "val_loss" in metrics

    preds = model.predict(X_val)
    assert preds.shape == (20,)
    assert np.all((preds >= 0) & (preds < 5))

    proba = model.predict_proba(X_val)
    assert proba.shape == (20, 5)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_augmented_save_load():
    """Model with aux features saves and loads correctly."""
    ctx_len = 8
    n_aux = 4
    X, y = _make_augmented_data(n_samples=50, context_length=ctx_len, n_aux=n_aux)

    model = WaveletGPT(
        vocab_size=1,
        context_length=ctx_len,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        epochs=2,
        batch_size=16,
        task="return_quantile",
        n_output_classes=5,
        input_mode="continuous",
        n_aux_features=n_aux,
    )
    model.fit(X, y)
    preds_before = model.predict(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model"
        model.save(save_path)

        loaded = WaveletGPT.load(save_path)
        assert loaded._n_aux_features == n_aux
        assert loaded._config["n_aux_features"] == n_aux

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)


def test_augmented_predict_all_horizons():
    """Multi-horizon prediction works with aux features."""
    ctx_len = 8
    n_aux = 4
    X, y = _make_augmented_data(n_samples=50, context_length=ctx_len, n_aux=n_aux)

    model = WaveletGPT(
        vocab_size=1,
        context_length=ctx_len,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        epochs=2,
        batch_size=16,
        task="return_quantile",
        n_output_classes=5,
        input_mode="continuous",
        n_aux_features=n_aux,
        prediction_horizons=[1, 2],
    )
    # Multi-horizon y: 2 columns
    rng = np.random.default_rng(42)
    y_multi = rng.integers(0, 5, size=(50, 2)).astype(np.float64)
    model.fit(X, y_multi)

    all_preds = model.predict_all_horizons(X)
    assert set(all_preds.keys()) == {1, 2}
    assert all_preds[1].shape == (50,)

    all_proba = model.predict_proba_all_horizons(X)
    assert set(all_proba.keys()) == {1, 2}
    assert all_proba[1].shape == (50, 5)


def test_zero_aux_backward_compat():
    """n_aux_features=0 (default) produces identical behavior to before."""
    ctx_len = 8
    rng = np.random.default_rng(42)
    X = np.column_stack([
        rng.standard_normal((50, ctx_len)),
        rng.integers(0, 3, size=(50, 1)),
        rng.integers(0, 5, size=(50, 1)),
    ]).astype(np.float64)
    y = rng.integers(0, 5, size=50).astype(np.float64)

    model = WaveletGPT(
        vocab_size=1,
        context_length=ctx_len,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        epochs=2,
        batch_size=16,
        task="return_quantile",
        n_output_classes=5,
        input_mode="continuous",
        # n_aux_features defaults to 0
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (50,)


def test_augmented_regression():
    """Aux features work with regression task too."""
    ctx_len = 8
    n_aux = 4
    rng = np.random.default_rng(42)
    ctx = rng.standard_normal((60, ctx_len))
    aux = rng.standard_normal((60, ctx_len * n_aux))
    levels = rng.integers(0, 3, size=(60, 1))
    acs = rng.integers(0, 5, size=(60, 1))
    X = np.column_stack([ctx, aux, levels, acs]).astype(np.float64)
    y = rng.standard_normal(60).astype(np.float64)

    model = WaveletGPT(
        vocab_size=1,
        context_length=ctx_len,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        epochs=2,
        batch_size=16,
        task="return_regression",
        input_mode="continuous",
        n_aux_features=n_aux,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (60,)
    # Regression: raw float outputs
    assert preds.dtype == np.float32 or preds.dtype == np.float64
