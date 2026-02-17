"""Integration tests for the return-target prediction pipeline.

End-to-end: synthetic prices → DWT → SAX → return targets → train → predict → evaluate.
"""

import numpy as np

from wavecast.evaluation.return_eval import evaluate_return_predictions
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.targets.returns import (
    assign_quantile_labels,
    compute_quantile_boundaries,
    compute_sample_returns,
)


def _make_synthetic_prices(n=500, seed=42) -> np.ndarray:
    """Generate synthetic prices with trends."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0001, 0.005, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return prices


def test_return_target_train_predict_cycle():
    """Full cycle: compute returns → quantile targets → train → predict."""
    prices = _make_synthetic_prices(200)

    # Simulate token positions for level 1 (2^1 = 2 bar span)
    n_samples = 100
    token_positions = np.arange(n_samples, dtype=np.int64)
    levels = np.ones(n_samples, dtype=np.int64)
    n_coeffs = np.full(n_samples, 100, dtype=np.int64)
    n_symbols = np.full(n_samples, 100, dtype=np.int64)

    # Compute returns
    returns, valid = compute_sample_returns(
        token_positions, levels, prices, n_coeffs, n_symbols
    )
    assert valid.sum() > 50

    # Compute boundaries from valid returns
    boundaries = compute_quantile_boundaries(
        returns, levels, valid, per_level=True
    )

    # Assign labels
    labels = assign_quantile_labels(returns, levels, boundaries)
    assert labels.max() <= 4
    assert labels.min() >= 0

    # Create training data (synthetic context tokens)
    rng = np.random.default_rng(42)
    context_length = 8
    vocab_size = 20
    contexts = rng.integers(0, vocab_size, size=(n_samples, context_length))
    level_ids = levels.reshape(-1, 1)
    asset_class_ids = np.zeros((n_samples, 1), dtype=np.int64)
    X = np.column_stack([contexts, level_ids, asset_class_ids]).astype(np.float64)
    y = labels.astype(np.float64)

    # Train
    model = WaveletGPT(
        vocab_size=vocab_size,
        context_length=context_length,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        epochs=5,
        batch_size=32,
        n_levels=5,
        n_asset_classes=4,
        task="return_quantile",
        n_output_classes=5,
    )
    train_metrics = model.fit(X[:80], y[:80], X[80:], y[80:])
    assert train_metrics["train_loss"] >= 0

    # Predict
    predicted = model.predict(X[80:])
    proba = model.predict_proba(X[80:])
    assert predicted.shape == (20,)
    assert proba.shape == (20, 5)

    # Evaluate
    metrics = evaluate_return_predictions(
        predicted.astype(np.int64),
        returns[80:],
        labels[80:],
        n_classes=5,
    )
    assert 0.0 <= metrics.quantile_accuracy <= 1.0
    assert 0.0 <= metrics.directional_accuracy <= 1.0


def test_regression_task_pipeline():
    """Return regression task completes end-to-end."""
    prices = _make_synthetic_prices(200)

    n_samples = 100
    token_positions = np.arange(n_samples, dtype=np.int64)
    levels = np.ones(n_samples, dtype=np.int64)
    n_coeffs = np.full(n_samples, 100, dtype=np.int64)
    n_symbols = np.full(n_samples, 100, dtype=np.int64)

    returns, valid = compute_sample_returns(
        token_positions, levels, prices, n_coeffs, n_symbols
    )

    rng = np.random.default_rng(42)
    context_length = 8
    vocab_size = 20
    contexts = rng.integers(0, vocab_size, size=(n_samples, context_length))
    level_ids = levels.reshape(-1, 1)
    asset_class_ids = np.zeros((n_samples, 1), dtype=np.int64)
    X = np.column_stack([contexts, level_ids, asset_class_ids]).astype(np.float64)
    y = np.where(valid, returns, 0.0).astype(np.float64)

    model = WaveletGPT(
        vocab_size=vocab_size,
        context_length=context_length,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        epochs=5,
        batch_size=32,
        n_levels=5,
        n_asset_classes=4,
        task="return_regression",
    )
    train_metrics = model.fit(X[:80], y[:80], X[80:], y[80:])
    assert "train_loss" in train_metrics

    predicted = model.predict(X[80:])
    assert predicted.shape == (20,)
    # Regression outputs are floats, not class IDs
    assert predicted.dtype in (np.float32, np.float64)


def test_save_load_return_model(tmp_path):
    """Return-quantile model save/load roundtrip in full pipeline."""
    rng = np.random.default_rng(42)
    n, ctx_len, vocab = 100, 8, 20
    contexts = rng.integers(0, vocab, size=(n, ctx_len))
    levels = rng.integers(0, 5, size=(n, 1))
    ac = rng.integers(0, 4, size=(n, 1))
    X = np.column_stack([contexts, levels, ac]).astype(np.float64)
    y = rng.integers(0, 5, size=n).astype(np.float64)

    model = WaveletGPT(
        vocab_size=vocab, context_length=ctx_len, embed_dim=16,
        num_heads=2, num_layers=1, epochs=3, batch_size=32,
        n_levels=5, n_asset_classes=4,
        task="return_quantile", n_output_classes=5,
    )
    model.fit(X[:80], y[:80])

    path = tmp_path / "return_model"
    model.save(path)
    loaded = WaveletGPT.load(path)

    assert loaded.task == "return_quantile"
    orig_preds = model.predict(X[80:])
    loaded_preds = loaded.predict(X[80:])
    np.testing.assert_array_equal(orig_preds, loaded_preds)
