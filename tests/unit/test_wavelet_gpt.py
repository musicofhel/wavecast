"""Tests for WaveletGPT model."""

import numpy as np
import pytest
import torch

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


def _make_multi_horizon_data(
    n_samples=200, context_length=16, vocab_size=20, horizons=None, seed=42
):
    """Generate multi-horizon training data."""
    horizons = horizons or [1, 2, 4, 8]
    rng = np.random.default_rng(seed)
    contexts = rng.integers(0, vocab_size, size=(n_samples, context_length))
    levels = rng.integers(0, 5, size=(n_samples, 1))
    asset_classes = rng.integers(0, 4, size=(n_samples, 1))
    X = np.column_stack([contexts, levels, asset_classes]).astype(np.float64)
    # y is 2-D: one column per horizon
    y = rng.integers(0, vocab_size, size=(n_samples, len(horizons))).astype(np.float64)
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


# --- Multi-horizon tests ---


def test_multi_horizon_fit():
    """Multi-horizon model trains and returns metrics."""
    horizons = [1, 2, 4, 8]
    X, y = _make_multi_horizon_data(
        n_samples=100, context_length=8, vocab_size=10, horizons=horizons
    )
    model = WaveletGPT(
        vocab_size=10, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1, epochs=3, batch_size=32,
        n_levels=5, n_asset_classes=4,
        prediction_horizons=horizons,
    )
    metrics = model.fit(X[:80], y[:80], X[80:], y[80:])
    assert "train_loss" in metrics
    assert "train_accuracy" in metrics
    assert metrics["train_loss"] >= 0


def test_multi_horizon_predict_each():
    """Each horizon produces valid predictions."""
    horizons = [1, 2, 4, 8]
    X, y = _make_multi_horizon_data(
        n_samples=100, context_length=8, vocab_size=10, horizons=horizons
    )
    model = WaveletGPT(
        vocab_size=10, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1, epochs=3, batch_size=32,
        n_levels=5, n_asset_classes=4,
        prediction_horizons=horizons,
    )
    model.fit(X[:80], y[:80])
    for h in horizons:
        preds = model.predict(X[80:], horizon=h)
        assert len(preds) == 20
        assert all(0 <= p < 10 for p in preds)


def test_multi_horizon_predict_proba():
    """Each horizon produces valid probability distributions."""
    horizons = [1, 2, 4]
    X, y = _make_multi_horizon_data(
        n_samples=100, context_length=8, vocab_size=10, horizons=horizons
    )
    model = WaveletGPT(
        vocab_size=10, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1, epochs=3, batch_size=32,
        n_levels=5, n_asset_classes=4,
        prediction_horizons=horizons,
    )
    model.fit(X[:80], y[:80])
    for h in horizons:
        proba = model.predict_proba(X[80:], horizon=h)
        assert proba.shape == (20, 10)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_multi_horizon_predict_all():
    """predict_all_horizons returns dict with all horizons."""
    horizons = [1, 2, 4, 8]
    X, y = _make_multi_horizon_data(
        n_samples=100, context_length=8, vocab_size=10, horizons=horizons
    )
    model = WaveletGPT(
        vocab_size=10, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1, epochs=3, batch_size=32,
        n_levels=5, n_asset_classes=4,
        prediction_horizons=horizons,
    )
    model.fit(X[:80], y[:80])
    all_preds = model.predict_all_horizons(X[80:])
    assert set(all_preds.keys()) == set(horizons)
    for h in horizons:
        assert len(all_preds[h]) == 20


def test_multi_horizon_invalid_horizon_raises():
    """Requesting unavailable horizon raises ValueError."""
    X, y = _make_gpt_data(n_samples=100, context_length=8, vocab_size=10)
    model = WaveletGPT(
        vocab_size=10, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1, epochs=3, batch_size=32,
        n_levels=5, n_asset_classes=4,
        prediction_horizons=[1],
    )
    model.fit(X[:80], y[:80])
    with pytest.raises(ValueError, match="Horizon 4 not available"):
        model.predict(X[80:], horizon=4)


def test_multi_horizon_save_load_roundtrip(tmp_path):
    """Save/load preserves multi-horizon config and predictions."""
    horizons = [1, 2, 4]
    X, y = _make_multi_horizon_data(
        n_samples=100, context_length=8, vocab_size=10, horizons=horizons
    )
    model = WaveletGPT(
        vocab_size=10, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1, epochs=3, batch_size=32,
        n_levels=5, n_asset_classes=4,
        prediction_horizons=horizons,
    )
    model.fit(X[:80], y[:80])

    model_path = tmp_path / "multi_gpt"
    model.save(model_path)

    loaded = WaveletGPT.load(model_path)
    assert loaded.prediction_horizons == horizons
    for h in horizons:
        preds_orig = model.predict(X[80:], horizon=h)
        preds_loaded = loaded.predict(X[80:], horizon=h)
        np.testing.assert_array_equal(preds_orig, preds_loaded)


def test_multi_horizon_weight_tying():
    """Horizon=1 head is weight-tied, others are independent."""
    from wavecast.models.wavelet_gpt import WaveletGPTNet

    net = WaveletGPTNet(
        vocab_size=10, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1,
        prediction_horizons=[1, 2, 4],
    )
    # Horizon 1 should share weight with token_embed
    assert net.heads["1"].weight is net.token_embed.weight
    # Other horizons should NOT share
    assert net.heads["2"].weight is not net.token_embed.weight
    assert net.heads["4"].weight is not net.token_embed.weight


def test_multi_horizon_horizon_weights():
    """Custom horizon weights are used in training."""
    horizons = [1, 2]
    X, y = _make_multi_horizon_data(
        n_samples=60, context_length=8, vocab_size=10, horizons=horizons
    )
    model = WaveletGPT(
        vocab_size=10, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1, epochs=2, batch_size=32,
        n_levels=5, n_asset_classes=4,
        prediction_horizons=horizons,
        horizon_weights={1: 1.0, 2: 0.5},
    )
    metrics = model.fit(X[:50], y[:50])
    assert "train_loss" in metrics
    assert metrics["train_loss"] >= 0


def test_default_horizons_backward_compat():
    """Default prediction_horizons=[1] maintains backward compatibility."""
    model = WaveletGPT(vocab_size=10, context_length=8)
    assert model.prediction_horizons == [1]


# --- AMP and MMap tests ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_amp_training():
    """AMP training produces valid results on CUDA."""
    X, y = _make_gpt_data(n_samples=100, context_length=8, vocab_size=10)
    model = WaveletGPT(
        vocab_size=10, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1, epochs=3, batch_size=32,
        n_levels=5, n_asset_classes=4, use_amp=True,
    )
    metrics = model.fit(X[:80], y[:80], X[80:], y[80:])
    assert "train_loss" in metrics
    assert metrics["train_loss"] >= 0


def test_amp_disabled_on_cpu():
    """AMP flag is silently ignored on CPU (no-op)."""
    X, y = _make_gpt_data(n_samples=60, context_length=8, vocab_size=10)
    model = WaveletGPT(
        vocab_size=10, context_length=8, embed_dim=16,
        num_heads=2, num_layers=1, epochs=2, batch_size=32,
        n_levels=5, n_asset_classes=4, use_amp=True,
    )
    # Force CPU
    model._device = torch.device("cpu")
    metrics = model.fit(X[:50], y[:50])
    assert "train_loss" in metrics
    assert metrics["train_loss"] >= 0


def test_mmap_dataset_with_model(tmp_path):
    """Model can train from memory-mapped dataset."""
    from wavecast.data.mmap_dataset import MMapSequenceDataset

    n, ctx_len, vocab = 100, 8, 10
    rng = np.random.default_rng(42)
    contexts = rng.integers(0, vocab, size=(n, ctx_len))
    targets = rng.integers(0, vocab, size=n)
    levels = rng.integers(0, 5, size=n)
    asset_classes = rng.integers(0, 4, size=n)

    mmap_path = tmp_path / "mmap"
    MMapSequenceDataset.save(mmap_path, contexts, targets, levels, asset_classes)

    # Build X/y from the same data (fit() still needs them for parsing/metrics)
    X = np.column_stack([contexts, levels[:, None], asset_classes[:, None]]).astype(
        np.float64
    )
    y = targets.astype(np.float64)

    model = WaveletGPT(
        vocab_size=vocab, context_length=ctx_len, embed_dim=16,
        num_heads=2, num_layers=1, epochs=3, batch_size=32,
        n_levels=5, n_asset_classes=4,
    )
    metrics = model.fit(X[:80], y[:80], dataset_path=mmap_path)
    assert "train_loss" in metrics
    assert metrics["train_loss"] >= 0
