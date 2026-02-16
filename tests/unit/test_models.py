"""Tests for forecasting models."""

import numpy as np

from tests.fixtures.generators import make_feature_matrix
from wavecast.models.gradient_boost import GradientBoostModel


def test_gradient_boost_fit_predict():
    X, y = make_feature_matrix(n=200, d=20)
    split = 160
    model = GradientBoostModel()
    metrics = model.fit(X[:split], y[:split], X[split:], y[split:])
    assert "train_rmse" in metrics
    preds = model.predict(X[split:])
    assert len(preds) == 40
    assert np.all(np.isfinite(preds))


def test_gradient_boost_save_load(tmp_path):
    X, y = make_feature_matrix(n=100, d=10)
    model = GradientBoostModel()
    model.fit(X[:80], y[:80])
    model.save(tmp_path / "xgb_model")

    loaded = GradientBoostModel.load(tmp_path / "xgb_model")
    preds_orig = model.predict(X[80:])
    preds_loaded = loaded.predict(X[80:])
    np.testing.assert_array_almost_equal(preds_orig, preds_loaded)


def test_wavelet_lstm_fit_predict():
    from wavecast.models.wavelet_lstm import WaveletLSTM

    X, y = make_feature_matrix(n=200, d=20)
    split = 160
    model = WaveletLSTM(
        branch_input_sizes=[5, 5, 5, 5],
        epochs=3,
        batch_size=32,
    )
    metrics = model.fit(X[:split], y[:split], X[split:], y[split:])
    assert "train_rmse" in metrics
    preds = model.predict(X[split:])
    assert len(preds) == 40
    assert np.all(np.isfinite(preds))


def test_ensemble_predict():
    from wavecast.models.ensemble import EnsembleModel

    X, y = make_feature_matrix(n=200, d=20)
    split = 160

    xgb = GradientBoostModel()
    xgb.fit(X[:split], y[:split])

    ensemble = EnsembleModel(models=[xgb], weights=[1.0])
    # Fit ensemble (some implementations require it)
    ensemble.fit(X[:split], y[:split])
    preds = ensemble.predict(X[split:])
    assert len(preds) == 40
