"""Tests for return target evaluation metrics."""

import numpy as np
import pytest

from wavecast.evaluation.return_eval import evaluate_return_predictions


def test_evaluate_perfect_predictions():
    """Perfect predictions yield 100% accuracy."""
    actual_classes = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    actual_returns = np.array([-0.05, -0.01, 0.0, 0.01, 0.05])
    predicted = actual_classes.copy()

    metrics = evaluate_return_predictions(predicted, actual_returns, actual_classes)
    assert metrics.quantile_accuracy == pytest.approx(1.0)
    assert metrics.directional_accuracy == pytest.approx(1.0)


def test_evaluate_random_predictions():
    """Random predictions yield ~20% accuracy for 5 classes."""
    rng = np.random.default_rng(42)
    n = 10000
    actual_classes = rng.integers(0, 5, n).astype(np.int64)
    actual_returns = rng.normal(0, 0.01, n)
    predicted = rng.integers(0, 5, n).astype(np.int64)

    metrics = evaluate_return_predictions(predicted, actual_returns, actual_classes)
    assert metrics.quantile_accuracy == pytest.approx(0.2, abs=0.03)


def test_evaluate_strong_signal_accuracy():
    """Strong signal accuracy only considers extreme classes."""
    # All predictions are class 4 (strong_up), actuals are all positive → 100% strong
    predicted = np.array([4, 4, 4, 4], dtype=np.int64)
    actual_returns = np.array([0.01, 0.02, 0.03, 0.04])
    actual_classes = np.array([3, 3, 4, 4], dtype=np.int64)

    metrics = evaluate_return_predictions(predicted, actual_returns, actual_classes)
    assert metrics.strong_signal_accuracy == pytest.approx(1.0)


def test_evaluate_economic_value():
    """Economic value is positive when directions are correct."""
    # Predict UP (class 3) when returns are positive
    predicted = np.array([3, 3, 1, 1], dtype=np.int64)
    actual_returns = np.array([0.01, 0.02, -0.01, -0.02])

    metrics = evaluate_return_predictions(predicted, actual_returns)
    assert metrics.economic_value > 0


def test_evaluate_empty():
    """Empty predictions return default metrics."""
    metrics = evaluate_return_predictions(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64),
    )
    assert metrics.quantile_accuracy == 0.0
    assert metrics.directional_accuracy == 0.5
