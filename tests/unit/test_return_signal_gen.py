"""Tests for ReturnSignalGenerator."""

import numpy as np

from wavecast.signals.generator import ReturnSignalGenerator


def test_quantile_up_signal():
    """Classes above midpoint produce UP direction."""
    gen = ReturnSignalGenerator(task="return_quantile", n_classes=5)
    predicted = np.array([3, 4], dtype=np.int64)
    timestamps = np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]")

    signals = gen.generate(predicted, timestamps, ticker="AAPL")
    assert signals.signals[0].direction == 1  # class 3 → UP
    assert signals.signals[1].direction == 1  # class 4 → UP


def test_quantile_down_signal():
    """Classes below midpoint produce DOWN direction."""
    gen = ReturnSignalGenerator(task="return_quantile", n_classes=5)
    predicted = np.array([0, 1], dtype=np.int64)
    timestamps = np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]")

    signals = gen.generate(predicted, timestamps)
    assert signals.signals[0].direction == -1  # class 0 → DOWN
    assert signals.signals[1].direction == -1  # class 1 → DOWN


def test_quantile_flat_signal():
    """Midpoint class produces FLAT direction."""
    gen = ReturnSignalGenerator(task="return_quantile", n_classes=5)
    predicted = np.array([2], dtype=np.int64)
    timestamps = np.array(["2025-01-01"], dtype="datetime64[ns]")

    signals = gen.generate(predicted, timestamps)
    assert signals.signals[0].direction == 0  # class 2 → FLAT


def test_regression_direction():
    """Regression: positive prediction → UP, negative → DOWN."""
    gen = ReturnSignalGenerator(task="return_regression", n_classes=5)
    predicted = np.array([0.01, -0.02, 0.0], dtype=np.float64)
    timestamps = np.array(["2025-01-01", "2025-01-02", "2025-01-03"], dtype="datetime64[ns]")

    signals = gen.generate(predicted, timestamps)
    assert signals.signals[0].direction == 1   # positive → UP
    assert signals.signals[1].direction == -1  # negative → DOWN
    assert signals.signals[2].direction == 0   # zero → FLAT


def test_confidence_threshold():
    """Low confidence signals get direction=0."""
    gen = ReturnSignalGenerator(
        task="return_quantile", n_classes=5, confidence_threshold=0.8
    )
    predicted = np.array([4], dtype=np.int64)
    timestamps = np.array(["2025-01-01"], dtype="datetime64[ns]")
    # No probabilities → confidence from distance heuristic
    # class 4, mid=2, distance=2, max_distance=2 → confidence=1.0
    signals = gen.generate(predicted, timestamps)
    assert signals.signals[0].direction == 1  # confidence = 1.0 >= 0.8

    # Class 3: distance=1, confidence=0.5 < 0.8 → filtered
    predicted2 = np.array([3], dtype=np.int64)
    signals2 = gen.generate(predicted2, timestamps)
    assert signals2.signals[0].direction == 0  # filtered by threshold
