"""Tests for token prediction evaluation."""

import numpy as np

from wavecast.evaluation.token_eval import (
    evaluate_token_predictions,
    token_predictions_to_signals,
)


def test_perfect_predictions():
    actual = np.array([2, 5, 8, 3, 7])
    predicted = np.array([2, 5, 8, 3, 7])
    metrics = evaluate_token_predictions(predicted, actual, vocab_size=10)
    assert metrics.token_accuracy == 1.0


def test_zero_accuracy():
    actual = np.array([0, 1, 2, 3, 4])
    predicted = np.array([5, 6, 7, 8, 9])
    metrics = evaluate_token_predictions(predicted, actual, vocab_size=10)
    assert metrics.token_accuracy == 0.0


def test_top3_accuracy_with_proba():
    actual = np.array([2, 5])
    predicted = np.array([3, 5])  # First wrong, second correct
    proba = np.zeros((2, 10))
    # For first sample: actual=2 is in top 3
    proba[0, 3] = 0.4
    proba[0, 2] = 0.3
    proba[0, 1] = 0.2
    # For second sample: actual=5 is in top 3
    proba[1, 5] = 0.5
    proba[1, 4] = 0.3
    proba[1, 3] = 0.2

    metrics = evaluate_token_predictions(predicted, actual, vocab_size=10, proba=proba)
    assert metrics.top3_accuracy == 1.0


def test_confusion_matrix_shape():
    rng = np.random.default_rng(42)
    actual = rng.integers(0, 5, size=100)
    predicted = rng.integers(0, 5, size=100)
    metrics = evaluate_token_predictions(predicted, actual, vocab_size=10)
    assert metrics.confusion_matrix.shape[0] >= 5
    assert metrics.confusion_matrix.shape[1] >= 5


def test_empty_predictions():
    metrics = evaluate_token_predictions(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        vocab_size=10,
    )
    assert metrics.token_accuracy == 0.0


def test_signals_buy_sell_hold():
    # vocab_size=20: midpoint=10, quarter=5
    # >= 15 -> buy (+1), <= 5 -> sell (-1), 6-14 -> hold (0)
    predicted = np.array([0, 5, 10, 15, 19])
    signals = token_predictions_to_signals(predicted, vocab_size=20)
    assert signals[0] == -1.0   # sell
    assert signals[1] == -1.0   # sell (edge)
    assert signals[2] == 0.0    # hold
    assert signals[3] == 1.0    # buy (edge)
    assert signals[4] == 1.0    # buy


def test_signals_with_label_map():
    predicted = np.array([0, 1, 2])
    label_map = {0: -1, 1: 0, 2: 1}
    signals = token_predictions_to_signals(predicted, vocab_size=10, label_map=label_map)
    np.testing.assert_array_equal(signals, [-1, 0, 1])


def test_directional_accuracy():
    actual = np.array([18, 18, 2, 2])     # 18 -> UP, 2 -> DOWN
    predicted = np.array([19, 15, 0, 3])  # Same directions
    metrics = evaluate_token_predictions(predicted, actual, vocab_size=20)
    assert metrics.directional_accuracy == 1.0
