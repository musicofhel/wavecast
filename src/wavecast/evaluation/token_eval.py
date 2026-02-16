"""Token prediction evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class TokenPredictionMetrics:
    """Metrics for token sequence prediction."""
    token_accuracy: float
    top3_accuracy: float
    directional_accuracy: float
    confusion_matrix: NDArray[np.float64]


@dataclass
class MultiHorizonMetrics:
    """Metrics for multi-horizon token prediction."""
    per_horizon: dict[int, TokenPredictionMetrics] = field(default_factory=dict)

    @property
    def horizons(self) -> list[int]:
        return sorted(self.per_horizon.keys())


def evaluate_token_predictions(
    predicted: NDArray,
    actual: NDArray,
    vocab_size: int,
    proba: NDArray | None = None,
) -> TokenPredictionMetrics:
    """Evaluate token predictions against actual.

    Args:
        predicted: Array of predicted token IDs (n_samples,)
        actual: Array of actual token IDs (n_samples,)
        vocab_size: Size of vocabulary (for confusion matrix)
        proba: Optional probability matrix (n_samples, vocab_size) for top-k accuracy

    Returns:
        TokenPredictionMetrics with accuracy, top-3 accuracy, directional accuracy
    """
    n = len(predicted)
    if n == 0:
        return TokenPredictionMetrics(
            token_accuracy=0.0,
            top3_accuracy=0.0,
            directional_accuracy=0.0,
            confusion_matrix=np.zeros((vocab_size, vocab_size)),
        )

    # Token accuracy
    token_accuracy = float(np.mean(predicted == actual))

    # Top-3 accuracy (if probabilities available)
    if proba is not None and proba.ndim == 2:
        top3_ids = np.argsort(proba, axis=1)[:, -3:]  # top 3 per sample
        top3_hits = np.array([actual[i] in top3_ids[i] for i in range(n)])
        top3_accuracy = float(np.mean(top3_hits))
    else:
        top3_accuracy = token_accuracy  # fallback: same as top-1

    # Directional accuracy: map tokens to directions
    # Higher token IDs (above midpoint) -> UP (+1), lower -> DOWN (-1), middle -> FLAT (0)
    midpoint = vocab_size // 2
    quarter = vocab_size // 4

    def _to_direction(token_ids: NDArray) -> NDArray:
        dirs = np.zeros_like(token_ids, dtype=np.float64)
        dirs[token_ids >= midpoint + quarter] = 1.0   # UP
        dirs[token_ids <= midpoint - quarter] = -1.0   # DOWN
        return dirs

    pred_dirs = _to_direction(predicted)
    actual_dirs = _to_direction(actual)
    # Only measure where actual has a direction
    has_dir = actual_dirs != 0.0
    if np.any(has_dir):
        directional_accuracy = float(np.mean(pred_dirs[has_dir] == actual_dirs[has_dir]))
    else:
        directional_accuracy = 0.5  # no directional data -> chance

    # Confusion matrix
    cm_size = min(vocab_size, int(max(predicted.max(), actual.max())) + 1) if n > 0 else vocab_size
    cm = np.zeros((cm_size, cm_size), dtype=np.float64)
    for p, a in zip(predicted, actual, strict=True):
        if p < cm_size and a < cm_size:
            cm[a, p] += 1

    return TokenPredictionMetrics(
        token_accuracy=token_accuracy,
        top3_accuracy=top3_accuracy,
        directional_accuracy=directional_accuracy,
        confusion_matrix=cm,
    )


def evaluate_multi_horizon(
    predictions: dict[int, NDArray],
    actuals: dict[int, NDArray],
    vocab_size: int,
    probas: dict[int, NDArray] | None = None,
) -> MultiHorizonMetrics:
    """Evaluate predictions across multiple horizons.

    Args:
        predictions: Dict of {horizon: predicted_token_ids} arrays.
        actuals: Dict of {horizon: actual_token_ids} arrays.
        vocab_size: Size of vocabulary.
        probas: Optional dict of {horizon: probability_matrix} arrays.

    Returns:
        MultiHorizonMetrics with per-horizon TokenPredictionMetrics.
    """
    result = MultiHorizonMetrics()
    for h in sorted(predictions.keys()):
        proba_h = probas.get(h) if probas is not None else None
        result.per_horizon[h] = evaluate_token_predictions(
            predictions[h], actuals[h], vocab_size, proba_h
        )
    return result


def token_predictions_to_signals(
    predicted: NDArray,
    vocab_size: int,
    label_map: dict[int, int] | None = None,
) -> NDArray:
    """Convert token predictions to trading signals (+1, -1, 0).

    Default mapping: tokens in upper quartile -> +1 (buy),
    lower quartile -> -1 (sell), middle -> 0 (hold).

    Args:
        predicted: Array of predicted token IDs
        vocab_size: Total vocabulary size
        label_map: Optional explicit mapping {token_id: signal}

    Returns:
        Array of trading signals
    """
    if label_map is not None:
        return np.array([label_map.get(int(p), 0) for p in predicted], dtype=np.float64)

    midpoint = vocab_size // 2
    quarter = vocab_size // 4
    signals = np.zeros(len(predicted), dtype=np.float64)
    signals[predicted >= midpoint + quarter] = 1.0
    signals[predicted <= midpoint - quarter] = -1.0
    return signals
