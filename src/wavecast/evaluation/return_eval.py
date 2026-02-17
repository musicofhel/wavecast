"""Return target prediction evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class ReturnTargetMetrics:
    """Metrics for return-target prediction.

    Attributes:
        quantile_accuracy: Fraction of correctly predicted quantile classes.
        directional_accuracy: Fraction where predicted direction matches
            sign(actual_return). Direction: classes 3,4→UP, class 2→FLAT (excluded),
            classes 0,1→DOWN.
        strong_signal_accuracy: Accuracy for strong_up (4) and strong_down (0) only.
        mean_return_per_class: Average actual return for each predicted class.
        confusion_matrix: (n_classes, n_classes) confusion matrix.
        economic_value: Mean of direction * actual_return (positive = profitable).
    """

    quantile_accuracy: float
    directional_accuracy: float
    strong_signal_accuracy: float
    mean_return_per_class: dict[int, float] = field(default_factory=dict)
    confusion_matrix: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((5, 5))
    )
    economic_value: float = 0.0


def evaluate_return_predictions(
    predicted_classes: NDArray[np.int64],
    actual_returns: NDArray[np.float64],
    actual_classes: NDArray[np.int64] | None = None,
    n_classes: int = 5,
    probs: NDArray[np.float64] | None = None,
) -> ReturnTargetMetrics:
    """Evaluate return-target predictions.

    Args:
        predicted_classes: Predicted quantile class IDs (n_samples,).
        actual_returns: Actual returns per sample (n_samples,).
        actual_classes: Actual quantile class IDs (n_samples,). Required for
            quantile accuracy and confusion matrix.
        n_classes: Number of quantile classes (default 5).
        probs: Optional softmax probabilities (n_samples, n_classes).

    Returns:
        ReturnTargetMetrics with all computed metrics.
    """
    n = len(predicted_classes)
    if n == 0:
        return ReturnTargetMetrics(
            quantile_accuracy=0.0,
            directional_accuracy=0.5,
            strong_signal_accuracy=0.5,
        )

    # Filter out NaN returns
    valid = ~np.isnan(actual_returns)
    pred_valid = predicted_classes[valid]
    ret_valid = actual_returns[valid]
    n_valid = int(valid.sum())

    # --- Quantile accuracy ---
    if actual_classes is not None:
        actual_valid = actual_classes[valid]
        quantile_accuracy = float(np.mean(pred_valid == actual_valid))

        # Confusion matrix
        cm = np.zeros((n_classes, n_classes), dtype=np.float64)
        for p, a in zip(pred_valid, actual_valid, strict=True):
            if 0 <= p < n_classes and 0 <= a < n_classes:
                cm[a, p] += 1
    else:
        quantile_accuracy = 0.0
        cm = np.zeros((n_classes, n_classes), dtype=np.float64)

    # --- Directional accuracy ---
    # Predicted direction: classes 3,4 → UP (+1), class 2 → FLAT (0), classes 0,1 → DOWN (-1)
    mid_class = n_classes // 2  # 2 for 5-class
    pred_dir = np.zeros(n_valid, dtype=np.float64)
    pred_dir[pred_valid > mid_class] = 1.0
    pred_dir[pred_valid < mid_class] = -1.0

    actual_dir = np.sign(ret_valid)

    # Only measure where actual has a direction (nonzero return)
    has_dir = actual_dir != 0.0
    if np.any(has_dir):
        directional_accuracy = float(
            np.mean(pred_dir[has_dir] == actual_dir[has_dir])
        )
    else:
        directional_accuracy = 0.5

    # --- Strong signal accuracy ---
    # Only for classes 0 (strong_down) and n_classes-1 (strong_up)
    strong_mask = (pred_valid == 0) | (pred_valid == n_classes - 1)
    if np.any(strong_mask & has_dir):
        strong_pred_dir = pred_dir[strong_mask & has_dir]
        strong_actual_dir = actual_dir[strong_mask & has_dir]
        strong_signal_accuracy = float(
            np.mean(strong_pred_dir == strong_actual_dir)
        )
    else:
        strong_signal_accuracy = 0.5

    # --- Mean return per predicted class ---
    mean_return_per_class: dict[int, float] = {}
    for c in range(n_classes):
        mask = pred_valid == c
        if np.any(mask):
            mean_return_per_class[c] = float(np.mean(ret_valid[mask]))

    # --- Economic value ---
    # Mean of predicted_direction * actual_return
    economic_value = float(np.mean(pred_dir * ret_valid))

    return ReturnTargetMetrics(
        quantile_accuracy=quantile_accuracy,
        directional_accuracy=directional_accuracy,
        strong_signal_accuracy=strong_signal_accuracy,
        mean_return_per_class=mean_return_per_class,
        confusion_matrix=cm,
        economic_value=economic_value,
    )
