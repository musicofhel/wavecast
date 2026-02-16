"""Directional accuracy and baseline metrics for multi-level SAX experiments."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from wavecast.tokenizer.vocabulary import UNK_ID

if TYPE_CHECKING:
    from collections.abc import Callable

    from wavecast.tokenizer.vocabulary import SAXVocabulary


def level0_directional_accuracy(
    predicted_tokens: NDArray,
    actual_tokens: NDArray,
    vocabulary: SAXVocabulary,
) -> float:
    """Compute directional accuracy for level-0 (approximation) SAX tokens.

    Level 0 represents trend. Higher SAX symbol ordinal = higher value.
    Direction is determined by comparing predicted symbol vs current (last in context):
        predicted > current -> UP
        predicted < current -> DOWN
        predicted == current -> HOLD (excluded from accuracy)

    For detail levels (1-5), oscillation magnitude has no meaningful direction,
    so only token accuracy applies there.

    Args:
        predicted_tokens: Predicted token IDs, shape (n_samples,).
        actual_tokens: Actual (target) token IDs, shape (n_samples,).
        vocabulary: SAXVocabulary used to decode tokens back to SAX words.

    Returns:
        Directional accuracy as a float in [0, 1]. Returns 0.5 if no
        directional samples exist.
    """
    if len(predicted_tokens) == 0 or len(actual_tokens) == 0:
        return 0.5

    n = len(predicted_tokens)
    correct = 0
    total = 0

    for i in range(n):
        pred_id = int(predicted_tokens[i])
        actual_id = int(actual_tokens[i])

        # Skip PAD (0) and UNK (1) tokens
        if pred_id <= UNK_ID or actual_id <= UNK_ID:
            continue

        # Decode to SAX words and use ordinal comparison
        try:
            pred_word = vocabulary.decode(pred_id)
            actual_word = vocabulary.decode(actual_id)
        except Exception:
            continue

        # Direction: compare the words lexicographically
        # SAX symbols are 'a' < 'b' < ... — higher = higher value
        if pred_word == actual_word:
            # HOLD — both predicted same direction, count as correct
            correct += 1
            total += 1
        else:
            # Compare ordinals relative to vocabulary midpoint
            total += 1
            mid = _midpoint_word(vocabulary)
            pred_above = pred_word >= mid
            actual_above = actual_word >= mid
            if pred_above == actual_above:
                correct += 1

    if total == 0:
        return 0.5

    return correct / total


def _midpoint_word(vocabulary: SAXVocabulary) -> str:
    """Get the median word from the vocabulary for direction splitting."""
    words = vocabulary.words
    if not words:
        return ""
    words_sorted = sorted(words)
    return words_sorted[len(words_sorted) // 2]


def compute_bootstrap_ci(
    metric_fn: Callable[[NDArray, NDArray], float],
    predictions: NDArray,
    actuals: NDArray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        metric_fn: Function(predictions, actuals) -> float.
        predictions: Predicted values array.
        actuals: Actual values array.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95 for 95% CI).

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    n = len(predictions)
    if n == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed=42)
    scores = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores[b] = metric_fn(predictions[idx], actuals[idx])

    alpha = (1 - ci) / 2
    lower = float(np.percentile(scores, alpha * 100))
    upper = float(np.percentile(scores, (1 - alpha) * 100))
    return (lower, upper)


def compute_baselines(
    train_tokens: NDArray,
    test_contexts: NDArray,
    test_targets: NDArray,
) -> dict[str, float]:
    """Compute baseline accuracy metrics on the test set.

    Args:
        train_tokens: All token IDs from training data (1-D).
        test_contexts: Context windows from test set, shape (n_samples, context_length).
        test_targets: Target token IDs, shape (n_samples,).

    Returns:
        Dict with keys 'most_frequent', 'persistence', 'momentum'.
    """
    n = len(test_targets)
    if n == 0:
        return {"most_frequent": 0.0, "persistence": 0.0, "momentum": 0.0}

    # --- Most-frequent baseline ---
    # Predict the mode of the training tokens for every test sample
    counter = Counter(int(t) for t in train_tokens if int(t) > 1)  # skip PAD/UNK
    most_frequent_token = counter.most_common(1)[0][0] if counter else 2
    mf_correct = int(np.sum(test_targets == most_frequent_token))
    most_frequent_acc = mf_correct / n

    # --- Persistence baseline ---
    # Predict next token = last token in context window
    last_in_context = test_contexts[:, -1]
    persist_correct = int(np.sum(test_targets == last_in_context))
    persistence_acc = persist_correct / n

    # --- Momentum baseline ---
    # Predict same direction as majority direction in context window.
    # Direction: compare consecutive tokens in context.
    # If more ups than downs, predict a token > last; else predict token < last.
    momentum_correct = 0
    for i in range(n):
        ctx = test_contexts[i]
        # Count direction changes in context
        ups = 0
        downs = 0
        for j in range(1, len(ctx)):
            if ctx[j] > ctx[j - 1]:
                ups += 1
            elif ctx[j] < ctx[j - 1]:
                downs += 1
        # Majority direction
        if ups > downs:
            # Predict UP: target > last context token
            if test_targets[i] > ctx[-1]:
                momentum_correct += 1
        elif downs > ups:
            # Predict DOWN: target < last context token
            if test_targets[i] < ctx[-1]:
                momentum_correct += 1
        else:
            # Tied: predict persistence (same as last)
            if test_targets[i] == ctx[-1]:
                momentum_correct += 1

    momentum_acc = momentum_correct / n

    return {
        "most_frequent": most_frequent_acc,
        "persistence": persistence_acc,
        "momentum": momentum_acc,
    }
