"""Tests for experiment metrics: directional accuracy, bootstrap CI, baselines."""

import numpy as np

from wavecast.experiments.metrics import (
    compute_baselines,
    compute_bootstrap_ci,
    level0_directional_accuracy,
)
from wavecast.tokenizer.vocabulary import SAXVocabulary


def _make_vocabulary() -> SAXVocabulary:
    """Build a small vocabulary from a synthetic corpus."""
    # alphabet=5, word_length=2 -> words like "aa", "ab", ..., "ee"
    words = []
    for a in "abcde":
        for b in "abcde":
            words.append(a + b)
    # Need at least 2 occurrences for min_freq=1
    corpus = [words, words]
    return SAXVocabulary.from_corpus(corpus, min_freq=1, max_size=500)


def test_directional_accuracy_perfect():
    vocab = _make_vocabulary()
    # Predict exactly the actuals -> all directions match
    targets = np.array([vocab.encode("cc")] * 10)
    acc = level0_directional_accuracy(targets, targets, vocab)
    assert acc == 1.0


def test_directional_accuracy_empty():
    vocab = _make_vocabulary()
    acc = level0_directional_accuracy(np.array([]), np.array([]), vocab)
    assert acc == 0.5  # default for empty


def test_directional_accuracy_bounded():
    vocab = _make_vocabulary()
    rng = np.random.default_rng(42)
    words = vocab.words
    ids = [vocab.encode(w) for w in words]
    pred = rng.choice(ids, size=50)
    actual = rng.choice(ids, size=50)
    acc = level0_directional_accuracy(pred, actual, vocab)
    assert 0.0 <= acc <= 1.0


def test_bootstrap_ci_returns_bounds():
    rng = np.random.default_rng(42)
    preds = rng.integers(0, 5, size=100)
    actuals = rng.integers(0, 5, size=100)

    def accuracy(p, a):
        return float(np.mean(p == a))

    lo, hi = compute_bootstrap_ci(accuracy, preds, actuals)
    assert lo <= hi
    # Point estimate should be within CI
    point = accuracy(preds, actuals)
    assert lo <= point <= hi


def test_bootstrap_ci_empty():
    lo, hi = compute_bootstrap_ci(lambda p, a: 0.0, np.array([]), np.array([]))
    assert lo == 0.0
    assert hi == 0.0


def test_bootstrap_ci_narrow_for_large_sample():
    rng = np.random.default_rng(42)
    n = 1000
    preds = rng.integers(0, 2, size=n)
    actuals = preds.copy()  # perfect predictions -> CI should be tight around 1.0

    def accuracy(p, a):
        return float(np.mean(p == a))

    lo, hi = compute_bootstrap_ci(accuracy, preds, actuals)
    assert lo >= 0.99
    assert hi == 1.0


def test_baselines_most_frequent():
    train_tokens = np.array([2, 2, 2, 3, 3, 4])  # mode = 2
    test_contexts = np.array([[2, 3, 2], [3, 2, 3], [2, 2, 2]])
    test_targets = np.array([2, 3, 2])  # 2 of 3 match mode=2
    baselines = compute_baselines(train_tokens, test_contexts, test_targets)
    assert baselines["most_frequent"] == pytest.approx(2 / 3)


def test_baselines_persistence():
    train_tokens = np.array([2, 3, 4])
    test_contexts = np.array([[2, 3, 5], [3, 2, 4]])
    test_targets = np.array([5, 4])  # last in context = [5, 4]
    baselines = compute_baselines(train_tokens, test_contexts, test_targets)
    assert baselines["persistence"] == 1.0  # both match last-in-context


def test_baselines_empty():
    baselines = compute_baselines(np.array([]), np.array([]).reshape(0, 3), np.array([]))
    assert baselines["most_frequent"] == 0.0
    assert baselines["persistence"] == 0.0
    assert baselines["momentum"] == 0.0


def test_baselines_keys():
    train_tokens = np.array([2, 3])
    test_contexts = np.array([[2, 3]])
    test_targets = np.array([3])
    baselines = compute_baselines(train_tokens, test_contexts, test_targets)
    assert set(baselines.keys()) == {"most_frequent", "persistence", "momentum"}


# Need pytest for approx
import pytest  # noqa: E402
