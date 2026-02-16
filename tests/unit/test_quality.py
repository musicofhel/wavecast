"""Tests for shapelet quality metrics."""

import numpy as np

from wavecast.shapelets.quality import entropy, f_statistic, information_gain


def test_entropy_uniform():
    """Uniform distribution should have maximum entropy."""
    labels = np.array(["a", "b", "c", "d"] * 25)
    e = entropy(labels)
    assert e > 0
    # Entropy of uniform over 4 classes = ln(4) ≈ 1.386 (natural log)
    np.testing.assert_almost_equal(e, np.log(4), decimal=5)


def test_entropy_pure():
    """Pure distribution should have zero entropy."""
    labels = np.array(["a"] * 100)
    e = entropy(labels)
    assert e == 0.0


def test_information_gain_nonnegative():
    rng = np.random.default_rng(42)
    values = rng.standard_normal(100)
    labels = rng.choice(["up", "down"], size=100)
    ig = information_gain(values, labels, 0.0)
    assert ig >= 0.0


def test_f_statistic_basic():
    rng = np.random.default_rng(42)
    values = np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)])
    labels = np.array(["a"] * 50 + ["b"] * 50)
    f = f_statistic(values, labels)
    assert f > 0  # Should be large since groups are well separated
