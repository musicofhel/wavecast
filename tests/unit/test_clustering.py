"""Tests for shapelet clustering/deduplication."""

import numpy as np

from wavecast.core.types import MarketLabel, Shapelet
from wavecast.shapelets.clustering import deduplicate


def test_deduplicate_removes_near_duplicates():
    """Near-identical shapelets should be deduplicated."""
    base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    shapelets = [
        Shapelet(
            id="a", coefficients=base.copy(), wavelet_level=1,
            ticker="T", label=MarketLabel.UP, information_gain=0.5,
            start_index=0, end_index=5, threshold=1.0,
        ),
        Shapelet(
            id="b", coefficients=base + 0.001, wavelet_level=1,
            ticker="T", label=MarketLabel.UP, information_gain=0.3,
            start_index=5, end_index=10, threshold=1.0,
        ),
    ]
    result = deduplicate(shapelets, threshold=0.1)
    # Should keep only one (the higher IG one)
    assert len(result) == 1
    assert result[0].id == "a"


def test_deduplicate_keeps_different():
    """Different shapelets should be kept."""
    shapelets = [
        Shapelet(
            id="a", coefficients=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            wavelet_level=1, ticker="T", label=MarketLabel.UP,
            information_gain=0.5, start_index=0, end_index=5, threshold=1.0,
        ),
        Shapelet(
            id="b", coefficients=np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
            wavelet_level=1, ticker="T", label=MarketLabel.DOWN,
            information_gain=0.3, start_index=5, end_index=10, threshold=1.0,
        ),
    ]
    result = deduplicate(shapelets, threshold=0.01)
    assert len(result) == 2
