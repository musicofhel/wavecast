"""Tests for HDF5 shapelet storage."""

import numpy as np

from wavecast.core.types import MarketLabel, Shapelet
from wavecast.data.storage import load_shapelets_h5, save_shapelets_h5


def test_save_load_roundtrip(tmp_path):
    shapelets = [
        Shapelet(
            id="abc12345",
            coefficients=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            wavelet_level=3,
            ticker="AAPL",
            label=MarketLabel.UP,
            information_gain=0.15,
            start_index=10,
            end_index=15,
            threshold=1.0,
        ),
        Shapelet(
            id="def67890",
            coefficients=np.array([5.0, 4.0, 3.0]),
            wavelet_level=2,
            ticker="SPY",
            label=MarketLabel.DOWN,
            information_gain=0.10,
            start_index=20,
            end_index=23,
            threshold=0.8,
        ),
    ]

    path = tmp_path / "test_shapelets.h5"
    save_shapelets_h5(path, shapelets)
    loaded = load_shapelets_h5(path)

    assert len(loaded) == 2

    by_id = {s.id: s for s in loaded}
    s1 = by_id["abc12345"]
    assert s1.wavelet_level == 3
    assert s1.ticker == "AAPL"
    assert s1.label == MarketLabel.UP
    np.testing.assert_array_almost_equal(s1.coefficients, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_save_empty(tmp_path):
    path = tmp_path / "empty.h5"
    save_shapelets_h5(path, [])
    loaded = load_shapelets_h5(path)
    assert len(loaded) == 0
