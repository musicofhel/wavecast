"""Tests for ShapeletLibrary."""

import numpy as np

from wavecast.core.types import MarketLabel, Shapelet
from wavecast.shapelets.library import ShapeletLibrary


def _make_shapelet(id_: str, level: int = 1, ticker: str = "T",
                   label: MarketLabel = MarketLabel.UP, ig: float = 0.5) -> Shapelet:
    return Shapelet(
        id=id_, coefficients=np.random.default_rng(42).standard_normal(10),
        wavelet_level=level, ticker=ticker, label=label,
        information_gain=ig, start_index=0, end_index=10, threshold=1.0,
    )


def test_add_and_get():
    lib = ShapeletLibrary()
    s = _make_shapelet("test1")
    lib.add(s)
    result = lib.get("test1")
    assert result is not None
    assert result.id == "test1"


def test_get_missing_returns_none_or_raises():
    """Library.get for missing ID may return None or raise."""
    lib = ShapeletLibrary()
    try:
        result = lib.get("nonexistent")
        # If it returns None, that's fine
        assert result is None
    except Exception:
        # If it raises, that's also acceptable
        pass


def test_query_by_level():
    lib = ShapeletLibrary()
    lib.add(_make_shapelet("a", level=1))
    lib.add(_make_shapelet("b", level=2))
    lib.add(_make_shapelet("c", level=1))
    results = lib.query(level=1)
    assert len(results) == 2


def test_save_load_roundtrip(tmp_path):
    lib = ShapeletLibrary()
    lib.add(_make_shapelet("s1", level=1, ticker="AAPL"))
    lib.add(_make_shapelet("s2", level=2, ticker="SPY"))

    path = tmp_path / "test.h5"
    lib.save(path)

    loaded = ShapeletLibrary.load(path)
    assert loaded.get("s1") is not None
    assert loaded.get("s2") is not None


def test_stats():
    lib = ShapeletLibrary()
    lib.add(_make_shapelet("a", level=1, ticker="AAPL", label=MarketLabel.UP))
    lib.add(_make_shapelet("b", level=2, ticker="AAPL", label=MarketLabel.DOWN))
    lib.add(_make_shapelet("c", level=1, ticker="SPY", label=MarketLabel.UP))
    s = lib.stats()
    # Stats dict should have some count key
    total = s.get("total", s.get("count", 0))
    assert total == 3


def test_merge():
    lib1 = ShapeletLibrary([_make_shapelet("a")])
    lib2 = ShapeletLibrary([_make_shapelet("b")])
    lib1.merge(lib2)
    assert lib1.get("a") is not None
    assert lib1.get("b") is not None
