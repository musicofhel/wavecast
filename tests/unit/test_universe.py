"""Tests for multi-asset universe."""

import pytest

from wavecast.core.types import AssetClass
from wavecast.core.universe import DEFAULT_UNIVERSE, AssetSpec, Universe, get_universe


def test_default_universe_size():
    assert len(DEFAULT_UNIVERSE) == 19


def test_default_universe_tickers():
    tickers = DEFAULT_UNIVERSE.tickers
    assert "AAPL" in tickers
    assert "X:BTCUSD" in tickers
    assert "GLD" in tickers


def test_by_class_equity():
    equities = DEFAULT_UNIVERSE.by_class(AssetClass.EQUITY)
    assert len(equities) == 8
    assert all(a.asset_class == AssetClass.EQUITY for a in equities)


def test_by_class_crypto():
    cryptos = DEFAULT_UNIVERSE.by_class(AssetClass.CRYPTO)
    assert len(cryptos) == 4


def test_by_class_forex():
    forex = DEFAULT_UNIVERSE.by_class(AssetClass.FOREX)
    assert len(forex) == 3


def test_by_class_commodity():
    commodities = DEFAULT_UNIVERSE.by_class(AssetClass.COMMODITY)
    assert len(commodities) == 4


def test_get_universe_default():
    u = get_universe("default")
    assert u is DEFAULT_UNIVERSE


def test_get_universe_unknown():
    with pytest.raises(ValueError, match="Unknown universe"):
        get_universe("nonexistent")


def test_custom_universe():
    u = Universe(
        name="test",
        assets=[
            AssetSpec("T1", "Test1", AssetClass.EQUITY),
            AssetSpec("T2", "Test2", AssetClass.CRYPTO),
        ],
    )
    assert len(u) == 2
    assert u.tickers == ["T1", "T2"]
