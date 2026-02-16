"""Tests for multi-asset universe."""

import pytest

from wavecast.core.types import AssetClass, Sector
from wavecast.core.universe import (
    DEFAULT_UNIVERSE,
    LEGACY_UNIVERSE,
    PHASE3_UNIVERSE,
    AssetSpec,
    Universe,
    get_universe,
)


def test_default_universe_size():
    assert len(DEFAULT_UNIVERSE) == 20


def test_default_universe_tickers():
    tickers = DEFAULT_UNIVERSE.tickers
    assert "AAPL" in tickers
    assert "NVDA" in tickers
    assert "GLD" in tickers
    assert "X:BTCUSD" not in tickers


def test_default_universe_is_phase3():
    assert DEFAULT_UNIVERSE is PHASE3_UNIVERSE


def test_by_class_equity():
    equities = DEFAULT_UNIVERSE.by_class(AssetClass.EQUITY)
    assert len(equities) == 16
    assert all(a.asset_class == AssetClass.EQUITY for a in equities)


def test_by_class_commodity():
    commodities = DEFAULT_UNIVERSE.by_class(AssetClass.COMMODITY)
    assert len(commodities) == 4


def test_by_sector_tech():
    tech = DEFAULT_UNIVERSE.by_sector(Sector.TECH)
    assert len(tech) == 5
    assert all(a.sector == Sector.TECH for a in tech)


def test_by_sector_finance():
    finance = DEFAULT_UNIVERSE.by_sector(Sector.FINANCE)
    assert len(finance) == 3


def test_by_sector_energy():
    energy = DEFAULT_UNIVERSE.by_sector(Sector.ENERGY)
    assert len(energy) == 3


def test_by_sector_healthcare():
    healthcare = DEFAULT_UNIVERSE.by_sector(Sector.HEALTHCARE)
    assert len(healthcare) == 3


def test_by_sector_broad_etf():
    broad = DEFAULT_UNIVERSE.by_sector(Sector.BROAD_ETF)
    assert len(broad) == 2


def test_by_sector_commodity_etf():
    commodity_etfs = DEFAULT_UNIVERSE.by_sector(Sector.COMMODITY_ETF)
    assert len(commodity_etfs) == 4


def test_get_universe_default():
    u = get_universe("default")
    assert u is DEFAULT_UNIVERSE


def test_get_universe_phase3():
    u = get_universe("phase3")
    assert u is PHASE3_UNIVERSE


def test_get_universe_legacy():
    u = get_universe("legacy")
    assert u is LEGACY_UNIVERSE


def test_get_universe_unknown():
    with pytest.raises(ValueError, match="Unknown universe"):
        get_universe("nonexistent")


def test_legacy_universe_size():
    assert len(LEGACY_UNIVERSE) == 19


def test_legacy_universe_has_crypto():
    tickers = LEGACY_UNIVERSE.tickers
    assert "X:BTCUSD" in tickers
    assert "X:ETHUSD" in tickers


def test_legacy_by_class_equity():
    equities = LEGACY_UNIVERSE.by_class(AssetClass.EQUITY)
    assert len(equities) == 8


def test_legacy_by_class_crypto():
    cryptos = LEGACY_UNIVERSE.by_class(AssetClass.CRYPTO)
    assert len(cryptos) == 4


def test_legacy_by_class_forex():
    forex = LEGACY_UNIVERSE.by_class(AssetClass.FOREX)
    assert len(forex) == 3


def test_legacy_by_class_commodity():
    commodities = LEGACY_UNIVERSE.by_class(AssetClass.COMMODITY)
    assert len(commodities) == 4


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


def test_phase3_universe_tickers():
    tickers = PHASE3_UNIVERSE.tickers
    for t in ["AAPL", "NVDA", "GS", "BAC", "CVX", "COP", "JNJ", "UNH", "PFE"]:
        assert t in tickers
    assert "X:BTCUSD" not in tickers


def test_asset_spec_sector_optional():
    a = AssetSpec("TEST", "Test", AssetClass.EQUITY)
    assert a.sector is None

    b = AssetSpec("TEST", "Test", AssetClass.EQUITY, Sector.TECH)
    assert b.sector == Sector.TECH
